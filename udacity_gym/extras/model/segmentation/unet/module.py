import torch
import torchinfo
from torch import nn


class DownSampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_groups: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class UpSampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_groups: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.up_sample(x)
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x


class UnetEncoder(nn.Module):

    def __init__(self, hidden_dims: list[int] = [32, 64, 128, 256, 384], num_groups: int = 32, in_channels: int = 3,
                 input_shape: tuple[int, int, int] = (3, 512, 512), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_dims = hidden_dims
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.input_shape = input_shape
        self.n_down_conv = len(hidden_dims)

        linear_projection_modules = []
        linear_projection_modules += [
            nn.Conv2d(in_channels, hidden_dims[0], 7, padding=3, padding_mode="reflect"),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dims[0], eps=1e-6, affine=True),
            nn.Conv2d(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=2, stride=2),
            nn.SiLU(inplace=True),
        ]

        for i in range(self.n_down_conv - 1):
            linear_projection_modules += [
                DownSampleBlock(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1], num_groups=num_groups),
            ]

        self.linear_projection = nn.Sequential(*linear_projection_modules)
        self.flatten = nn.Flatten(2)

    def forward(self, x: torch.Tensor):
        x = self.linear_projection(x)
        x = self.flatten(x).transpose(1, 2)
        return x


class UnetDecoder(nn.Module):

    def __init__(self, hidden_dims: list[int] = [32, 64, 128, 256, 384][::-1], num_groups: int = 32,
                 use_tanh: bool = True,
                 out_channels: int = 3, input_shape: tuple[int, int, int] = (3, 512, 512), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_dims = hidden_dims
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.input_shape = input_shape
        self.n_up_conv = len(hidden_dims)
        self.use_tanh = use_tanh

        linear_projection_modules = []
        for i in range(self.n_up_conv - 1):
            linear_projection_modules += [
                UpSampleBlock(in_channels=hidden_dims[i], out_channels=hidden_dims[i + 1],
                              num_groups=num_groups),
            ]

        linear_projection_modules += [
            nn.Upsample(scale_factor=2),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dims[-1], eps=1e-6, affine=True),
            nn.Conv2d(in_channels=hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=hidden_dims[-1], eps=1e-6, affine=True),
            nn.Conv2d(hidden_dims[-1], out_channels, 7, padding=3, padding_mode="reflect"),
        ]

        if self.use_tanh:
            linear_projection_modules += [nn.Tanh()]

        self.linear_projection = nn.Sequential(*linear_projection_modules)

        self.unflatten = nn.Unflatten(
            dim=2, unflattened_size=(
                input_shape[1] // (2 ** (self.n_up_conv)), input_shape[2] // (2 ** (self.n_up_conv)))
        )

    def forward(self, x: torch.Tensor, output_shape: tuple[int, int] = (512, 512)):
        x = self.unflatten(x.transpose(1, 2))
        x = self.linear_projection(x)
        return x

class PositionalEncoder(nn.Module):

    def __init__(self, dim: int, seq_len: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.seq_len = seq_len

        self.encoding = nn.Parameter(torch.rand(1, self.seq_len, self.dim))

    def forward(self, x: torch.Tensor):
        return x + self.encoding

class ClassEncoder(nn.Module):

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

        self.class_token = nn.Parameter(torch.rand(1, 1, self.dim))
        self.encoding = nn.Parameter(torch.rand(1))

    def forward(self, x: torch.Tensor):
        return torch.cat([self.class_token] * x.shape[0]), x + self.encoding

class TimestepEncoder(nn.Module):

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, dim),
        )

    def forward(self, x: torch.Tensor):
        return self.timestep_encoder(x.reshape(x.shape[0], 1)).reshape(x.shape[0], 1, self.dim)

if __name__ == '__main__':
    hidden_dims = [8, 32, 128, 512]
    # hidden_dims = [64, 128, 256, 512]
    input_shape = (1, 512, 512)
    # hidden_dims = [32, 384]
    # input_shape = (4, 64, 64)
    num_groups = 8

    batch_size = 8
    patch_size = (2 ** len(hidden_dims), 2 ** len(hidden_dims))
    seq_len = (input_shape[1] * input_shape[2]) // (patch_size[0] * patch_size[1])

    model = UnetEncoder(hidden_dims=hidden_dims, num_groups=num_groups, input_shape=input_shape,
                        in_channels=input_shape[0])
    torchinfo.summary(model, (batch_size, input_shape[0], input_shape[1], input_shape[2]),
                      col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5)

    model = nn.Transformer(d_model=hidden_dims[-1], nhead=16, num_encoder_layers=12, dim_feedforward=2048,
                           batch_first=True).to("cuda:0")
    torchinfo.summary(model.encoder, (batch_size, seq_len, hidden_dims[-1]),
                      col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5)
    torchinfo.summary(model, input_data={
        'src': torch.rand(batch_size, seq_len * 3, hidden_dims[-1]).to("cuda:0"),
        'tgt': torch.rand(batch_size, seq_len, hidden_dims[-1]).to("cuda:0"),
    }, col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5)
    # torchinfo.summary(model.decoder, (1, seq_len, hidden_dims[-1]), col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5)

    model = UnetDecoder(hidden_dims=hidden_dims[::-1], num_groups=num_groups, input_shape=input_shape,
                               out_channels=input_shape[0])
    torchinfo.summary(model, (batch_size, seq_len, hidden_dims[-1]),
                      col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=5)
