import logging
import pathlib
import time
from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
# import pygame
# import torch
# import torchvision

from .action import UdacityAction
from .simulator import UdacitySimulator
from .observation import UdacityObservation


class UdacityAgent:

    def __init__(self, before_action_callbacks=None, after_action_callbacks=None):
        self.after_action_callbacks = after_action_callbacks
        self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
        self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []

    def on_before_action(self, observation: UdacityObservation):
        for callback in self.before_action_callbacks:
            callback(observation)

    def on_after_action(self, observation: UdacityObservation):
        for callback in self.after_action_callbacks:
            callback(observation)

    def action(self, observation: UdacityObservation, *args, **kwargs):
        raise NotImplementedError('UdacityAgent does not implement __call__')

    def __call__(self, observation: UdacityObservation, *args, **kwargs):
        return self.action(observation, *args, **kwargs)


class PIDUdacityAgent(UdacityAgent):

    def __init__(self, kp, kd, ki, before_action_callbacks=None, after_action_callbacks=None):
        super().__init__(before_action_callbacks, after_action_callbacks)
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain
        self.ki = ki  # Integral gain
        self.prev_error = 0.0
        self.total_error = 0.0

    def action(self, observation: UdacityObservation, *args, **kwargs):
        error = (observation.next_cte + observation.cte) / 2
        diff_err = error - self.prev_error

        # Calculate steering angle
        steering_angle = - (self.kp * error) - (self.kd * diff_err) - (self.ki * self.total_error)
        steering_angle = max(-1, min(steering_angle, 1))

        # Calculate throttle
        throttle = 0.5

        # Save error for next prediction
        self.total_error += error
        self.total_error = self.total_error * 0.99
        self.prev_error = error

        return UdacityAction(steering_angle=steering_angle, throttle=throttle)

# class LaneKeepingAgent:
#
#     def __init__(self, model, before_action_callbacks=None, after_action_callbacks=None, transform_callbacks=None,
#                  shadow_mode: bool = True):
#         self.after_action_callbacks = after_action_callbacks
#         self.model = model
#         self.shadow_mode = shadow_mode
#         self.before_action_callbacks = before_action_callbacks if before_action_callbacks is not None else []
#         self.after_action_callbacks = after_action_callbacks if after_action_callbacks is not None else []
#         self.transform_callbacks = transform_callbacks if transform_callbacks is not None else []
#
#     def action(self, observation: UdacityObservation) -> UdacityAction:
#         if observation.input_image is None:
#             return UdacityAction(steering_angle=0.0, throttle=0.0)
#         for callback in self.before_action_callbacks:
#             callback(observation)
#         if self.shadow_mode:
#             shadow_prediction = self.model(
#                 torchvision.transforms.ToTensor()(observation.input_image).to(DEFAULT_DEVICE))
#             shadow_action = UdacityAction(steering_angle=shadow_prediction.item() * 1.4, throttle=0.2)
#         for callback in self.transform_callbacks:
#             callback(observation)
#         prediction = self.model(torchvision.transforms.ToTensor()(observation.input_image).to(DEFAULT_DEVICE))
#         action = UdacityAction(steering_angle=prediction.item() * 1.4, throttle=0.2)
#         for callback in self.after_action_callbacks:
#             if self.shadow_mode:
#                 callback(observation, action=action, shadow_action=shadow_action)
#             else:
#                 callback(observation, action=action)
#         return action
#
#     def __call__(self, observation: UdacityObservation):
#         return self.action(observation)
#
#
# class AgentCallback:
#
#     def __init__(self, name: str, verbose: bool = False):
#         self.name = name
#         self.verbose = verbose
#
#     def __call__(self, observation: UdacityObservation, *args, **kwargs):
#         if self.verbose:
#             logging.getLogger(str(self.__class__)).info(f"Activating callback {self.name}")
#
#
# class PauseSimulationCallback(AgentCallback):
#
#     def __init__(self, simulator: UdacitySimulator):
#         super().__init__('stop_simulation')
#         self.simulator = simulator
#
#     def __call__(self, observation: UdacityObservation, *args, **kwargs):
#         super().__call__(observation, *args, **kwargs)
#         self.simulator.pause()
#
#
# class ResumeSimulationCallback(AgentCallback):
#
#     def __init__(self, simulator: UdacitySimulator):
#         super().__init__('resume_simulation')
#         self.simulator = simulator
#
#     def __call__(self, observation: UdacityObservation, *args, **kwargs):
#         super().__call__(observation, *args, **kwargs)
#         self.simulator.resume()
#
#
# class LogObservationCallback(AgentCallback):
#
#     def __init__(self, path, enable_pygame_logging=False):
#         super().__init__('log_observation')
#         self.path = pathlib.Path(path)
#         self.path.mkdir(parents=True, exist_ok=True)
#         self.logs = []
#         self.logging_file = self.path.joinpath('log.csv')
#         self.enable_pygame_logging = enable_pygame_logging
#         if self.enable_pygame_logging:
#             pygame.init()
#             self.screen = pygame.display.set_mode((320, 160))
#             camera_surface = pygame.surface.Surface((320, 160), 0, 24).convert()
#             self.screen.blit(camera_surface, (0, 0))
#
#     def __call__(self, observation: UdacityObservation, *args, **kwargs):
#         super().__call__(observation, *args, **kwargs)
#         metrics = observation.get_metrics()
#         image_name = f"frame_{observation.time:020d}.jpg"
#         torchvision.utils.save_image(
#             tensor=torchvision.transforms.ToTensor()(observation.input_image),
#             fp=self.path.joinpath(image_name)
#         )
#         metrics['input_image'] = image_name
#         if 'action' in kwargs.keys():
#             metrics['predicted_steering_angle'] = kwargs['action'].steering_angle
#             metrics['predicted_throttle'] = kwargs['action'].throttle
#         if 'shadow_action' in kwargs.keys():
#             metrics['shadow_predicted_steering_angle'] = kwargs['shadow_action'].steering_angle
#             metrics['shadow_predicted_throttle'] = kwargs['shadow_action'].throttle
#         self.logs.append(metrics)
#
#         if self.enable_pygame_logging:
#             pixel_array = np.swapaxes(observation.input_image, 0, 1)
#             new_surface = pygame.pixelcopy.make_surface(pixel_array)
#             self.screen.blit(new_surface, (0, 0))
#             pygame.display.flip()
#
#     def save(self):
#         logging_dataframe = pd.DataFrame(self.logs)
#         logging_dataframe = logging_dataframe.set_index('time', drop=True)
#         logging_dataframe.to_csv(self.logging_file)
#         if self.enable_pygame_logging:
#             pygame.quit()
#
#
# class TransformObservationCallback(AgentCallback):
#
#     def __init__(self, transformation: Callable):
#         super().__init__('transform_observation')
#         self.transformation = transformation
#
#     def __call__(self, observation: UdacityObservation, *args, **kwargs):
#         super().__call__(observation, *args, **kwargs)
#         augmented_image: torch.Tensor = self.transformation(
#             torchvision.transforms.ToTensor()(observation.input_image).to(DEFAULT_DEVICE)
#         )
#         observation.input_image = (augmented_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
#         # return observation
