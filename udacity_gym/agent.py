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
        # TODO: the image should never be none (by design)
        if observation.input_image is None:
            return UdacityAction(steering_angle=0.0, throttle=0.0)

        for callback in self.before_action_callbacks:
            callback(observation)

        action = self.action(observation, *args, **kwargs)

        for callback in self.after_action_callbacks:
            callback(observation)

        return action


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
