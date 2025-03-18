import numpy as np
from enum import IntEnum


def get_steer_angle(phi):
    if phi >= 0.0:
        return (np.pi / 2) - phi
    return (-np.pi / 2) - phi


class GemAckermann:
    class DOFs(IntEnum):
        left_steering = 0
        right_steering = 1
        left_front_wheel = 2
        right_front_wheel = 3
        left_rear_wheel = 4
        right_rear_wheel = 5

    def __init__(self, wheel_diameter, wheel_base, steer_dist_half):
        self.wheel_diameter = wheel_diameter
        self.wheel_inv_circ = 1.0 / (np.pi * wheel_diameter)
        self.wheel_base = wheel_base
        self.wheel_base_inv = 1.0 / (wheel_base * 1.)
        self.wheel_base_sq = wheel_base ** 2
        self.steer_dist_half = steer_dist_half
        self.center_y = 0

        self.outputs = np.zeros(len(self.DOFs))

    def control(self, steering, velocity):
        """
        Control the steering and velocity of the vehicle.
        :param steering: The steering angle in radians
        :param velocity: The velocity of the wheels in m/s
        :return: The steering and velocity of the wheels
        """
        self.control_steering(steering)
        self.control_wheels(velocity)
        return self.outputs

    def control_steering(self, steering):
        """
        Control the steering of the front wheels,
        overwriting center_y.
        :param steering: The steering angle in radians
        """
        steering = np.clip(steering, -np.pi / 2, np.pi / 2)
        self.center_y = self.wheel_base * np.tan((np.pi / 2) - steering)

        self.outputs[self.DOFs.left_steering] = get_steer_angle(
            np.arctan(self.wheel_base_inv * (self.center_y - self.steer_dist_half)))
        self.outputs[self.DOFs.right_steering] = get_steer_angle(
            np.arctan(self.wheel_base_inv * (self.center_y + self.steer_dist_half)))

    def control_wheels(self, velocity):
        """
        Control the velocity of the wheels.
        :param velocity: The velocity of the wheels in m/s
        """
        left_dist = self.center_y - self.steer_dist_half
        right_dist = self.center_y + self.steer_dist_half

        gain = (2 * np.pi) * velocity / abs(self.center_y)
        r = np.sqrt(left_dist ** 2 + self.wheel_base_sq)
        self.outputs[self.DOFs.left_front_wheel] = gain * r * self.wheel_inv_circ
        r = np.sqrt(right_dist ** 2 + self.wheel_base_sq)
        self.outputs[self.DOFs.right_front_wheel] = gain * r * self.wheel_inv_circ
        gain = (2 * np.pi) * velocity / self.center_y
        self.outputs[self.DOFs.left_rear_wheel] = gain * left_dist * self.wheel_inv_circ
        self.outputs[self.DOFs.right_rear_wheel] = gain * right_dist * self.wheel_inv_circ


if __name__ == "__main__":
    gem_ackermann = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6)
    outputs = gem_ackermann.control(steering=0.2, velocity=1.0)
    print(outputs)
