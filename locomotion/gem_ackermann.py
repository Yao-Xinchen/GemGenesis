import torch
from enum import IntEnum


def get_steer_angle(phi):
    mask = phi >= 0.0
    result = torch.where(mask, (torch.pi / 2) - phi, (-torch.pi / 2) - phi)
    return result


class GemAckermann:
    class DOFs(IntEnum):
        left_steering = 0
        right_steering = 1
        left_front_wheel = 2
        right_front_wheel = 3
        left_rear_wheel = 4
        right_rear_wheel = 5

    def __init__(self, wheel_diameter, wheel_base, steer_dist_half, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wheel_diameter = torch.tensor(wheel_diameter, device=self.device)
        self.wheel_inv_circ = 1.0 / (torch.pi * self.wheel_diameter)
        self.wheel_base = torch.tensor(wheel_base, device=self.device)
        self.wheel_base_inv = 1.0 / (self.wheel_base * 1.)
        self.wheel_base_sq = self.wheel_base ** 2
        self.steer_dist_half = torch.tensor(steer_dist_half, device=self.device)
        self.center_y = None  # Will be initialized with batch size

    def control(self, steering, velocity):
        """
        Control the steering and velocity of the vehicle.
        :param steering: The steering angle in radians [batch_size]
        :param velocity: The velocity of the wheels in m/s [batch_size]
        :return: The steering and velocity of the wheels [batch_size, 6]
        """
        batch_size = steering.shape[0]
        if self.center_y is None or self.center_y.shape[0] != batch_size:
            self.center_y = torch.zeros(batch_size, device=self.device)
            self.outputs = torch.zeros((batch_size, len(self.DOFs)), device=self.device)

        self.control_steering(steering)
        self.control_wheels(velocity)
        return self.outputs

    def control_steering(self, steering):
        """
        Control the steering of the front wheels,
        overwriting center_y.
        :param steering: The steering angle in radians [batch_size]
        """
        steering = torch.clamp(steering, -torch.pi / 2, torch.pi / 2)
        self.center_y = self.wheel_base * torch.tan((torch.pi / 2) - steering)

        self.outputs[:, self.DOFs.left_steering] = get_steer_angle(
            torch.atan(self.wheel_base_inv * (self.center_y - self.steer_dist_half)))
        self.outputs[:, self.DOFs.right_steering] = get_steer_angle(
            torch.atan(self.wheel_base_inv * (self.center_y + self.steer_dist_half)))

    def control_wheels(self, velocity):
        """
        Control the velocity of the wheels.
        :param velocity: The velocity of the wheels in m/s [batch_size]
        """
        left_dist = self.center_y - self.steer_dist_half
        right_dist = self.center_y + self.steer_dist_half

        gain = (2 * torch.pi) * velocity / torch.abs(self.center_y)
        r = torch.sqrt(left_dist ** 2 + self.wheel_base_sq)
        self.outputs[:, self.DOFs.left_front_wheel] = gain * r * self.wheel_inv_circ
        r = torch.sqrt(right_dist ** 2 + self.wheel_base_sq)
        self.outputs[:, self.DOFs.right_front_wheel] = gain * r * self.wheel_inv_circ
        gain = (2 * torch.pi) * velocity / self.center_y
        self.outputs[:, self.DOFs.left_rear_wheel] = gain * left_dist * self.wheel_inv_circ
        self.outputs[:, self.DOFs.right_rear_wheel] = gain * right_dist * self.wheel_inv_circ


if __name__ == "__main__":
    gem_ackermann = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6)
    # Test with batch size of 3
    ste = torch.tensor([0.2, 0.3, 0.1], device=gem_ackermann.device)
    vel = torch.tensor([1.0, 1.5, 2.0], device=gem_ackermann.device)
    outputs = gem_ackermann.control(steering=ste, velocity=vel)
    print(outputs)
