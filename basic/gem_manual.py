import torch
import genesis as gs
import time
from pynput import keyboard

from basic.gem_ackermann import GemAckermann


class ManualController:
    def __init__(self):
        self.running = True
        self.pressed_keys = set()
        self.steering_sens = 0.2
        self.speed_sens = 10.0
        self.debug = False

        # Listen for keyboard events
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self.listener.start()

    def __del__(self):
        self.listener.stop()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
            if self.debug:
                print(f"Key pressed: {key}")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
            if self.debug:
                print(f"Key released: {key}")
        except KeyError:
            pass

    def update(self):
        speed = 0
        steering = 0
        if keyboard.Key.up in self.pressed_keys:
            speed += self.speed_sens
        if keyboard.Key.down in self.pressed_keys:
            speed -= self.speed_sens
        if keyboard.Key.left in self.pressed_keys:
            steering += self.steering_sens
        if keyboard.Key.right in self.pressed_keys:
            steering -= self.steering_sens
        return steering, speed


def main():
    # Use CUDA if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gs.init(backend=gs.cuda if torch.cuda.is_available() else gs.cpu)

    # Create the scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    gem = scene.add_entity(
        gs.morphs.URDF(
            file="assets/gem/urdf/gem.urdf",
        ),
    )

    # scene.viewer.follow_entity(gem)
    scene.build()

    # Set the gains for the steering and velocity joints
    pos_joints = [
        "left_steering_hinge_joint",
        "right_steering_hinge_joint",
    ]
    pos_idx = [gem.get_joint(name).dof_idx_local for name in pos_joints]

    vel_joints = [
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ]
    vel_idx = [gem.get_joint(name).dof_idx_local for name in vel_joints]

    joint_idx = pos_idx + vel_idx

    kp_tensor = torch.tensor([35., 35., 0., 0., 0., 0.], device=device)
    kv_tensor = torch.tensor([5., 5., 30., 30., 30., 30.], device=device)

    # Set the gains for the joints
    gem.set_dofs_kp(kp=kp_tensor, dofs_idx_local=joint_idx)
    gem.set_dofs_kv(kv=kv_tensor, dofs_idx_local=joint_idx)

    # Create the locomotion controller
    locomotion = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6, device=device)

    # Pre-allocate tensors for better performance
    steering = torch.zeros(1, device=device)
    speed = torch.zeros(1, device=device)

    # Create the controller
    controller = ManualController()
    print(
        "Controls:\n"
        "↑: Forward\n"
        "↓: Backward\n"
        "←: Left\n"
        "→: Right\n"
        "ESC: Quit"
    )

    try:
        while controller.running:
            # Get control inputs
            steering_val, speed_val = controller.update()

            steering[0] = steering_val
            speed[0] = speed_val

            outputs = locomotion.control(steering, speed)

            gem.control_dofs_position(outputs[0, :2], pos_idx)
            gem.control_dofs_velocity(outputs[0, 2:], vel_idx)

            scene.step()
            time.sleep(1 / 60)
    except Exception as e:
        print(f"Error in simulation loop: {e}")
    finally:
        if scene.viewer:
            scene.viewer.stop()


if __name__ == "__main__":
    main()
