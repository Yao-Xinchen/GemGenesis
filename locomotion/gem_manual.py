import numpy as np
import genesis as gs
import time
from pynput import keyboard

from gem_ackermann import GemAckermann

class Controller:
    def __init__(self):
        self.running = True
        self.pressed_keys = set()
        self.steering_sens = 0.2
        self.speed_sens = 10.0
        self.debug = False

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
    gs.init(backend=gs.cpu)
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

    joint_names = pos_joints + vel_joints
    joint_idx = pos_idx + vel_idx

    # Set the gains for the joints
    gem.set_dofs_kp(
        kp=np.array([35., 35., 0., 0., 0., 0.]),
        dofs_idx_local=joint_idx,
    )
    gem.set_dofs_kv(
        kv=np.array([5., 5., 30., 30., 30., 30.]),
        dofs_idx_local=joint_idx,
    )

    # Create the locomotion controller
    locomotion = GemAckermann(wheel_diameter=0.59, wheel_base=1.75, steer_dist_half=0.6)

    # Create the controller
    controller = Controller()
    print(
        "Controls:\n"
        "↑: Forward\n"
        "↓: Backward\n"
        "←: Left\n"
        "→: Right\n"
        "ESC: Quit"
    )

    # Listen for keyboard events
    listener = keyboard.Listener(
        on_press=controller.on_press,
        on_release=controller.on_release,
    )
    listener.start()

    while controller.running:
        try:
            steering, speed = controller.update()
            outputs = locomotion.control(steering, speed)
            gem.control_dofs_position(outputs[:2], pos_idx)
            gem.control_dofs_velocity(outputs[2:], vel_idx)

            scene.step()
            time.sleep(1 / 60)  # Limit simulation rate
        except Exception as e:
            print(f"Error in simulation loop: {e}")

    if scene.viewer:
        scene.viewer.stop()
    listener.stop()

if __name__ == "__main__":
    main()