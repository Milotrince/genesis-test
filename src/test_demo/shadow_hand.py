import numpy as np
import genesis as gs
from gs_runner import GenesisRunner


class ShadowHandDemo(GenesisRunner):
    def setup(self):
        scene = self.scene

        self.plane = scene.add_entity(gs.morphs.Plane())

        self.target_1 = scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.05,
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
        )
        self.target_2 = scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.05,
            ),
            surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),
        )
        self.target_3 = scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.05,
            ),
            surface=gs.surfaces.Default(color=(0.5, 0.5, 1.0, 1)),
        )

        self.robot = scene.add_entity(
            morph=gs.morphs.URDF(
                scale=1.0,
                file="urdf/shadow_hand/shadow_hand.urdf",
                pos=(0.0, 0.0, 0.0),
            ),
            surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),
        )
        robot = self.robot

        self.target_quat = np.array([1, 0, 0, 0])
        self.index_finger_distal = robot.get_link("index_finger_distal")
        self.middle_finger_distal = robot.get_link("middle_finger_distal")
        # print("!! Links:", robot.links)
        # for link in robot.links:
        #     print("Link:", link.name)
        self.forearm = robot.get_link("wrist")

        self.center = np.array([0.5, 0.5, 0.2])
        self.r1 = 0.1
        self.r2 = 0.13

    def step(self, i):
        robot, center, r1, r2, target_quat = (
            self.robot,
            self.center,
            self.r1,
            self.r2,
            self.target_quat,
        )
        target_1, target_2, target_3 = self.target_1, self.target_2, self.target_3
        index_finger_distal, middle_finger_distal, forearm = (
            self.index_finger_distal,
            self.middle_finger_distal,
            self.forearm,
        )

        index_finger_pos = (
            center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r1
        )
        middle_finger_pos = (
            center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r2
        )
        forearm_pos = index_finger_pos - np.array([0, 0, 0.40])

        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))
        target_2.set_qpos(np.concatenate([middle_finger_pos, target_quat]))
        target_3.set_qpos(np.concatenate([forearm_pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal, middle_finger_distal, forearm],
            poss=[index_finger_pos, middle_finger_pos, forearm_pos],
        )
        robot.set_qpos(qpos)

    def on_complete(self):
        pass


if __name__ == "__main__":

    ShadowHandDemo(
        video_out="out/demo_shadow_hand.mp4",
        camera_pos=(2.5, 0.0, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
            enable_collision=False,
            enable_joint_limit=False,
        ),
    ).run(steps=1000)
