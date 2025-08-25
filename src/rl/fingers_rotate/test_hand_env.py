import argparse

import genesis as gs
import torch

FPS = 60


def get_motors_info(robot):
    motors_dof_idx = []
    motors_dof_name = []
    for joint in robot.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            continue
        elif joint.type == gs.JOINT_TYPE.FIXED:
            continue
        dofs_idx_local = robot.get_joint(joint.name).dofs_idx_local
        if dofs_idx_local:
            if len(dofs_idx_local) == 1:
                dofs_name = [joint.name]
            else:
                dofs_name = [f"{joint.name}_{i_d}" for i_d in dofs_idx_local]
            motors_dof_idx += dofs_idx_local
            motors_dof_name += dofs_name
    return motors_dof_idx, motors_dof_name


def parse_args():
    parser = argparse.ArgumentParser(description="Test hand environment with configurable parameters")

    # File and basic settings
    parser.add_argument(
        "--filename",
        type=str,
        default="/home/trinityc/.cache/huggingface/hub/datasets--Genesis-Intelligence--assets/snapshots/35183e88cce2b1ca1b2d0f777df71d4bb21f3331/allegro_hand/allegro_hand_right_glb.urdf",
        help="Path to the robot URDF/XML file",
    )
    parser.add_argument("--collision", action="store_true", help="Enable collision detection")
    parser.add_argument("--rotate", action="store_true", help="Enable rotation")
    parser.add_argument("--show-link-frame", action="store_true", help="Show link frames")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the robot")
    parser.add_argument("--n_envs", "-b", type=int, default=2048, help="Number of environments")
    parser.add_argument("--show-viewer", "-v", action="store_true", help="Show the viewer")
    parser.add_argument("--randomize", action="store_true", help="Add random perturbation to joint angles")

    # Hand configuration
    parser.add_argument(
        "--hand-init-pos", type=float, nargs=3, default=[0.0, 0.0, 0.5], help="Initial position of the hand (x, y, z)"
    )
    parser.add_argument(
        "--hand-init-euler",
        type=float,
        nargs=3,
        default=[0.0, -90.0, 0.0],
        help="Initial euler angles of the hand (roll, pitch, yaw in degrees)",
    )

    # Object configuration
    parser.add_argument(
        "--obj-base-pos", type=float, nargs=3, default=[0.03, 0.0, 0.62], help="Base position of the object (x, y, z)"
    )
    parser.add_argument(
        "--obj-init-euler",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Initial euler angles of the object (roll, pitch, yaw in degrees)",
    )
    parser.add_argument("--obj-morph", type=str, default="cylinder", help="Object morph type")
    parser.add_argument("--obj-size", type=float, default=0.04, help="Object size/radius")

    # Simulation settings
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -10.0], help="Gravity vector (x, y, z)")
    parser.add_argument("--camera-fov", type=float, default=40.0, help="Camera field of view")
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")

    return parser.parse_args()


def main():
    args = parse_args()

    from vis_hand_env import cfg

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=tuple(args.gravity),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(cfg["hand_init_pos"][0] + 0.3, cfg["hand_init_pos"][1] + 0.3, cfg["hand_init_pos"][2] + 0.7),
            camera_lookat=cfg["hand_init_pos"],
            camera_fov=args.camera_fov,
            max_FPS=FPS,
        ),
        vis_options=gs.options.VisOptions(
            show_link_frame=args.show_link_frame,
        ),
        show_viewer=args.show_viewer,
    )

    if args.filename.endswith(".urdf"):
        morph_cls = gs.morphs.URDF
    elif args.filename.endswith(".xml"):
        morph_cls = gs.morphs.MJCF
    else:
        morph_cls = gs.morphs.Mesh

    entity = scene.add_entity(
        morph_cls(
            file=args.filename,
            scale=args.scale,
            pos=cfg["hand_init_pos"],
            euler=cfg["hand_init_euler"],
            fixed=True,
            merge_fixed_links=False,
        ),
    )

    obj = scene.add_entity(
        gs.morphs.Cylinder(
            radius=cfg["obj_size"],
            height=0.08,
            pos=cfg["obj_base_pos"],
            euler=cfg["obj_init_euler"],
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.0, 0.5),
        ),
    )

    scene.build(n_envs=args.n_envs)

    motors_dof_idx, motors_name = get_motors_info(entity)

    # for each env, add small perturbance to the init joint angle
    joint_angles = torch.tensor(cfg["default_joint_angles"]).repeat(args.n_envs, 1)
    if args.randomize:
        joint_angles = joint_angles + torch.randn_like(joint_angles) * 0.1

    entity.set_dofs_position(joint_angles, motors_dof_idx)
    entity.control_dofs_position(joint_angles, motors_dof_idx)

    scene.sim.set_gravity([0, 0, -10])
    for i in range(args.steps // 3):
        scene.step()

    scene.sim.set_gravity([0, 0, 5])
    for i in range(args.steps // 3):
        scene.step()

    scene.sim.set_gravity([2, 2, 0])
    for i in range(args.steps // (3 * 2)):
        scene.step()

    scene.sim.set_gravity([-1, -4, 0])
    for i in range(args.steps // (3 * 2)):
        scene.step()

    obj_pos_diff = obj.get_pos() - torch.tensor(cfg["obj_base_pos"])
    # get index of most stable obj (min pos diff)
    obj_pos_diff[torch.isnan(obj_pos_diff)] = float("inf")
    min_idx = torch.norm(obj_pos_diff, dim=1).argmin()
    print(min_idx, obj_pos_diff[min_idx])
    print("joint angles:")
    print(joint_angles[min_idx])


if __name__ == "__main__":
    main()
