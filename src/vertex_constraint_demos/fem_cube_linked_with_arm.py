import genesis as gs
import numpy as np
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", 
        choices=["explicit", "implicit"], 
        default="explicit",
        help="FEM solver type (default: explicit)"
    )
    parser.add_argument(
        "--dt", 
        type=float, 
        help="Time step (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--substeps", 
        type=int, 
        help="Number of substeps (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=0,
        help="Number of environments (default: 0)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    parser.add_argument(
        "--vis", "-v",
        action="store_true",
        help="Show visualization GUI"
    )

    args = parser.parse_args()
    
    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 1e-3
        substeps = args.substeps if args.substeps is not None else 1

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level=None)
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=substeps,
            gravity=(0, 0, -9.81),
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=args.solver == "implicit",
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.e7, # stiffness
            nu=0.45, # compressibility (0 to 0.5)
            rho=1000.0, # density
            model='stable_neohookean'
        ),
    )
    arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(0, 0, 0),
        ),
    )

    # Setup camera for recording
    video_fps = 1 / dt
    max_fps = 100
    frame_interval = max(1, int(video_fps / max_fps)) if max_fps > 0 else 1
    cam = scene.add_camera(
        res=(640, 480), 
        pos=(-2, 3, 2), 
        lookat=(0.5, 0.5, 0.5),
        fov=30, 
        GUI=args.vis
    )

    # scene.build(n_envs=args.n_envs, env_spacing=(1.0, 1.0))
    scene.build(n_envs=args.n_envs)
    cam.start_recording()

    try:
        joint_names = [j.name for j in arm.joints]
        dofs_idx_local = []
        for j in arm.joints:
            dofs_idx_local += j.dofs_idx_local
        end_joint = arm.get_joint(joint_names[-1])


        arm.set_dofs_kp([450, 450, 350, 350, 200, 200, 200, 10, 10])
        arm.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 10, 10])
        arm.set_dofs_force_range(
            [-87, -87, -87, -87, -12, -12, -12, -100, -100],
            [87, 87, 87, 87, 12, 12, 12, 100, 100],
        )

        dofs_position = [0.9643, -0.3213, -0.6685, -2.3139, -0.2890,  2.0335, -1.6014,  0.0306, 0.0306]
        if args.n_envs > 0:
            dofs_position = [dofs_position] * args.n_envs
        for i in range(100):
            arm.set_dofs_position(np.array(dofs_position))
            scene.step()
            if i % frame_interval == 0:
                cam.render()
        
        pin_idx = [1, 5]
        cube.set_vertex_constraints(
            verts_idx=pin_idx,
            link=end_joint.link,
        )
        scene.draw_debug_spheres(poss=cube.init_positions[pin_idx], radius=0.02, color=(1, 0, 1, 0.8))

        arm_target_pos = [0.3, 0.2, 0.6]
        scene.draw_debug_spheres(poss=[arm_target_pos], radius=0.02, color=(0, 1, 0, 0.8))
        scene.draw_debug_spheres(poss=end_joint.link.get_pos(), radius=0.02, color=(0, 0, 1, 0.8))

        pos = arm_target_pos
        quat = [0, 1, 0, 0]
        if args.n_envs > 0:
            pos = [pos] * args.n_envs
            quat = [quat] * args.n_envs
        qpos = arm.inverse_kinematics(
            link=end_joint.link,
            pos=np.array(pos),
            quat=np.array(quat),
        )
       
        steps = int(1 / dt)
        for i in tqdm(range(steps), total=steps):
            arm.control_dofs_position(qpos)
            scene.step()
            if i % frame_interval == 0:
                cam.render()
        
        print("Now dropping the cube")
        cube.remove_vertex_constraints()
        steps = 1
        # steps = steps // 2
        for i in tqdm(range(steps), total=steps):
            arm.control_dofs_position(qpos)
            scene.step()
            if i % frame_interval == 0:
                cam.render()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        actual_fps = video_fps / frame_interval
        video_filename = f"cube_link_arm_{args.solver}_nenvs={args.n_envs}_dt={dt}_substeps={substeps}.mp4"
        cam.stop_recording(save_to_filename=video_filename, fps=actual_fps)
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
