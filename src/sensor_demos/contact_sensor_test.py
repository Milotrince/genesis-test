import argparse
import genesis as gs
from genesis.sensors import RigidContactSensor, RigidContactForceSensor, RecordingOptions, SensorDataRecorder
from genesis.sensors.data_handlers import NPZFileWriter, CSVFileWriter, VideoFileStreamer
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds", "-t", type=float, default=1, help="Number of seconds to simulate"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-3,
        help="Simulation time step"
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="Number of substeps (auto-selected based on solver if not specified)",
    )
    parser.add_argument(
        "--n_envs", type=int, default=0, help="Number of environments (default: 0)"
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--vis", "-v", action="store_true", help="Show visualization GUI"
    )

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            substeps=args.substeps,
            gravity=(0, 0, -9.81),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        vis_options=gs.options.VisOptions(
            # env_separate_rigid=True, # NOT TESTED
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    block_contact_sensor = scene.add_sensor(
        RigidContactSensor,
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.5),
            size=(0.05, 0.05, 0.2),
        ),
        material=gs.materials.Rigid(),
    )

    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    hand = robot.get_link("hand")
    hand_contact_sensor = robot.add_sensor(RigidContactSensor, link_idx=hand.idx)
    hand_force_sensor = robot.add_sensor(RigidContactForceSensor, link_idx=hand.idx)
    print("hand link idx:", hand.idx)
    print("block link idx:", block_contact_sensor.entity.base_link_idx)

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(
        res=(640, 480), pos=(-2, 3, 2), lookat=(0.5, 0.5, 0.5), fov=30, GUI=args.vis
    )

    scene.build(n_envs=args.n_envs)

    dofs_position = [-1, 0.8, 1, -2, 1, 0.5, 0.5, 0.04, 0.04]
    if args.n_envs > 0:
        dofs_position = [dofs_position] * args.n_envs
    robot.set_dofs_position(np.array(dofs_position))

    data_recorder = SensorDataRecorder(step_dt=args.dt)
    data_recorder.add_sensor(
        cam, RecordingOptions(handler=VideoFileStreamer(filename="camera_sensor.mp4", fps=1 / args.dt))
    )
    data_recorder.add_sensor(
        block_contact_sensor, RecordingOptions(handler=CSVFileWriter(filename="contact_sensor.csv"))
    )
    data_recorder.add_sensor(
        hand_force_sensor, RecordingOptions(handler=NPZFileWriter(filename="force_contact_sensor.npz"))
    )

    data_recorder.start_recording()

    if args.n_envs > 0:
        gs.logger.info(
            f"Drawing contacts for first environment only (n_envs={args.n_envs})"
        )

    try:
        for _ in tqdm(range(steps), total=steps):
            scene.step()
            data_recorder.step()

            is_hand_contact = hand_contact_sensor.read()
            is_contact = block_contact_sensor.read()
            if args.n_envs > 0:
                is_hand_contact = is_hand_contact[0]
                is_contact = is_contact[0]

            if is_hand_contact:
                scene.draw_debug_spheres(
                    poss=hand.get_pos(), radius=0.06, color=(1, 0, 0, 0.4)
                )
            if is_contact:
                scene.draw_debug_spheres(
                    poss=block_contact_sensor.entity.get_pos(),
                    radius=0.06,
                    color=(0, 0, 1, 0.4),
                )
            cam.render()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        data_recorder.stop_recording()


if __name__ == "__main__":
    main()
