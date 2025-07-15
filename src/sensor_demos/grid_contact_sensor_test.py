import argparse
import genesis as gs
from genesis.sensors import DataRecordingOptions, RigidContactForceGridSensor
from genesis.sensors.data_handlers import VideoFileStreamer, CSVFileWriter, NPZFileWriter
from tqdm import tqdm

import numpy as np

np.set_printoptions(precision=4, suppress=True)


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
        help="Number of substeps",
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

    scene.add_entity(gs.morphs.Plane())

    block_sensor = scene.add_sensor(
        RigidContactForceGridSensor,
        grid_size=(4, 4, 2),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(2.0, 2.0, 0.1),
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 0.5, 0.5, 1.0),
        ),
        material=gs.materials.Rigid(),
    )

    sphere1 = scene.add_entity(
        gs.morphs.Sphere(pos=(0.5, 0.4, 0.3), radius=0.1),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(pos=(-0.5, -0.3, 0.4), radius=0.1),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    sphere3 = scene.add_entity(
        gs.morphs.Sphere(pos=(-0.2, 0.7, 0.5), radius=0.1),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(
        res=(640, 480), pos=(-2, 3, 1.5), lookat=(0.0, 0.0, 0.1), fov=30, GUI=args.vis
    )

    scene.build(n_envs=args.n_envs)

    cam.start_recording(
        DataRecordingOptions(
            handler=VideoFileStreamer(filename="grid_test.mp4", fps=1 / args.dt)
        )
    )
    block_sensor.start_recording(
        # DataRecordingOptions(handler=CSVFileWriter(filename="grid_test.csv"))
        DataRecordingOptions(handler=NPZFileWriter(filename="grid_test.npz"))
    )

    try:
        for _ in tqdm(range(steps), total=steps):
            scene.step()

    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        from utils import timer

        with timer():
            scene.stop_recording_all()


if __name__ == "__main__":
    main()
