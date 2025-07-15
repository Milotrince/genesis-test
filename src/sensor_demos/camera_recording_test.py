import argparse
import genesis as gs
from genesis.sensors import DataRecordingOptions
from genesis.sensors.data_handlers import  VideoFileStreamer, VideoFileWriter
from tqdm import tqdm
import numpy as np
from utils import timer, memory_usage


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

    # sphere1 = scene.add_entity(
    #     gs.morphs.Sphere(pos=(0.5, 0.4, 0.3), radius=0.1),
    #     surface=gs.surfaces.Default(
    #         color=(1.0, 0.0, 0.0, 1.0),
    #     ),
    # )
    # sphere2 = scene.add_entity(
    #     gs.morphs.Sphere(pos=(-0.5, -0.3, 0.4), radius=0.1),
    #     surface=gs.surfaces.Default(
    #         color=(0.0, 1.0, 0.0, 1.0),
    #     ),
    # )
    # sphere3 = scene.add_entity(
    #     gs.morphs.Sphere(pos=(-0.2, 0.7, 0.5), radius=0.1),
    #     surface=gs.surfaces.Default(
    #         color=(0.0, 0.0, 1.0, 1.0),
    #     ),
    # )


    cam = scene.add_camera(
        res=(640, 480), pos=(-2, 3, 2), lookat=(0.5, 0.5, 0.5), fov=30, GUI=args.vis
    )

    scene.build(n_envs=args.n_envs)

    steps = int(args.seconds / args.dt)
    fps = 1 / args.dt

    # with memory_usage("test 1: small"):
    #     small_list = [0]

    # with memory_usage("test 2: large"):
    #     large_list = [0] * 10**7
   
    with timer("Time for manual render"), memory_usage("Max memory usage for manual render"):
        cam.start_recording()
        for _ in tqdm(range(steps), total=steps):
            scene.step()
            cam.render()
        cam.stop_recording(save_to_filename="camera_sensor.mp4", fps=fps)

    with timer("Time for VideoFileWriter"), memory_usage("Max memory usage for VideoFileWriter"):
        cam.start_recording(
            DataRecordingOptions(
                handler=VideoFileWriter(filename="camera_sensor_write.mp4", fps=fps)
            )
        )
        for _ in tqdm(range(steps), total=steps):
            scene.step()
        cam.stop_recording()

    with timer("Time for VideoFileStreamer"), memory_usage("Max memory usage for VideoFileStreamer"):
        cam.start_recording(
            DataRecordingOptions(
                handler=VideoFileStreamer(filename="camera_sensor_stream.mp4", fps=fps)
            )
        )
        for _ in tqdm(range(steps), total=steps):
            scene.step()
        cam.stop_recording()



if __name__ == "__main__":
    main()
