import genesis as gs
import numpy as np
import argparse
from tqdm import tqdm
from rigid_tactile_sensor import RigidForceSensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seconds", "-t", 
        type=int, 
        default=1,
        help="Number of seconds to simulate"
    )
    parser.add_argument(
        "--dt", 
        type=float, 
        default=1e-3,
        help="Time step (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--substeps", 
        type=int, 
        default=1,
        help="Number of substeps (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--vis", "-v",
        action="store_true",
        help="Show visualization GUI"
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
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    rigid_sensor = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.05),
            size=(0.50, 0.50, 0.05),
        ),
    )
    rigid_sensor.apply_plugin(RigidForceSensor())


    # Setup camera for recording
    video_fps = 1 / args.dt
    steps = int(args.seconds / video_fps)
    max_fps = 100
    frame_interval = max(1, int(video_fps / max_fps)) if max_fps > 0 else 1
    cam = scene.add_camera(
        res=(640, 480), 
        pos=(-2, 3, 2), 
        lookat=(0.5, 0.5, 0.5),
        fov=30, 
        GUI=args.vis
    )

    scene.build()
    cam.start_recording()

    try:
        for i in tqdm(range(steps), total=steps):
            scene.step()
            if i % frame_interval == 0:
                cam.render()


    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")

        actual_fps = video_fps / frame_interval
        video_filename = f"sensor_test_dt={args.dt}_substeps={args.substeps}.mp4"
        cam.stop_recording(save_to_filename=video_filename, fps=actual_fps)
        gs.logger.info(f"Saved video to {video_filename}")


if __name__ == "__main__":
    main()
