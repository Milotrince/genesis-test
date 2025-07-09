import argparse
import genesis as gs
from genesis.engine.sensors import ContactSensor
from tqdm import tqdm

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

    sensor = scene.add_sensor(
        ContactSensor,
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 2.0),
            size=(0.05, 0.05, 0.2),
        ),
    )

    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    hand = robot.get_link("hand")
    sensor_hand = robot.add_sensor(ContactSensor, link_idx=hand.idx)

    # Setup camera for recording
    video_fps = 1 / args.dt
    steps = int(args.seconds * video_fps)
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

            is_hand_contact = sensor_hand.read()
            if is_hand_contact:
                scene.draw_debug_spheres(poss=hand.get_pos(), radius=0.06, color=(1, 0, 0, 0.4))

            is_contact = sensor.read()
            if is_contact:
                scene.draw_debug_spheres(poss=sensor.entity.get_pos(), radius=0.06, color=(0, 0, 1, 0.4))

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
