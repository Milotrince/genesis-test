import argparse

import numpy as np
from tqdm import tqdm

import genesis as gs
from genesis.sensors import RecordingOptions, RigidNormalTangentialForceSensor, SensorDataRecorder
from genesis.sensors.data_handlers import CSVFileWriter, VideoFileWriter

np.set_printoptions(suppress=True, precision=4, linewidth=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=1, help="Number of seconds to simulate")
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--n_envs", type=int, default=0, help="Number of environments (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", type=int, default=2, help="Number of blocks to simulate")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0, 0, -10.0),
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

    blocks = []
    sensors = []
    for i in range(args.b):
        block = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.1 * i, 0.0, 0.1 * i + 0.1),
                size=(0.1, 0.1, 0.1),
            ),
            material=gs.materials.Rigid(rho=(i + 1) * 1000),
        )
        blocks.append(block)
        sensors.append(RigidNormalTangentialForceSensor(entity=block))

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(res=(640, 480), pos=(-2, 3, 2), lookat=(0.5, 0.5, 0.5), fov=30, GUI=args.vis)

    scene.build(n_envs=args.n_envs)

    data_recorder = SensorDataRecorder(step_dt=args.dt)
    data_recorder.add_sensor(
        cam, RecordingOptions(handler=VideoFileWriter(filename="camera_sensor.mp4", fps=1 / args.dt))
    )
    for i, sensor in enumerate(sensors):
        data_recorder.add_sensor(sensor, RecordingOptions(handler=CSVFileWriter(filename=f"contact_sensor_{i}.csv")))

    # data_recorder.start_recording()

    for _ in tqdm(range(steps), total=steps):
        scene.step()
        # data_recorder.step()

        for i, sensor in enumerate(sensors):
            if args.n_envs > 0:
                print(f"sensor {i}:", sensor.read(envs_idx=0))
            else:
                print(f"sensor {i}:", sensor.read())

    # data_recorder.stop_recording()


if __name__ == "__main__":
    main()
