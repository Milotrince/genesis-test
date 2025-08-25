import argparse

import genesis as gs
import numpy as np
import pandas as pd
from genesis.sensors.data_handlers import CSVFileWriter
from genesis.sensors.tactile import ForceSensorOptions
from genesis.utils.misc import tensor_to_array
from plot import plot_force_sensor_data

CSV_FILE_NAME = "drop_force_test.csv"


if __name__ == "__main__":
    GRAVITY = -10.0

    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("--seconds", "-t", type=int, default=0.2, help="Number of steps to run")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0, 0, GRAVITY),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.1),
            size=(0.1, 0.1, 0.1),
        ),
        material=gs.materials.Rigid(),
    )
    force_sensor = scene.add_sensor(ForceSensorOptions(entity_idx=block.idx))
    force_sensor.add_recorder(
        handler=CSVFileWriter(filename=CSV_FILE_NAME),
        rec_options=gs.options.RecordingOptions(
            preprocess_func=lambda data, ground_truth_data: tensor_to_array(data["force"])
        ),
    )
    # cam = scene.add_camera(res=(640, 480), pos=(-2, 3, 2), lookat=(0.5, 0.5, 0.5), fov=30, GUI=args.vis)

    scene.build()

    force_sensor.start_recording()

    try:
        for _ in range(int(args.seconds / args.dt)):
            scene.step()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping recording...")
        force_sensor.stop_recording()

        data = pd.read_csv(CSV_FILE_NAME, header=None, names=["force_x", "force_y", "force_z"])
        x_time = np.arange(0, args.seconds, args.dt)
        plot_force_sensor_data(f"Drop Force Test dt={args.dt}s", data, x_time=x_time)
