import argparse

import genesis as gs
import numpy as np
import pandas as pd
from genesis.sensors.data_handlers import CSVFileWriter
from genesis.sensors.tactile import ForceSensorOptions
from genesis.utils.misc import tensor_to_array
from plot import plot_force_sensor_data

CSV_FILE_NAME = "push_force_test.csv"


if __name__ == "__main__":
    GRAVITY = -10.0

    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    parser.add_argument("--seconds", "-t", type=int, default=2, help="Number of secondsto run")
    parser.add_argument("--constraint_timeconst", "-c", type=float, default=0.01, help="Constraint time constant")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0, 0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_timeconst=args.constraint_timeconst,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    # weight = scene.add_entity(
    #     morph=gs.morphs.Box(
    #         pos=(0.0, 0.0, 0.15),
    #         size=(0.1, 0.1, 0.1),
    #     ),
    #     material=gs.materials.Rigid(),
    # )

    block = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
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
    cam = scene.add_camera(res=(640, 480), pos=(-1, 2, 1), lookat=(0.0, 0.0, 0.0), fov=30, GUI=args.vis)

    scene.build()

    cam.start_recording()
    force_sensor.start_recording()

    mass = 10 + 5 * np.sin(np.arange(0, args.seconds, args.dt) * 2 * np.pi / args.seconds)

    try:
        for i in range(int(args.seconds / args.dt)):
            # weight.set_mass(mass[i])
            block.set_mass(mass[i])
            scene.step()
            cam.render()
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping recording...")
        force_sensor.stop_recording()
        cam.stop_recording(save_to_filename="push_force_test.mp4")

    data = pd.read_csv(CSV_FILE_NAME, header=None, names=["force_x", "force_y", "force_z"])
    x_time = np.arange(0, args.seconds, args.dt)
    data.head()
    plot_force_sensor_data(f"Push Force Test dt={args.dt}s c={args.constraint_timeconst}", data, x_time=x_time)

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(x_time, mass)
    plt.savefig("mass.png")
