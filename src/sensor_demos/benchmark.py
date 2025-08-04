import argparse
import time

import numpy as np
import taichi as ti
from tqdm import tqdm

import genesis as gs
from genesis._main import clean
from genesis.sensors import (
    RigidContactForceGridSensor,
    RigidContactForceSensor,
    RigidContactSensor,
    RigidNormalTangentialForceGridSensor,
    RigidNormalTangentialForceSensor,
)
from genesis.sensors.tactile_old import (
    OldRigidContactForceGridSensor,
    OldRigidContactForceSensor,
    OldRigidContactSensor,
)

np.set_printoptions(suppress=True, precision=4, linewidth=120)


def test_scene(args, sensor_type):
    if args.clean:
        clean()

    scene_start_time = time.perf_counter()

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=args.dt,
            gravity=(0, 0, -9.81),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
    )

    # Setup scene entities
    scene.add_entity(gs.morphs.Plane())

    blocks = []
    sensors = []
    pos_radius = 0.06
    for i in range(args.b):
        block = scene.add_entity(
            gs.morphs.Box(
                pos=(pos_radius * np.cos(i * np.pi / 5) + 0.02, pos_radius * np.sin(i * np.pi / 5), 0.3 + 0.04 * i),
                size=(0.02, 0.02, 0.02),
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 1.0, 1.0, 0.5),
            ),
        )
        blocks.append(block)
        if sensor_type is not None:
            kwargs = {}
            if "Grid" in sensor_type.__name__:
                kwargs["grid_size"] = tuple(args.grid_size)
            sensor = sensor_type(entity=block, **kwargs)
            sensors.append(sensor)

    steps = int(args.seconds / args.dt)

    scene_prebuild_time = time.perf_counter()
    scene.build(n_envs=args.n_envs)
    scene_postbuild_time = time.perf_counter()

    ti.sync()
    start_time = time.perf_counter()
    for i in tqdm(range(steps), total=steps):
        scene.step()

        if sensor_type is not None:
            for sensor in sensors:
                sensor.read()
    ti.sync()
    end_time = time.perf_counter()
    sim_loop_time = end_time - start_time

    # scene_end_time = time.perf_counter()
    # total_scene_prebuild_time = scene_prebuild_time - scene_start_time
    # total_scene_build_time = scene_postbuild_time - scene_prebuild_time
    # total_scene_time = scene_end_time - scene_start_time
    # return total_scene_time, sim_loop_time, total_scene_prebuild_time, total_scene_build_time
    return sim_loop_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=1, help="Number of seconds to simulate")
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--n_envs", type=int, default=2048, help="Number of environments (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-b", type=int, default=10, help="Number of sensors (attached to rigid blocks) to simulate")
    parser.add_argument("--grid_size", type=int, nargs=3, default=(10, 10, 1), help="Grid size for sensors")
    parser.add_argument("--clean", action="store_true", help="Whether to `gs clean` caches before running the test")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    print("Measuring how long it takes to read every sensor in a scene with multiple blocks...")

    times_dict = {}

    print("Testing scene without sensors...")
    times_dict["No Sensors"] = test_scene(args, sensor_type=None)
    for sensor_type in [
        RigidContactSensor,
        # OldRigidContactSensor,
        RigidContactForceSensor,
        RigidNormalTangentialForceSensor,
        # OldRigidContactForceSensor,
        RigidContactForceGridSensor,
        RigidNormalTangentialForceGridSensor,
        # OldRigidContactForceGridSensor,
    ]:
        print(f"Testing {sensor_type.__name__}...")
        times_dict[sensor_type.__name__] = test_scene(args, sensor_type=sensor_type)

    steps = int(args.seconds / args.dt)
    print("batch size:", args.n_envs, ", num sensors:", args.b, ", grid size:", args.grid_size, ", sim steps:", steps)

    # print in table format average time per step for both sensors
    print(f"{'Sensor Type':<32} {'Total sim loop time (s)':<20}")
    for sensor_type, time_taken in times_dict.items():
        print(f"{sensor_type:<32} {time_taken:<20.4f}")
