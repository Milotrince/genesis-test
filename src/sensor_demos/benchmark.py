import argparse
import time

import genesis as gs
import numpy as np
import taichi as ti
from genesis.sensors import RigidContactForceGridSensor
from genesis.sensors.tactile_old import OldRigidContactForceGridSensor
from tqdm import tqdm

np.set_printoptions(suppress=True, precision=4, linewidth=120)


def test_scene(args, sensor_type):
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
        sensor = sensor_type(entity=block, grid_size=tuple(args.grid_size))
        sensors.append(sensor)

    steps = int(args.seconds / args.dt)

    scene.build(n_envs=args.n_envs)

    total_time = 0.0

    for i in tqdm(range(steps), total=steps):
        scene.step()

        ti.sync()
        start_time = time.perf_counter()
        for sensor in sensors:
            sensor.read()
        ti.sync()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    print(f"Total time spent reading sensors: {total_time:.4f} seconds")
    print(f"Average time reading all sensors per step: {total_time / steps:.4f} seconds")

    ti.sync()
    scene_end_time = time.perf_counter()
    scene_total_time = scene_end_time - scene_start_time
    print(f"Scene setup and execution time: {scene_total_time:.4f} seconds")

    return total_time, scene_total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", "-t", type=float, default=1, help="Number of seconds to simulate")
    parser.add_argument("--dt", type=float, default=1e-2, help="Simulation time step")
    parser.add_argument("--n_envs", type=int, default=0, help="Number of environments (default: 0)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("-b", type=int, default=2, help="Number of sensors (attached to rigid blocks) to simulate")
    parser.add_argument("--grid_size", type=int, nargs=3, default=(6, 6, 1), help="Grid size for sensors")

    args = parser.parse_args()

    gs.init(backend=gs.gpu, logging_level=None)

    print("Measuring how long it takes to read every sensor in a scene with multiple blocks...")

    print("Testing new RigidContactForceGridSensor...")
    new_t, new_t_scene = test_scene(args, sensor_type=RigidContactForceGridSensor)
    print()
    print("Testing old RigidContactForceGridSensor...")
    old_t, old_t_scene = test_scene(args, sensor_type=OldRigidContactForceGridSensor)

    steps = int(args.seconds / args.dt)
    print("batch size:", args.n_envs, ", num sensors:", args.b, ", grid size:", args.grid_size, ", sim steps:", steps)

    # print in table format average time per step for both sensors
    print(
        f"{'Sensor Type':<40} {'Total read time (s)':<20} {'Avg read time/step (s)':<25} {'Total Sim Build to Completion Time (s)':<40}"
    )
    print(f"{'Old RigidContactForceGridSensor':<40} {old_t:<20.4f} {old_t / steps:<25.4f} {old_t_scene:<40.4f}")
    print(f"{'New RigidContactForceGridSensor':<40} {new_t:<20.4f} {new_t / steps:<25.4f} {new_t_scene:<40.4f}")
