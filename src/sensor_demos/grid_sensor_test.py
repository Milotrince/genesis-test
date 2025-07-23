import argparse

import genesis as gs
import numpy as np
import trimesh
from genesis.sensors import RecordingOptions, RigidContactGridSensor, SensorDataRecorder
from genesis.sensors.data_handlers import VideoFileWriter
from genesis.utils.misc import tensor_to_array
from tqdm import tqdm

np.set_printoptions(suppress=True, precision=4, linewidth=120)


def visualize_grid_sensor(scene: gs.Scene, sensor: RigidContactGridSensor):
    """
    Draws debug objects on scene to visualize the contact grid sensor data.

    Note: This method is very inefficient and purely for demo/debugging purposes.
    This processes the grid data from the sensor, which means the transformation from global-> local frame is undone to
    revert back to into global frame to draw the debug objects.
    """
    grid_data = sensor.read()

    link_pos = tensor_to_array(scene._sim.rigid_solver.get_links_pos(links_idx=sensor.link_idx).squeeze(axis=1))
    link_quat = tensor_to_array(scene._sim.rigid_solver.get_links_quat(links_idx=sensor.link_idx).squeeze(axis=1))

    sensor_dims = sensor.max_bounds - sensor.min_bounds
    grid_cell_size = sensor_dims / np.array(sensor.grid_size)

    debug_objs = []

    for x in range(grid_data.shape[1]):
        for y in range(grid_data.shape[2]):
            for z in range(grid_data.shape[3]):
                is_contact = grid_data[0, x, y, z]

                color = np.array([int(is_contact), 0.0, 1.0 - int(is_contact), 0.4])

                mesh = trimesh.creation.box(extents=grid_cell_size)
                mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, [len(mesh.vertices), 1]))

                local_pos = sensor.min_bounds + (np.array([x, y, z]) + 0.5) * grid_cell_size

                T = trimesh.transformations.quaternion_matrix(link_quat)
                T[:3, 3] = link_pos

                local_T = np.eye(4)
                local_T[:3, 3] = local_pos
                final_T = T @ local_T

                debug_objs.append(scene.draw_debug_mesh(mesh, T=final_T))

    return debug_objs


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

    blocks = []
    sensors = []
    for i in range(args.b):
        block = scene.add_entity(
            # morph=gs.morphs.Box(pos=(0.2 * i, 0.0, 0.1 * i), size=(0.05, 0.05, 0.1), visualization=False),
            morph=gs.morphs.Sphere(pos=(0.2 * i, 0.0, 0.6 - 0.1 * i), radius=0.05),
            material=gs.materials.Rigid(rho=i * 100 + 200),
        )
        blocks.append(block)
        sensors.append(RigidContactGridSensor(entity=block, grid_size=(1, 1, 2)))

    steps = int(args.seconds / args.dt)
    cam = scene.add_camera(res=(640, 480), pos=(-1, 2, 1), lookat=(0.5, 0.5, 0.0), fov=30, GUI=args.vis)

    scene.build(n_envs=args.n_envs)

    data_recorder = SensorDataRecorder(step_dt=args.dt)
    data_recorder.add_sensor(
        cam, RecordingOptions(handler=VideoFileWriter(filename="camera_sensor.mp4", fps=1 / args.dt))
    )
    # for i, sensor in enumerate(sensors):
    # data_recorder.add_sensor(sensor, RecordingOptions(handler=CSVFileWriter(filename=f"contact_sensor_{i}.csv")))

    data_recorder.start_recording()

    for i in tqdm(range(steps), total=steps):
        scene.step()
        if i % 5 == 0:
            scene.clear_debug_objects()
        data_recorder.step()

        for i, sensor in enumerate(sensors):
            if args.vis:
                visualize_grid_sensor(scene, sensor)
            else:
                if args.n_envs > 0:
                    print(f"sensor {i}:", sensor.read(envs_idx=0))
                else:
                    print(f"sensor {i}:", sensor.read())

    data_recorder.stop_recording()


if __name__ == "__main__":
    main()
