import argparse

import genesis as gs
import numpy as np
import pandas as pd
from genesis.sensors import ForceSensorOptions
from genesis.sensors.data_handlers import CSVFileWriter
from genesis.utils import tensor_to_array
from plot import plot_force_sensor_data

CSV_FILE_NAME = "drag_hand_test.csv"


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
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
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

    initial_pos = np.array([0.0, 0.0, 0.20])

    block = scene.add_entity(
        gs.morphs.Box(pos=initial_pos + np.array([0.0, 0.0, 0.2]), euler=(0.0, 0.0, 0.0), size=(0.05, 0.05, 0.05))
    )
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            pos=initial_pos,
            euler=(100.0, 90.0, 0.0),
            scale=1.0,
            file="/home/trinityc/genesis/eden_assets/hands/shadow_hand/shadow_hand_right_full.urdf",
            # fixed=True,
        ),
    )
    print([link.name for link in robot.links])
    wrist = robot.get_link("wrist")
    hand_base = robot.get_link("base")

    finger1 = robot.get_link("ffdistal")
    finger2 = robot.get_link("mfdistal")
    dofs_idx_local = [finger1.idx_local, finger2.idx_local]

    force_sensor = scene.add_sensor(ForceSensorOptions(entity_idx=robot.idx, link_idx_local=finger1.idx_local))
    force_sensor.add_recorder(
        handler=CSVFileWriter(filename=CSV_FILE_NAME),
        rec_options=gs.options.RecordingOptions(
            preprocess_func=lambda data, ground_truth_data: tensor_to_array(data["force"])
        ),
    )
    cam = scene.add_camera(res=(640, 480), pos=(1, 2, 1), lookat=(0.0, 0.0, 0.0), fov=30, GUI=args.vis)

    scene.build()

    scene.sim.set_gravity((0.0, 0.0, GRAVITY))
    # print(robot.get_dofs_stiffness())
    # print(robot.get_dofs_damping())

    # scene.rigid_solver.add_weld_constraint(robot.base_link_idx, block.base_link_idx)

    cam.start_recording()
    force_sensor.start_recording()
    # q = np.array([0, 0, 0, 0])

    print(robot.get_qpos())
    print("robot.q_start", robot.q_start)
    print("robot.q_end", robot.q_end)
    print("base q_start", hand_base.q_start)
    print("base q_end", hand_base.q_end)

    q = hand_base.q_start

    try:
        for i in range(int(args.seconds / args.dt)):
            pos = initial_pos + np.array([0.0, 0.001, 0.00]) * i
            # will not work because the whole hand stays in initial pose unaffected by gravity
            # robot.set_pos(pos, zero_velocity=True)

            robot.set_qpos(pos, qs_idx_local=[q, q + 1, q + 2], zero_velocity=False)

            # set_pos not defined on link
            # base.set_pos(pos, zero_velocity=True)

            # weld constraint not working as expected either with set_pos
            # block.set_pos(pos)
            # block.set_quat(np.array([0.0, 0.0, 0.0, 1.0]))

            scene.step()
            cam.render()

            # euler_x = int(input("Enter euler_x: "))
            # euler_y = int(input("Enter euler_y: "))
            # euler_z = int(input("Enter euler_z: "))

            # quat = euler_to_quat(np.array([euler_x, euler_y, euler_z]))
            # robot.set_quat(quat)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping recording...")
        force_sensor.stop_recording()
        cam.stop_recording(save_to_filename="drag_hand_test.mp4")

    data = pd.read_csv(CSV_FILE_NAME, header=None, names=["force_x", "force_y", "force_z"])
    x_time = np.arange(0, args.seconds, args.dt)
    data.head()
    plot_force_sensor_data(f"Drag Hand Test dt={args.dt}s c={args.constraint_timeconst}", data, x_time=x_time)
