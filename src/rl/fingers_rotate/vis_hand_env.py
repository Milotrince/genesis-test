import multiprocessing
import tkinter as tk
from functools import partial
from tkinter import ttk

import genesis as gs
import numpy as np
import torch

FPS = 60


filename = "/home/trinityc/.cache/huggingface/hub/datasets--Genesis-Intelligence--assets/snapshots/35183e88cce2b1ca1b2d0f777df71d4bb21f3331/allegro_hand/allegro_hand_right_glb.urdf"
collision = False
rotate = False
show_link_frame = False
scale = 1.0

cfg = {
    "default_joint_angles": [
        0.1029,
        0.1343,
        0.0645,
        1.1922,
        1.3538,
        1.1274,
        1.1962,
        1.0774,
        0.2687,
        0.1853,
        0.5328,
        0.1505,
        0.1862,
        0.5078,
        0.2611,
        0.8198,
        # -0.0852,
        # 0.0627,
        # 0.0724,
        # 1.1736,
        # 1.2923,
        # 1.1202,
        # 1.1374,
        # 1.1983,
        # 0.1551,
        # 0.1088,
        # 0.3383,
        # 0.1499,
        # 0.1343,
        # 0.5355,
        # 0.2164,
        # 0.8535,
    ],
    "hand_init_pos": [0.0, 0.0, 0.5],
    "hand_init_euler": [0.0, -90.0, 0.0],
    # ===== object configuration ======
    "obj_base_pos": [0.01, 0.0, 0.64],
    "obj_init_euler": [0.0, 0.0, 0.0],  # roll, pitch, yaw in degrees
    "obj_morph": "cylinder",
    "obj_size": 0.05,
}

# -----------------------------------


class JointControlGUI:
    def __init__(self, master, motors_name, motors_position_limit, motors_position):
        self.master = master
        self.master.title("Joint Controller")  # Set the window title
        self.motors_name = motors_name
        self.motors_position_limit = motors_position_limit
        self.motors_position = motors_position
        self.motors_default_position = np.clip(
            np.zeros(len(motors_name)), motors_position_limit[:, 0], motors_position_limit[:, 1]
        )
        self.sliders = []
        self.values_label = []
        self.create_widgets()
        self.reset_motors_position()

    def create_widgets(self):
        for i_m, name in enumerate(self.motors_name):
            self.update_joint_position(i_m, self.motors_default_position[i_m])
            min_limit, max_limit = map(float, self.motors_position_limit[i_m])
            frame = tk.Frame(self.master)
            frame.pack(pady=5, padx=10, fill=tk.X)

            tk.Label(frame, text=f"{name}", font=("Arial", 12), width=20).pack(side=tk.LEFT)

            slider = ttk.Scale(
                frame,
                from_=min_limit,
                to=max_limit,
                orient=tk.HORIZONTAL,
                length=300,
                command=partial(self.update_joint_position, i_m),
            )
            slider.pack(side=tk.LEFT, padx=5)
            self.sliders.append(slider)

            value_label = tk.Label(frame, text=f"{slider.get():.2f}", font=("Arial", 12))
            value_label.pack(side=tk.LEFT, padx=5)
            self.values_label.append(value_label)

            # Update label dynamically
            def update_label(s=slider, l=value_label):
                def callback(event):
                    l.config(text=f"{s.get():.2f}")

                return callback

            slider.bind("<Motion>", update_label())

        tk.Button(self.master, text="Reset", font=("Arial", 12), command=self.reset_motors_position).pack(pady=20)

    def update_joint_position(self, idx, val):
        self.motors_position[idx] = float(val)

    def reset_motors_position(self):
        self.set_motors_position(self.motors_default_position)

    def set_motors_position(self, motors_position):
        for i_m, slider in enumerate(self.sliders):
            slider.set(motors_position[i_m])
            self.values_label[i_m].config(text=f"{motors_position[i_m]:.2f}")
            self.motors_position[i_m] = motors_position[i_m]


def get_motors_info(robot):
    motors_dof_idx = []
    motors_dof_name = []
    for joint in robot.joints:
        if joint.type == gs.JOINT_TYPE.FREE:
            continue
        elif joint.type == gs.JOINT_TYPE.FIXED:
            continue
        dofs_idx_local = robot.get_joint(joint.name).dofs_idx_local
        if dofs_idx_local:
            if len(dofs_idx_local) == 1:
                dofs_name = [joint.name]
            else:
                dofs_name = [f"{joint.name}_{i_d}" for i_d in dofs_idx_local]
            motors_dof_idx += dofs_idx_local
            motors_dof_name += dofs_name
    return motors_dof_idx, motors_dof_name


def _start_gui(motors_name, motors_position_limit, motors_position, stop_event):
    def on_close():
        nonlocal after_id
        if after_id is not None:
            root.after_cancel(after_id)
            after_id = None
        stop_event.set()
        root.destroy()
        root.quit()

    root = tk.Tk()
    app = JointControlGUI(root, motors_name, motors_position_limit, motors_position)
    root.protocol("WM_DELETE_WINDOW", on_close)

    app.set_motors_position(cfg["default_joint_angles"])

    def check_event():
        nonlocal after_id
        if stop_event.is_set():
            on_close()
        elif root.winfo_exists():
            after_id = root.after(100, check_event)

    after_id = root.after(100, check_event)
    root.mainloop()


if __name__ == "__main__":
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(cfg["hand_init_pos"][0] + 0.3, cfg["hand_init_pos"][1] + 0.3, cfg["hand_init_pos"][2] + 0.7),
            camera_lookat=cfg["hand_init_pos"],
            camera_fov=40,
            max_FPS=FPS,
        ),
        vis_options=gs.options.VisOptions(
            show_link_frame=show_link_frame,
        ),
        show_viewer=True,
    )

    if filename.endswith(".urdf"):
        morph_cls = gs.morphs.URDF
    elif filename.endswith(".xml"):
        morph_cls = gs.morphs.MJCF
    else:
        morph_cls = gs.morphs.Mesh
    entity = scene.add_entity(
        morph_cls(
            file=filename,
            collision=collision,
            scale=scale,
            pos=cfg["hand_init_pos"],
            euler=cfg["hand_init_euler"],
            fixed=True,
            merge_fixed_links=False,
        ),
        surface=gs.surfaces.Default(
            vis_mode="visual" if not collision else "collision",
        ),
    )

    obj = scene.add_entity(
        gs.morphs.Cylinder(
            radius=cfg["obj_size"],
            height=0.05,
            pos=cfg["obj_base_pos"],
            euler=cfg["obj_init_euler"],
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.4, 0.0, 0.5),
        ),
    )

    scene.build(compile_kernels=False)

    # Get motor info
    motors_dof_idx, motors_name = get_motors_info(entity)
    # Get motor position limits.
    # Makes sure that all joints are bounded, included revolute joints.
    if motors_dof_idx:
        motors_position_limit = torch.stack(entity.get_dofs_limit(motors_dof_idx), dim=1).numpy()
        motors_position_limit[motors_position_limit == -np.inf] = -np.pi
        motors_position_limit[motors_position_limit == +np.inf] = +np.pi

        print(motors_name)
        print(motors_position_limit)

        # Start the GUI process
        manager = multiprocessing.Manager()
        motors_position = manager.list([0.0 for _ in motors_dof_idx])
        stop_event = multiprocessing.Event()
        gui_process = multiprocessing.Process(
            target=_start_gui, args=(motors_name, motors_position_limit, motors_position, stop_event), daemon=True
        )
        gui_process.start()
    else:
        stop_event = multiprocessing.Event()

    try:
        t = 0
        while scene.viewer.is_alive() and not stop_event.is_set():
            # Rotate entity if requested
            if rotate:
                t += 1 / FPS
                entity.set_quat(gs.utils.geom.xyz_to_quat(np.array([0, 0, t * 50]), rpy=True, degrees=True))

            if motors_dof_idx:
                entity.set_dofs_position(
                    position=torch.tensor(motors_position),
                    dofs_idx_local=motors_dof_idx,
                    zero_velocity=True,
                )
            scene.visualizer.update(force=True)
        stop_event.set()
        if motors_dof_idx:
            gui_process.join()
    except KeyboardInterrupt:
        pass
    finally:
        print("--------------------------------")
        print(entity.get_dofs_position())
        print("--------------------------------")
        stop_event.set()
        if motors_dof_idx:
            gui_process.join()
