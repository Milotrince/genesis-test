import genesis as gs
import numpy as np
import argparse
from sensor import *
from utils import *
from gs_runner import GenesisRunner


class SensorTestRunner(GenesisRunner):
    def setup(self):
        # self.scene.add_entity(gs.morphs.Plane())

        rigid_ball = self.scene.add_entity(
            material=gs.materials.Rigid(
                rho=500.0,  # density
                friction=0.5
            ),
            morph=gs.morphs.Sphere(pos=(0.0, 0.0, 0.5), radius=0.1),
            surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
        )
        # duck = self.scene.add_entity(
        #     material=gs.materials.MPM.Elastic(E=6.e4, rho=500),
        #     morph=gs.morphs.Mesh(
        #         file="meshes/duck.obj",
        #         pos=(0.0, 0.0, 0.5),
        #         scale=0.07,
        #         euler=(90.0, 0.0, 90.0),
        #     ),
        #     surface=gs.surfaces.Default(
        #         color=(0.9, 0.8, 0.2, 1.0),
        #     ),
        #     # vis_mode="particle",
        # )

        self.sensor = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                # file="./objects/dome2.obj",
                # euler=(180.0, 0.0, 0.0),
                file="./objects/pad.obj",
                euler=(0.0, 90.0, 0.0),
                # pos=(0.0, 0.0, 0.5),
                pos=(0.0, 0.0, 0.3),
            ),
            material=gs.materials.FEM.Elastic(
                E=1.e6, # stiffness
                nu=0.45, # compressibility (0 to 0.5)
                rho=1000.0, # density
                # model='stable_neohookean'
            ),
        )

        verts, elems = gs.utils.element.mesh_to_elements(
                file=self.sensor._morph.file,
                pos=self.sensor._morph.pos,
                scale=self.sensor._morph.scale,
                tet_cfg=self.sensor.tet_cfg,
            )
        print(f"Sensor vertices: {verts.shape}")
        np.save("out/sensor_verts.npy", verts)

        # tet_cfg = gs.utils.mesh.generate_tetgen_config_from_morph(self.morph)

        self.init_vert_pos = self.sensor.init_positions.cpu().numpy()
        print(f"Initial vertex positions shape: {self.init_vert_pos.shape}")


        # sensor_idx = self.sensor.idx

        self.fem_solver = None
        for solver in self.scene.sim.solvers:
            if solver.__class__.__name__ == "FEMSolver":
                self.fem_solver = solver
                break
        print(
            f"FEM Solver found with {self.fem_solver.n_vertices} vertices and {self.fem_solver.n_elements} elements"
        )




        self.all_forces = []

    def step(self, i):
        if i == 0:
            pinned_vertices = np.arange(0, 10, 1) + self.sensor._v_start 
            print(f"Pinned vertices: {pinned_vertices}")
            target_positions = self.init_vert_pos[pinned_vertices]
            print(f"Target positions shape: {target_positions.shape}")
            self.fem_solver.set_vertex_constraints(
                vertex_indices=pinned_vertices,
                target_positions=target_positions,
                constraint_type="hard",
                # stiffness=1e6  # with constraint=soft
            )
        if i % 10 == 0:
            # forces = self.fem_solver.get_vertex_forces()
            forces = get_vertex_forces(self.fem_solver).to_numpy()
            self.all_forces.append(forces)
    
    def on_complete(self):
        all_forces = np.array(self.all_forces)
        print("Sensor forces shape:", all_forces.shape)
        forces_file = "out/forces.npy"
        np.save(forces_file, all_forces)
        gs.logger.info(f"Saved forces to {forces_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FEM Sensor Test with configurable solver")
    parser.add_argument(
        "--solver", 
        choices=["explicit", "implicit"], 
        default="explicit",
        help="FEM solver type (default: explicit)"
    )
    parser.add_argument(
        "--dt", 
        type=float, 
        help="Time step (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--substeps", 
        type=int, 
        help="Number of substeps (auto-selected based on solver if not specified)"
    )
    parser.add_argument(
        "--seconds", 
        type=float, 
        default=1.0,
        help="Simulation duration in seconds (default: 1.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 5e-3
        substeps = args.substeps if args.substeps is not None else 1
    
    runner = SensorTestRunner(
        video_out=get_next_filename(f"out/sensor_test_{args.solver}_dt={dt}_substeps={substeps}_##.mp4"),
        logging=True,

        camera_pos=(0.0, 2.0, 1.0),
        camera_lookat=(0.0, 0.0, 0.5),

        dt=dt,
        substeps=substeps,
        fem_options=gs.options.FEMOptions(
            gravity=(0.0, 0.0, 0.0),
            use_implicit_solver=args.solver == "implicit",
        ),
        mpm_options=gs.options.MPMOptions(
            gravity=(0.0, 0.0, -9.81),
        ),
 
        # vis_options=gs.options.VisOptions(
        #     show_world_frame=False,
        # ),
    )

    with timer():
        runner.run(
            seconds=args.seconds,
            # n_envs=100
        )
