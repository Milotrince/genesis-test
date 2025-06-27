import genesis as gs
import numpy as np
import argparse
from sensor import *
from utils import *
from gs_runner import GenesisRunner

SCENE_POS = (0.5, 0.5, 1.0)

class SensorTestRunner(GenesisRunner):
    def __init__(self, item=None, **kwargs):
        super().__init__(**kwargs)
        self.item = item

    def setup(self):
        self.scene.add_entity(gs.morphs.Plane())

        pos = tuple(map(sum, zip(SCENE_POS, (0, 0, 0.2))))
        if self.item == "duck":
            self.scene.add_entity(
                material=gs.materials.MPM.Elastic(E=4.e4, rho=500),
                morph=gs.morphs.Mesh(
                    file="meshes/duck.obj",
                    pos=pos,
                    scale=0.07,
                    euler=(90.0, 0.0, 90.0),
                ),
                surface=gs.surfaces.Default(
                    color=(0.9, 0.8, 0.2, 1.0),
                ),
                vis_mode="particle",
            )
        elif self.item == "ball":
            self.scene.add_entity(
                material=gs.materials.Rigid(
                    rho=500.0,  # density
                    friction=0.5
                ),
                morph=gs.morphs.Sphere(pos=pos, radius=0.1),
                surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
            )
        

        self.sensor = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                # file="./objects/dome2.obj",
                # euler=(180.0, 0.0, 0.0),
                file="./objects/pad.obj",
                euler=(0.0, 90.0, 0.0),
                pos=tuple(map(sum, zip(SCENE_POS, (0, 0, 0)))),
            ),
            material=gs.materials.FEM.Elastic(
                E=1.e6, # stiffness
                # E=5.e5, # stiffness
                nu=0.47, # compressibility (0 to 0.5)
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
        # print(f"Sensor vertices: {verts.shape}")
        np.save("out/sensor_verts.npy", verts)

        # def get_verts(obj_file):
        #     import trimesh
        #     mesh = trimesh.load(obj_file, force="mesh", skip_texture=True)
        #     return mesh.vertices
        # pinned_vertices = get_verts("./objects/pad_verts_to_pin_opp.obj")
        # self.pinned_indices = np.array(get_pinned_vertex_indices(verts, pinned_vertices)) + self.sensor._v_start
        # self.pinned_indices = np.arange(0, 8, 1) + self.sensor._v_start 
        self.pinned_indices = np.arange(0, 2, 1) + self.sensor._v_start 
        print(f"Number of pinned vertices: {len(self.pinned_indices)}")

        init_vert_pos = self.sensor.init_positions.cpu().numpy()
        self.target_positions = init_vert_pos[self.pinned_indices]


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
            self.scene.draw_debug_spheres(poss=self.target_positions, radius=0.02, color=(1, 0, 1, 0.8))
            self.fem_solver.set_vertex_constraints(
                vertex_indices=self.pinned_indices,
                target_positions=self.target_positions,
                constraint_type="hard",
                # constraint_type="soft",
                # stiffness=1e4
            )
        if i % 100 == 0:
            forces = self.fem_solver.get_forces()
            self.all_forces.append(forces)
    
    def on_complete(self):
        all_forces = np.array(self.all_forces)
        print("Sensor forces shape:", all_forces.shape)
        forces_file = "out/forces.npy"
        np.save(forces_file, all_forces)
        gs.logger.info(f"Saved forces to {forces_file}")


if __name__ == "__main__":
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FEM Sensor Test with configurable solver")
    parser.add_argument(
        "--item", 
        type=str,
        default="",
    )
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

    args = parser.parse_args()
    
    if args.solver == "explicit":
        dt = args.dt if args.dt is not None else 1e-4
        substeps = args.substeps if args.substeps is not None else 5
    else:  # implicit
        dt = args.dt if args.dt is not None else 1e-3
        substeps = args.substeps if args.substeps is not None else 1
    

    with timer():
        SensorTestRunner(
            item=args.item,
            logging=True,

            video_out=get_next_filename(f"out/sensor_test_{args.solver}_dt={dt}_substeps={substeps}_##.mp4"),
            # camera_pos=(0, 4, 1),
            camera_pos=(-2, 3, 2),
            camera_lookat=tuple(map(sum, zip(SCENE_POS, (0, 0, -0.8)))),
            # vis_options=gs.options.VisOptions(show_world_frame=False),

            dt=dt,
            substeps=substeps,
            fem_options=gs.options.FEMOptions(
                gravity=(0, 0, 0) if args.item else (0, 0, -9.81),
                use_implicit_solver=args.solver == "implicit",
            ),
            mpm_options=gs.options.MPMOptions(
                gravity=(0, 0, -9.81),
            ),
        ) .run(
            seconds=args.seconds,
            # n_envs=100
        )
