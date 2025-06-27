import genesis as gs
import numpy as np
import torch
import argparse
from sensor import *
from utils import *
from gs_runner import GenesisRunner

SCENE_POS = (0.5, 0.5, 1.0)

class PinVertexTestRunner(GenesisRunner):
    def setup(self):
        self.scene.add_entity(gs.morphs.Plane())

        self.blob = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=tuple(map(sum, zip(SCENE_POS, (-0.3, -0.3, 0)))),
                radius=0.1,
            ),
            material=gs.materials.FEM.Elastic(
                E=1.e4, # stiffness
                nu=0.49, # compressibility (0 to 0.5)
                rho=1000.0, # density
                model='stable_neohookean'
            ),
        )
        self.cube = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=tuple(map(sum, zip(SCENE_POS, (0.3, 0.3, 0)))),
                size=(0.2, 0.2, 0.2),
            ),
            material=gs.materials.FEM.Elastic(
                E=1.e6, # stiffness
                nu=0.49, # compressibility (0 to 0.5)
                rho=1000.0, # density
                model='stable_neohookean'
            ),
        )

        self.pin_idx = np.arange(0, 1, 1)  # pin one

        self.debug_circle = None
        self._circle_offset = None



    def step(self, i):
        if i == 0:
            target_positions = self.blob.init_positions[self.pin_idx]
            self.scene.draw_debug_spheres(poss=target_positions, radius=0.02, color=(1, 0, 1, 0.8))
            self.blob.set_vertex_constraints(
                vertex_indices=self.pin_idx,
                target_positions=target_positions,
                constraint_type="soft",
                stiffness=1e2,
                damping=1e1,
            )

            self._circle_offset = self.cube.init_positions[self.pin_idx] - self._get_circle_path(i)
            target_positions = self._get_circle_path(i)
            self.debug_circle = self.scene.draw_debug_spheres(poss=target_positions, radius=0.02, color=(1, 0, 1, 0.8))
            self.cube.set_vertex_constraints(
                vertex_indices=self.pin_idx,
                target_positions=target_positions,
            )

        else:
            # set cube target position to move in circle
            self.scene.clear_debug_object(self.debug_circle)
            new_pos = self._get_circle_path(i)
            self.debug_circle = self.scene.draw_debug_spheres(poss=new_pos, radius=0.02, color=(1, 0, 1, 0.8))
            self.cube.update_constraint_targets(
                vertex_indices=self.pin_idx,
                target_positions=new_pos,
            )
    
    def _get_circle_path(self, i):
        rate = self.dt * i / 10.0
        pos = self.cube.init_positions[self.pin_idx] + torch.Tensor([
            -0.3 * np.cos(2 * np.pi * rate),
            -0.3 * np.sin(2 * np.pi * rate),
            0.0
        ]).to(self.cube.init_positions.device)
        if self._circle_offset is not None:
            pos += self._circle_offset
        return pos

    
    def on_complete(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        PinVertexTestRunner(
            logging=True,

            video_out=get_next_filename(f"out/sensor_test_{args.solver}_dt={dt}_substeps={substeps}_##.mp4"),
            camera_pos=(-2, 3, 2),
            camera_lookat=tuple(map(sum, zip(SCENE_POS, (0, 0, -0.8)))),

            dt=dt,
            substeps=substeps,
            fem_options=gs.options.FEMOptions(
                use_implicit_solver=args.solver == "implicit",
            ),
        ) .run(seconds=args.seconds)
