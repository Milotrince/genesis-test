import argparse
import genesis as gs
import os
from gs_runner import GenesisRunner

class SoftBallRunner(GenesisRunner):
    def __init__(self, fem_model, dt=1e-2, substeps=1, logging=True, backend=gs.gpu, gui=False, video_out="output.mp4", **scene_kwargs):
        super().__init__(dt=dt, substeps=substeps, logging=logging, backend=backend, gui=gui, video_out=video_out, **scene_kwargs)
        self.fem_model = fem_model

    def setup(self):
        self.scene.add_entity(gs.morphs.Plane())

        E = 1.e4 # 1e6; Young's modulus; higher values make the ball stiffer
        nu = 0.45 # Poisson's ratio; 0.5 is incompressible, 0.45 is a good approximation
        rho = 1000. # Density; higher values make the ball heavier

        self.scene.add_entity(
            morph=gs.morphs.Sphere(pos=(0.3,0,0.3), radius=0.2),
            material=gs.materials.FEM.Elastic(
                E=E, nu=nu, rho=rho,
                model=self.fem_model
            )
        )
    
    def step(self, i):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, required=True, help='Time step size')
    parser.add_argument('--substeps', type=int, required=True, help='Substeps per frame')
    parser.add_argument('--solver', choices=['explicit', 'implicit'], required=True, help='FEM solver type')
    parser.add_argument('--model', choices=['linear', 'stable_neohookean'], required=True, help='FEM model type')
    parser.add_argument('--seconds', type=float, default=1, help='Duration of the simulation in seconds')
    parser.add_argument('--out', type=str, default="out", help='output folder')
    args = parser.parse_args()

    video_out = f"{args.out}/{args.model}/{args.solver}/{args.substeps}/{args.dt}/softball_{args.solver}_{args.substeps}_{args.dt}.mp4"

    if os.path.exists(video_out):
        print(f"Video file {video_out} already exists. Skipping simulation.")
        exit(0)

    SoftBallRunner(
        fem_model=args.model,
        logging=False,
        video_out=video_out,
        dt=args.dt,
        substeps=args.substeps,
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=(args.solver == 'implicit')
        ),
    ).run(seconds=1)
