import numpy as np
import genesis as gs
from gs_runner import GenesisRunner

class Demo3Runner(GenesisRunner):
    def setup(self):
        self.scene.add_entity(gs.morphs.Plane())

        self.robot_mpm = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.5, 0.2, 0.3),
                radius=0.1,
            ),
            material=gs.materials.MPM.Muscle(
                E=3.e4,
                nu=0.45,
                rho=1000.,
                model='neohooken',
            ),
        )
        self.robot_fem = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(0.5, -0.2, 0.3),
                radius=0.1,
            ),
            material=gs.materials.FEM.Muscle(
                E=3.e4,
                nu=0.45,
                rho=1000.,
                model='stable_neohooken',
            ),
        )

    
    def step(self, i):
        actu = np.array([0.2 * (0.5 + np.sin(0.01 * np.pi * i))])
        self.robot_mpm.set_actuation(actu)
        self.robot_fem.set_actuation(actu)

if __name__ == "__main__":
    dt = 5e-4
    runner = Demo3Runner(video_out="out/demo3.mp4",
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            dt=dt,
            lower_bound=(-1.0, -1.0, -0.2),
            upper_bound=( 1.0,  1.0,  1.0),
        ),
        fem_options=gs.options.FEMOptions(
            dt=dt,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        )
    )
    runner.run(steps=1000)