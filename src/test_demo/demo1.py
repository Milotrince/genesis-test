import genesis as gs
from gs_runner import GenesisRunner

class Demo1Runner(GenesisRunner):
    def setup(self):
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )
    
    def step(self, i):
        pass

if __name__ == "__main__":
    runner = Demo1Runner(video_out="out/demo1.mp4")
    runner.run(steps=1000)
