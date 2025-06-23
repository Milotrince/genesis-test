import genesis as gs
from abc import ABC, abstractmethod


class GenesisRunner(ABC):
    def __init__(
        self,
        dt=1e-2,
        substeps=1,
        logging=True,
        show_progress=True,
        backend=gs.gpu,
        gui=False,
        video_out="output.mp4",
        camera_pos=(0, 3, 2),
        camera_lookat=(0, 0, 0.5),
        camera_fov=30,
        camera_res=(640, 480),
        show_fps=False,
        **scene_kwargs,
    ):
        self.gui = gui
        self.video_out = video_out
        self.video_fps = 1 / dt
        self.show_progress = show_progress

        gs.init(backend=backend, logging_level=None if logging else "warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=dt,
                substeps=substeps,
                gravity=(0, 0, -9.81),
            ),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=show_fps,
            ),
            show_viewer=gui,
            **scene_kwargs,
        )
        self.cam = self.scene.add_camera(
            res=camera_res, pos=camera_pos, lookat=camera_lookat, fov=camera_fov, GUI=self.gui
        )


    def run(self, seconds=None, steps=None, **build_kwargs):
        """
        Arguments:
            steps: int, optional
                Specify the number of steps to run the simulation.
            seconds: float, optional
                Specify the duration in real-time seconds (based on substep time) to run the simulation.
                Will override `steps` if both are provided.
        """
        if not steps:
            if not seconds:
                gs.logger.warning("Provide `steps` or `seconds` to run the simulation.")
                return
            steps = int(self.video_fps * seconds)

        self.setup()
        self.scene.build(**build_kwargs)
        self.cam.start_recording()

        if seconds:
            gs.logger.info(f"Running simulation for {seconds} seconds ({steps} steps with dt={self.scene.sim_options.dt} at {self.video_fps} FPS)")
        else:
            gs.logger.info(f"Running simulation for {steps} steps.")

        try:
            steps_iter = range(steps)
            if self.show_progress:
                from tqdm import tqdm
                steps_iter = tqdm(steps_iter, desc="simulation steps", total=steps)
            for i in steps_iter:
                self.step(i)
                self.scene.step()
                self.cam.render()
        except KeyboardInterrupt:
            gs.logger.info("Simulation interrupted, exiting.")
        finally:
            gs.logger.info("Simulation finished.")
            self.cam.stop_recording(save_to_filename=self.video_out, fps=self.video_fps)
            gs.logger.info(f"Saved video to {self.video_out}")
            self.on_complete()

    @abstractmethod
    def on_complete(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def step(self, i):
        pass
