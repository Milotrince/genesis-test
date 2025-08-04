import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=""):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    args = [actual, desired]
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            arg = tensor_to_array(arg)
        elif isinstance(arg, (tuple, list)):
            arg = [tensor_to_array(val) for val in arg]
        args[i] = np.asanyarray(arg)

    if all(e.size == 0 for e in args):
        return

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer=True,
    sim_options=gs.options.SimOptions(
        dt=0.01,
        substeps=1,
    ),
)

sphere = scene.add_entity(gs.morphs.Sphere())
scene.build(n_envs=7)

scene.sim.set_gravity(torch.tensor([-1.0, 0.0, 0.0]))
scene.sim.set_gravity(torch.tensor([[-2.0, 0.0, 0.0]] * 7))
scene.sim.set_gravity(torch.tensor([0.0, 0.0, 0.0]), envs_idx=[0, 1])
scene.sim.set_gravity(torch.tensor([0.0, 0.0, 100.0]), envs_idx=3)
scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -1.0]]), envs_idx=[2, 4])

with np.testing.assert_raises(AssertionError):
    scene.sim.set_gravity(torch.tensor([0.0, -10.0]))

with np.testing.assert_raises(AssertionError):
    scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]), envs_idx=1)

scene.step()

print(sphere.get_links_acc())

assert_allclose(
    np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 100.0],
            [0.0, 0.0, -1.0],
            [-2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    ),
    sphere.get_links_acc().squeeze(dim=1),
    tol=1e-6,
)