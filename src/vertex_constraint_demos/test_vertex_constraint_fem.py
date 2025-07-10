import genesis as gs
import numpy as np
import torch
from genesis.utils.misc import tensor_to_array
from utils import timer

def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    if isinstance(actual, torch.Tensor):
        actual = tensor_to_array(actual)
    actual = np.asanyarray(actual)
    if isinstance(desired, torch.Tensor):
        desired = tensor_to_array(desired)
    desired = np.asanyarray(desired)

    if all(e.size == 0 for e in (actual, desired)):
        return

    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)


# these are meant to go in test_fem.py

def test_box_hard_vertex_constraint(show_viewer):
    """Test if a box with hard vertex constraints has those vertices fixed, 
    and verify updating and removing constraints works correctly."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    verts_idx = [0, 3]
    initial_target_poss = box.init_positions[verts_idx]
    
    scene.build(n_envs=2)

    if show_viewer:
        scene.draw_debug_spheres(
            poss=initial_target_poss,
            radius=0.02,
            color=(1, 0, 1, 0.8)
        )

    box.set_vertex_constraints(
        verts_idx=verts_idx,
        target_poss=initial_target_poss
    )

    for _ in range(100):
        scene.step()

    positions = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions, initial_target_poss, tol=0.0
    ), "Vertices should stay at initial target positions with hard constraints"
    new_target_poss = initial_target_poss + gs.tensor(
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], 
    )
    box.update_constraint_targets(
        verts_idx=verts_idx,
        target_poss=new_target_poss
    )

    for _ in range(100):
        scene.step()

    positions_after_update = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions_after_update,
        new_target_poss,
        tol=0.0
    ), "Vertices should be at new target positions after updating constraints"

    box.remove_vertex_constraints()

    for _ in range(100):
        scene.step()

    positions_after_removal = box.get_state().pos[0][verts_idx]

    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            positions_after_removal,
            new_target_poss,
            tol=1e-3
        ), "Vertices should have moved after removing constraints"


def test_box_soft_vertex_constraint(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.FEM.Elastic()
    )
    verts_idx = [0, 1]
    target_poss = box.init_positions[verts_idx]

    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(
            poss=target_poss,
            radius=0.02,
            color=(1, 0, 1, 0.8)
        )

    box.set_vertex_constraints(
        verts_idx=verts_idx,
        target_poss=target_poss,
        is_soft_constraint=True,
        stiffness=1.e7
    )
    box.set_velocity(gs.tensor([0.1, 0.1, 0.1]))

    with timer():
        for _ in range(1000):
            scene.step()

    positions = box.get_state().pos[0][verts_idx]

    assert_allclose(
        positions,
        target_poss,
        tol=5e-5
    ), "Vertices should be near target positions with strong soft constraints"



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FEM vertex constraints.")
    parser.add_argument("--vis", "-v", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    # print("test_box_hard_vertex_constraint")
    # test_box_hard_vertex_constraint(show_viewer=args.vis)

    print("test_box_soft_vertex_constraint")
    test_box_soft_vertex_constraint(show_viewer=args.vis)
