import genesis as gs
import numpy as np
import torch
from genesis.utils.misc import tensor_to_array

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
            substeps=2,
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
    vertex_indices = [0, 3]
    initial_target_positions = box.init_positions[vertex_indices]
    
    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(
            poss=initial_target_positions,
            radius=0.02,
            color=(1, 0, 1, 0.8)
        )

    box.set_vertex_constraints(
        vertex_indices=vertex_indices,
        target_positions=initial_target_positions
    )

    for _ in range(100):
        scene.step()

    positions = box.get_state().pos[0][vertex_indices]
    assert_allclose(
        positions, initial_target_positions, tol=0.0
    ), "Vertices should stay at initial target positions with hard constraints"
    new_target_positions = initial_target_positions + gs.tensor(
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], 
    )
    box.update_constraint_targets(
        vertex_indices=vertex_indices,
        target_positions=new_target_positions
    )

    for _ in range(100):
        scene.step()

    positions_after_update = box.get_state().pos[0][vertex_indices]
    assert_allclose(
        positions_after_update,
        new_target_positions,
        tol=0.0
    ), "Vertices should be at new target positions after updating constraints"

    box.remove_vertex_constraints(vertex_indices)

    for _ in range(100):
        scene.step()

    positions_after_removal = box.get_state().pos[0][vertex_indices]

    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            positions_after_removal,
            new_target_positions,
            tol=1e-3
        ), "Vertices should have moved after removing constraints"


def test_box_soft_vertex_constraint(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-4,
            substeps=10,
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
        material=gs.materials.FEM.Elastic(),
    )
    vertex_indices = [0]
    target_positions = box.init_positions[vertex_indices]

    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(
            poss=target_positions,
            radius=0.02,
            color=(1, 0, 1, 0.8)
        )

    box.set_vertex_constraints(
        vertex_indices=vertex_indices,
        target_positions=target_positions,
        is_soft_constraint=True,
        stiffness=1.e9
    )
    box.set_velocity(gs.tensor([0.0, 1.0, 0.0]))

    for _ in range(1000):
        scene.step()

    positions = box.get_state().pos[0][vertex_indices]

    assert_allclose(
        positions,
        target_positions,
        tol=1e-5
    ), "Vertices should be near target positions with strong soft constraints"



if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    # print("test_box_hard_vertex_constraint")
    # test_box_hard_vertex_constraint(show_viewer=False)
    print("test_box_soft_vertex_constraint")
    test_box_soft_vertex_constraint(show_viewer=False)
