import genesis as gs
import numpy as np

def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if all(isinstance(e, np.ndarray) and e.size == 0 for e in (actual, desired)):
        return
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)


# these are meant to go in test_fem.py

def test_box_hard_vertex_constraint(show_viewer):
    """Test if a box with hard vertex constraints has those vertices fixed."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
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
    target_positions = box.init_positions[vertex_indices]
    if show_viewer:
        scene.draw_debug_spheres(
            poss=target_positions,
            radius=0.02,
            color=(1, 0, 1, 0.8)
        )

    scene.build()

    box.set_vertex_constraints(
        vertex_indices=vertex_indices,
        target_positions=target_positions,
        constraint_type="hard"
    )

    for _ in range(1000):
        scene.step()

    positions = box.solver.get_state(0).pos.cpu()
    print(positions[0, vertex_indices, :])
    print(target_positions.cpu())

    assert_allclose(
        positions[0, vertex_indices, :],
        target_positions.cpu(),
        atol=1e-5,
        rtol=1e-5
    )


def test_box_soft_vertex_constraint(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=5,
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
        constraint_type="soft",
        stiffness=1.e3,
        damping=1.e1,
    )

    for _ in range(1000):
        scene.step()

    positions = box.solver.get_state(0).pos.cpu()
    print(positions[0, vertex_indices, :])
    print(target_positions.cpu())

    assert_allclose(
        positions[0, vertex_indices, :],
        target_positions.cpu(),
        atol=1e-2,
        rtol=1e-1
    )



if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    print("test_box_hard_vertex_constraint")
    test_box_hard_vertex_constraint(show_viewer=False)
    print("test_box_soft_vertex_constraint")
    test_box_soft_vertex_constraint(show_viewer=False)
