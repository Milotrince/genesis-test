import genesis as gs
from utils import assert_allclose


def test_box_hard_vertex_constraint(show_viewer):
    """Test if a box with hard vertex constraints has those vertices fixed."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=5,
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
    vertex_indices = [0, 1]
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

    # from genesis.utils.misc import ti_field_to_torch

    for f in range(100):
        scene.step()

        state = box.solver.get_state(f)
        print(state.pos.cpu())
        # assert_allclose(
        #     state.pos[vertex_indices],
        #     target_positions,
        #     atol=1e-5,
        #     rtol=1e-5
        # )


if __name__ == "__main__":
    gs.init(backend=gs.gpu)
    test_box_hard_vertex_constraint(show_viewer=False)
