import genesis as gs


def test_fem(show_viewer):
    """hi ziheng"""

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
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
        material=gs.materials.FEM.Elastic(),
    )
    verts_idx = [0, 1]
    target_poss = box.init_positions[verts_idx]

    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(poss=target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    box.set_velocity(gs.tensor([1.0, 1.0, 1.0]) * 1e-2)

    for _ in range(500):
        scene.step()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test FEM vertex constraints.")
    parser.add_argument(
        "--vis", "-v", action="store_true", help="Show visualization GUI"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Run on CPU (default is GPU if available)"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    test_fem(show_viewer=args.vis)
