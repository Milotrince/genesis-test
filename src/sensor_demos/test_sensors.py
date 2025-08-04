import torch

# from .utils import assert_allclose
from utils import assert_allclose, assert_array_equal

import genesis as gs


def test_rigid_tactile_sensors_gravity_force(show_viewer):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )

    bool_sensor = gs.sensors.RigidContactSensor(entity=box)
    force_sensor = gs.sensors.RigidContactForceSensor(entity=box)
    normtan_force_sensor = gs.sensors.RigidNormalTangentialForceSensor(entity=box)

    grid_size = (1, 1, 2)
    grid_bool_sensor = gs.sensors.RigidContactGridSensor(entity=box, grid_size=grid_size)
    grid_force_sensor = gs.sensors.RigidContactForceGridSensor(entity=box, grid_size=grid_size)
    grid_normtan_force_sensor = gs.sensors.RigidNormalTangentialForceGridSensor(entity=box, grid_size=grid_size)

    scene.build()

    assert not bool_sensor.read()[0], "RigidContactSensor should not be in contact with the ground yet."
    assert_array_equal(force_sensor.read(), 0.0), "RigidContactForceSensor should be zero before contact."
    (
        assert_array_equal(normtan_force_sensor.read(), 0.0),
        "RigidNormalTangentialForceSensor should be zero before contact.",
    )
    (
        assert_array_equal(grid_bool_sensor.read(), False),
        "RigidContactGridSensor should not be in contact with the ground yet.",
    )
    (
        assert_array_equal(grid_force_sensor.read(), 0.0),
        "RigidContactForceGridSensor should be zero before contact.",
    )
    (
        assert_array_equal(grid_normtan_force_sensor.read(), 0.0),
        "RigidNormalTangentialForceGridSensor should be zero before contact.",
    )

    for _ in range(500):
        scene.step()

    assert bool_sensor.read()[0], "RigidContactSensor should detect contact with the ground"
    (
        assert_allclose(force_sensor.read()[0], torch.tensor([0.0, 0.0, -GRAVITY]), tol=1e-5),
        "RigidContactForceSensor should detect force from gravity",
    )
    (
        assert_allclose(normtan_force_sensor.read()[0], torch.tensor([-GRAVITY, 0.0, 0.0, 0.0]), tol=1e-5),
        "RigidNormalTangentialForceSensor should detect force from gravity",
    )
    assert not grid_bool_sensor.read()[0, 0, 0, 1], "Top of RigidContactGridSensor should not be detecting contact"
    assert grid_bool_sensor.read()[0, 0, 0, 0], "Bottom of RigidContactGridSensor should detect contact with the ground"
    (
        assert_allclose(grid_force_sensor.read()[0, 0, 0, 1, :], torch.tensor([0.0, 0.0, 0.0]), tol=1e-9),
        "RigidContactForceGridSensor should detect zero force at the top of the box",
    )
    (
        assert_allclose(grid_force_sensor.read()[0, 0, 0, 0, :], torch.tensor([0.0, 0.0, -GRAVITY]), tol=1e-5),
        "RigidContactForceGridSensor should detect -gravity (normal) force at the bottom of the box",
    )

    (
        assert_allclose(grid_normtan_force_sensor.read()[0, 0, 0, 1, :], torch.tensor([0.0, 0.0, 0.0, 0.0]), tol=1e-9),
        "RigidNormalTangentialForceGridSensor should detect zero tangential force at the top of the box",
    )
    (
        assert_allclose(
            grid_normtan_force_sensor.read()[0, 0, 0, 0, :],
            torch.tensor([-GRAVITY, 0.0, 0.0, 0.0]),
            tol=1e-5,
        ),
        "RigidNormalTangentialForceGridSensor should detect -gravity (normal) force at the bottom of the box, with no tangential force",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--show_viewer", action="store_true", help="Show the viewer")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    test_rigid_tactile_sensors_gravity_force(args.show_viewer)
