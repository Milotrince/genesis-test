import genesis as gs
from genesis.engine.entities import EntityPlugin, RigidEntity

class RigidForceSensor(EntityPlugin):

    def __init__(self):
        super().__init__(self)
    
    def setup(self, entity):
        assert isinstance(entity, RigidEntity), "RigidForceSensor can only be applied to RigidEntity."
        super().setup(entity)

    def step(self):
        pass

    def read(self, entity: RigidEntity):
        """
        Read the sensor data from the rigid entity.
        """
        if not isinstance(entity, RigidEntity):
            raise TypeError("RigidForceSensor can only be applied to RigidEntity.")
        
        # Example: return the force applied to the entity
        return entity.sim.get_rigid_force(entity.idx)


