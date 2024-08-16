from typing import Callable, override

from casadi import MX

from .pennation_angle import PennationAngleConstant
from .force_active import ForceActiveHillType
from .force_damping import ForceDampingConstant
from .force_passive import ForcePassiveHillType
from .force_velocity import ForceVelocityHillType
from ..muscle_model_abstract import MuscleModelAbstract

"""
Compute the pennation angle

Parameters
----------
muscle_length: MX
    The muscle length
element: MX
    The element to apply the pennation angle to

Returns
-------
MX
    The pennation angle corresponding to the given muscle length
"""
type PennationAngleCallable = Callable[[MX, MX], MX]

"""
Returns the normalized force from the passive force-length relationship

Parameters
----------
normalized_muscle_length: MX
    The normalized muscle length

Returns
-------
MX
    The normalized passive force corresponding to the given muscle length
"""
type ForcePassiveCallable = Callable[[MX], MX]

"""
Compute the normalized force from the active force-length relationship

Parameters
----------
normalized_muscle_length: MX
    The normalized muscle length

Returns
-------
MX
    The normalized active force corresponding to the given muscle length
"""
type ForceActiveCallable = Callable[[MX], MX]


"""
Compute the normalized force from the force-velocity relationship

Parameters
----------
normalized_muscle_length: MX
    The normalized muscle length
normalized_muscle_velocity: MX
    The normalized muscle velocity

Returns
-------
MX
    The normalized force corresponding to the given muscle length and velocity
"""
type ForceVelocityCallable = Callable[[MX, MX], MX]


"""
Compute the normalized force from the damping

Parameters
----------
normalized_muscle_velocity: MX
    The normalized muscle velocity

Returns
-------
MX
    The normalized force corresponding to the given muscle velocity
"""
type ForceDampingCallable = Callable[[MX], MX]


class MuscleModelHillRigidTendon(MuscleModelAbstract):
    def __init__(
        self,
        name: str,
        maximal_force: MX,
        optimal_length: MX,
        maximal_velocity: MX = 5.0,
        pennation_angle: PennationAngleCallable = PennationAngleConstant(),
        force_passive: ForcePassiveCallable = ForcePassiveHillType(),
        force_active: ForceActiveCallable = ForceActiveHillType(),
        force_velocity: ForceVelocityCallable = ForceVelocityHillType(),
        force_damping: ForceDampingCallable = ForceDampingConstant(),
    ):
        """
        Parameters
        ----------
        name: str
            The muscle name
        maximal_force: MX
            The maximal force the muscle can produce
        optimal_length: MX
            The optimal length of the muscle
        maximal_velocity: MX
            The maximal velocity of the muscle
        pennation_angle: PennationAngleCallable
            The pennation angle function
        force_passive: ForcePassiveCallable
            The passive force-length relationship function
        force_active: ForceActiveCallable
            The active force-length relationship function
        force_velocity: ForceVelocityCallable
            The force-velocity relationship function
        force_damping: ForceDampingCallable
            The damping function
        """
        super().__init__(name=name)

        if maximal_force < 0:
            raise ValueError("The maximal force must be positive")
        self.maximal_force = maximal_force
        if optimal_length < 0:
            raise ValueError("The optimal length must be positive")
        self.optimal_length = optimal_length
        if maximal_velocity < 0:
            raise ValueError("The maximal velocity must be positive")
        self.maximal_velocity = maximal_velocity

        self.pennation_angle = pennation_angle

        self.force_passive = force_passive
        self.force_active = force_active
        self.force_velocity = force_velocity
        self.force_damping = force_damping

    @override
    def normalize_muscle_length(self, muscle_fiber_length: MX) -> MX:
        return muscle_fiber_length / self.optimal_length

    @override
    def normalize_muscle_velocity(self, muscle_fiber_velocity: MX) -> MX:
        return muscle_fiber_velocity / self.maximal_velocity

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        raise RuntimeError("The tendon length should not be normalized for this muscle model")

    @override
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        # Get the normalized muscle length and velocity
        normalized_length = self.normalize_muscle_length(muscle_fiber_length)
        normalized_velocity = self.normalize_muscle_velocity(muscle_fiber_velocity)

        # Compute the passive, active, velocity and damping factors
        force_passive = self.force_passive(normalized_length)
        force_active = self.force_active(normalized_length)
        force_velocity = self.force_velocity(normalized_velocity)
        force_damping = self.force_damping(normalized_velocity)

        return self.pennation_angle(
            muscle_fiber_length,
            self.maximal_force * (force_passive + activation * force_active * force_velocity + force_damping),
        )

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return 0
