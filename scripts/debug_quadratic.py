from musculotendon_ocp import (
    MuscleHillModels,
    ComputeForceDampingMethods,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
)
from casadi import MX, cos

muscle = MuscleHillModels.FlexibleTendon(
    name="Mus1",
    label="Quadratic",
    maximal_force=1000,
    optimal_length=0.2,
    tendon_slack_length=0.020000000000000004,
    compute_force_damping=ComputeForceDampingMethods.Linear(factor=0.1),
    maximal_velocity=5.0,
    compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(),
    compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
)


# resized_model.muscle_fiber_lengths_equilibrated(MX(1), MX(0.0770611), MX(0))

# muscle_tendon_length = 0.19900781249999988
# muscle_fiber_length = 0.1790571
# muscle_fiber_velocity_initial_guess = 0.0
# activation = 1.0

# muscle_tendon_length = 0.19900781249999988
# muscle_fiber_length = 0.1789139
# muscle_fiber_velocity_initial_guess = -1.432144
# activation = 1.0

muscle_tendon_length = 0.19900781249999988
muscle_fiber_length = 0.1789139
muscle_fiber_velocity_initial_guess = 0.5
activation = 1


tendon_length = muscle.compute_tendon_length(muscle_tendon_length, muscle_fiber_length)

# Compute some normalized values
normalized_length = muscle.normalize_muscle_fiber_length(muscle_fiber_length)
normalized_velocity = muscle.normalize_muscle_fiber_velocity(muscle_fiber_velocity_initial_guess)
pennation_angle = muscle.compute_pennation_angle(muscle_fiber_length)

# Compute the normalized forces
force_passive = muscle.compute_force_passive(normalized_length)
force_active = muscle.compute_force_active(normalized_length)
normalized_tendon_force = muscle.compute_tendon_force(tendon_length) / muscle.maximal_force

# Compute the derivatives
first_derivative = muscle.compute_force_velocity.first_derivative(normalized_velocity)
second_derivative = muscle.compute_force_velocity.second_derivative(normalized_velocity)

# Compute the polynomial coefficients of the Taylor expansion at second order of force-velocity relationship
biais = (
    muscle.compute_force_velocity(normalized_velocity)
    - first_derivative * normalized_velocity
    + second_derivative * normalized_velocity**2 / 2
)
slope = first_derivative - second_derivative * normalized_velocity
quadratic_coeff = second_derivative / 2

# Compute the polynomial coefficients of the differential equation of the muscle equilibrium equation
polynomial_quadratic_coeff = activation * force_active * quadratic_coeff
polynomial_slope = activation * force_active * slope + muscle.compute_force_damping.factor
polynomial_bias = force_passive - (normalized_tendon_force / cos(pennation_angle)) + biais * activation * force_active

# Compute the roots of the polynomial
discriminant = polynomial_slope**2 - 4 * polynomial_quadratic_coeff * polynomial_bias


if discriminant > 0:
    print("Two real roots")

if discriminant == 0:
    print("One real root")

if discriminant < 0:
    print("Two complex roots - not viable for muscle equilibrium equation")
