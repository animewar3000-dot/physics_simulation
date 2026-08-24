import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dashboard import TelemetryDashboard, TelemetryData
from physics import ELEMENTARY_CHARGE, PROTON_MASS, ParticleState


def test_particle_magnetic_quantities_accept_field_strength() -> None:
    particle = ParticleState(
        position=np.zeros(3),
        velocity=np.array([3.0, 4.0, 0.0]),
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
    )

    assert particle.gyroradius(2.0) == 2.5 * PROTON_MASS / ELEMENTARY_CHARGE
    assert particle.gyrofrequency(2.0) == 2.0 * ELEMENTARY_CHARGE / PROTON_MASS

    with pytest.raises(ValueError, match="matching shapes"):
        ParticleState(
            position=np.zeros(3),
            velocity=np.zeros((2, 3)),
            mass=PROTON_MASS,
            charge=ELEMENTARY_CHARGE,
        )


def test_three_dimensional_axis_profiles_use_matching_coordinates() -> None:
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    telemetry = TelemetryData(B_field=(values + 100, np.zeros_like(values), values))

    axial = telemetry.get_field_components_along_axis(
        "z", z_coords=np.array([-2.0, -1.0, 0.0, 1.0])
    )
    radial = telemetry.get_field_components_along_axis(
        "r", r_coords=np.array([0.0, 0.1, 0.2])
    )

    np.testing.assert_array_equal(axial["B_z"], values[0, 0, :])
    np.testing.assert_array_equal(axial["z"], [-2.0, -1.0, 0.0, 1.0])
    np.testing.assert_array_equal(radial["B_z"], values[0, :, 2])
    np.testing.assert_array_equal(radial["r"], [0.0, 0.1, 0.2])


def test_dashboard_handles_zero_power_and_three_dimensional_profiles() -> None:
    field = np.ones((2, 3, 4))
    density = np.arange(24, dtype=float).reshape(2, 3, 4) + 1.0
    temperature = density * 10
    dashboard = TelemetryDashboard()
    telemetry = TelemetryData(
        B_field=(field, np.zeros_like(field), field),
        plasma_density=density,
        plasma_temperature=temperature,
    )

    dashboard.update(telemetry, {"r_grid": np.array([0.0, 0.1, 0.2])})

    density_line, temperature_line = dashboard.axes["plasma_profiles"].lines
    np.testing.assert_array_equal(density_line.get_ydata(), density[0, :, 2])
    np.testing.assert_array_equal(temperature_line.get_ydata(), temperature[0, :, 2])
