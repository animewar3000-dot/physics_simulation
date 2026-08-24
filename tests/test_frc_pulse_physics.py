import numpy as np

from physics import KEV_TO_J, MU_0, calculate_frc_physics, fusion_reactivity


def test_dhe3_pressure_and_beta_use_kev_energy_without_boltzmann_constant() -> None:
    density = 3.0e20
    temperature = 50.0
    field = 8.0

    result = calculate_frc_physics(density, temperature, field)

    expected_pressure = 2.5 * density * temperature * KEV_TO_J
    expected_beta = expected_pressure / (field**2 / (2 * MU_0))
    assert np.isclose(result["plasma_pressure_pa"], expected_pressure)
    assert np.isclose(result["beta"], expected_beta)
    assert result["beta"] > 0.1


def test_dhe3_reactivity_peaks_near_seventy_kev() -> None:
    peak = fusion_reactivity(70.0, "D-He3")
    assert np.isclose(peak, 2.4e-22)
    assert peak > fusion_reactivity(25.0, "D-He3")
    assert peak > fusion_reactivity(180.0, "D-He3")


def test_dhe3_bremsstrahlung_uses_kev_coefficient() -> None:
    density = 3.0e20
    temperature = 50.0
    result = calculate_frc_physics(density, temperature, 8.0)
    expected = 1.69e-35 * (1.5 * density) * density * 1.667**2 * np.sqrt(temperature)
    assert np.isclose(result["brems_w_m3"], expected)
