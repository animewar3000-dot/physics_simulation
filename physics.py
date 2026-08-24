"""
physics.py - Physics Engine for FRC Simulation

This module implements the core physics solvers for particle dynamics and
plasma behavior in a Field-Reversed Configuration (FRC) reactor.

Key Physics:
    1. Lorentz Force: F = q(E + v × B)
    2. Rigid-Rotor Profile: Plasma rotation model for FRC equilibrium
    3. Separatrix: Boundary between closed and open field lines

Constants:
    e: Elementary charge = 1.602 × 10⁻¹⁹ C
    m_p: Proton mass = 1.673 × 10⁻²⁷ kg
    m_e: Electron mass = 9.109 × 10⁻³¹ kg
"""

import numpy as np
from typing import Tuple, Optional, Callable, List, Dict
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from scipy.constants import e, m_p, m_e
import warnings


# Physical constants (also available in scipy.constants)
ELEMENTARY_CHARGE = e  # 1.602176634e-19 C
PROTON_MASS = m_p      # 1.67262192369e-27 kg
ELECTRON_MASS = m_e    # 9.1093837015e-31 kg
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space [T·m/A]
K_B = 1.380649e-23
KEV_TO_J = 1.0e3 * ELEMENTARY_CHARGE


@dataclass(frozen=True)
class PulseParameters:
    """Inputs to the zero-dimensional pulsed-FRC energy balance model.

    This intentionally is a reactor *concept* model, not a design-certified
    transport calculation.  Temperatures are ion/electron-equilibrated keV,
    densities are total ion densities, and the initial plasma is an ellipsoid.
    """

    b_ext: float = 8.0
    density: float = 3.0e20
    temperature_kev: float = 45.0
    compression_ratio: float = 4.0
    duration_ms: float = 8.0
    fuel: str = "D-He3"
    initial_radius_m: float = 0.32
    initial_length_m: float = 1.2
    gamma: float = 5.0 / 3.0
    coil_turns: int = 40
    coil_resistance_ohm: float = 0.025
    coupling_efficiency: float = 0.82

    def __post_init__(self) -> None:
        if self.b_ext <= 0 or self.density <= 0 or self.temperature_kev <= 0:
            raise ValueError("Field, density, and temperature must be positive")
        if self.compression_ratio < 1 or self.duration_ms <= 0:
            raise ValueError("Compression ratio must be >= 1 and duration positive")
        if self.fuel not in {"D-He3", "D-T"}:
            raise ValueError("fuel must be 'D-He3' or 'D-T'")


def fusion_reactivity(temperature_kev: np.ndarray | float, fuel: str = "D-He3") -> np.ndarray:
    """Return Maxwellian-averaged fusion reactivity in m³/s.

    The smooth fits retain the Gamow rise and high-temperature roll-off of
    published D--He3/D--T reactivity curves in the 1--300 keV range.  They are
    deliberately bounded outside that range to keep an interactive ODE stable.
    """
    t = np.clip(np.asarray(temperature_kev, dtype=float), 0.2, 400.0)
    if fuel == "D-He3":
        # Log-normal fit to the D--He3 Maxwellian reactivity peak (~70 keV).
        # It avoids the low-temperature overprediction of a simple T² fit.
        rate = 2.4e-22 * np.exp(-((np.log(t) - np.log(70.0)) ** 2) / 1.5)
    elif fuel == "D-T":
        rate = 8.7e-22 * (t / 65.0) ** 1.55 * np.exp(1.55 * (1.0 - t / 65.0))
    else:
        raise ValueError("Unsupported fuel")
    return np.maximum(rate, 0.0)


def calculate_frc_physics(
    n_ion_m3: np.ndarray | float,
    temperature_kev: np.ndarray | float,
    b_ext_t: np.ndarray | float,
    compression_ratio: float = 1.0,
    fuel: str = "D-He3",
) -> Dict[str, np.ndarray]:
    """Calculate internally consistent local power-density and beta diagnostics.

    ``temperature_kev * KEV_TO_J`` is already the thermal energy per particle;
    therefore no extra Boltzmann constant belongs in the pressure expression.
    For an equimolar D--He3 mixture, quasi-neutrality gives ``n_e=1.5 n_i``
    and the total particle density is ``2.5 n_i``.
    """
    n_i = np.asarray(n_ion_m3, dtype=float)
    temp = np.clip(np.asarray(temperature_kev, dtype=float), 0.15, None)
    b_field = np.asarray(b_ext_t, dtype=float)
    if np.any(n_i <= 0) or np.any(b_field <= 0):
        raise ValueError("Ion density and magnetic field must be positive")

    if fuel == "D-He3":
        n_e, n_total, z_eff, fusion_mev = 1.5 * n_i, 2.5 * n_i, 1.667, 18.4
        # This coefficient expects T in keV and is not converted again.
        p_brems = 1.69e-35 * n_e * n_i * z_eff**2 * np.sqrt(temp)
    elif fuel == "D-T":
        n_e, n_total, z_eff, fusion_mev = n_i, 2.0 * n_i, 1.0, 17.6
        p_brems = 5.34e-37 * n_e * n_i * np.sqrt(temp)
    else:
        raise ValueError("Unsupported fuel")

    p_plasma = n_total * temp * KEV_TO_J
    p_magnetic = b_field**2 / (2.0 * MU_0)
    p_fusion = 0.25 * n_i**2 * fusion_reactivity(temp, fuel) * fusion_mev * 1e6 * ELEMENTARY_CHARGE
    p_sync = 1.0e-38 * n_e * b_field**2 * temp
    p_net = p_fusion - p_brems - p_sync
    return {"plasma_pressure_pa": p_plasma, "magnetic_pressure_pa": p_magnetic,
            "beta": p_plasma / p_magnetic, "fusion_w_m3": p_fusion,
            "brems_w_m3": p_brems, "sync_w_m3": p_sync, "net_w_m3": p_net,
            "particle_density_m3": n_total}


def _pulse_volume(time_s: np.ndarray, params: PulseParameters) -> np.ndarray:
    """Prescribed compression then expansion geometry used by the coupled ODE."""
    t_end = params.duration_ms * 1e-3
    burn_end = 0.62 * t_end
    initial = 4.0 / 3.0 * np.pi * params.initial_radius_m**2 * (params.initial_length_m / 2.0)
    fraction = np.where(time_s <= burn_end, time_s / burn_end, 1.0 - (time_s - burn_end) / (t_end - burn_end))
    # Compression ratio is a volume ratio; return smoothly to the initial volume.
    return initial / (1.0 + (params.compression_ratio - 1.0) * np.clip(fraction, 0.0, 1.0))


def simulate_frc_pulse(params: PulseParameters, samples: int = 360) -> Dict[str, np.ndarray | str | bool]:
    """Solve a tightly coupled 0D FRC temperature/energy balance with ``solve_ivp``.

    The ODE contains adiabatic work ``-(gamma-1) T d(ln V)/dt`` and fusion,
    bremsstrahlung, and synchrotron terms. Density follows particle conservation
    exactly (``n V = constant``), so each loss and fusion term feeds back into
    the thermal state at every integrator step.
    """
    t_end = params.duration_ms * 1e-3
    time = np.linspace(0.0, t_end, samples)
    v0 = float(_pulse_volume(np.array([0.0]), params)[0])
    n_particles = params.density * v0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        temp = max(float(y[0]), 0.15)
        dt = max(1e-8, t_end * 1e-5)
        volume = float(_pulse_volume(np.array([t]), params)[0])
        dlnv_dt = (np.log(float(_pulse_volume(np.array([min(t + dt, t_end)]), params)[0])) -
                    np.log(float(_pulse_volume(np.array([max(t - dt, 0.0)]), params)[0]))) / (min(t + dt, t_end) - max(t - dt, 0.0) + 1e-15)
        density = n_particles / volume
        b = params.b_ext * np.sqrt(v0 / volume)
        local = calculate_frc_physics(density, temp, b, params.compression_ratio, params.fuel)
        # U = (3/2) n_total T.  n_total includes ions and electrons.
        heat_capacity = 1.5 * float(local["particle_density_m3"]) * KEV_TO_J
        return [-(params.gamma - 1.0) * temp * dlnv_dt + float(local["net_w_m3"]) / heat_capacity]

    solution = solve_ivp(rhs, (0.0, t_end), [params.temperature_kev], t_eval=time,
                         method="RK45", rtol=2e-6, atol=1e-7, max_step=t_end / 240.0)
    if not solution.success:
        raise RuntimeError(f"Pulse integration failed: {solution.message}")

    temp = np.maximum(solution.y[0], 0.15)
    volume = _pulse_volume(time, params)
    density = n_particles / volume
    field = params.b_ext * np.sqrt(v0 / volume)
    local = calculate_frc_physics(density, temp, field, params.compression_ratio, params.fuel)
    pfus, pbrem = local["fusion_w_m3"], local["brems_w_m3"]
    psync, pnet, beta = local["sync_w_m3"], local["net_w_m3"], local["beta"]
    radius = params.initial_radius_m * (volume / v0) ** (1.0 / 3.0)
    flux = field * np.pi * radius**2
    emf = -params.coil_turns * np.gradient(flux, time)
    # A passive resistive load sees I=EMF/R, then coupling reduces delivered power.
    current = emf / params.coil_resistance_ohm
    pelec = params.coupling_efficiency * emf * current if params.fuel == "D-He3" else np.zeros_like(emf)
    unstable = bool(np.any(beta > 1.0))
    net_positive = bool(np.max(pnet) > 0.0 and np.trapz(pelec, time) > 0.0)
    status = "UNSTABLE" if unstable else ("IGNITION / NET POSITIVE" if net_positive else "SUB-CRITICAL")
    return {"time_s": time, "temperature_kev": temp, "density_m3": density, "volume_m3": volume,
            "field_t": field, "fusion_w_m3": pfus, "brems_w_m3": pbrem, "sync_w_m3": psync,
            "net_w_m3": pnet, "beta": beta, "emf_v": emf, "electric_w": pelec,
            "unstable": unstable, "status": status, "fuel": params.fuel}


@dataclass
class ParticleState:
    """
    Represents the state of a single particle or particle ensemble.
    
    Attributes:
        position: 3D position vector (x, y, z) or (r, θ, z) [m]
        velocity: 3D velocity vector (vx, vy, vz) or (vr, vθ, vz) [m/s]
        mass: Particle mass [kg]
        charge: Particle charge [C]
        label: Optional identifier for the particle
    """
    position: np.ndarray  # Shape: (3,) or (N, 3) for multiple particles
    velocity: np.ndarray  # Shape: (3,) or (N, 3) for multiple particles
    mass: float
    charge: float
    label: str = "particle"
    
    def __post_init__(self):
        """Validate input arrays."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        
        if self.position.shape[-1] != 3:
            raise ValueError("Position must be 3D vector")
        if self.velocity.shape[-1] != 3:
            raise ValueError("Velocity must be 3D vector")
        if self.position.shape != self.velocity.shape:
            raise ValueError("Position and velocity must have matching shapes")
        if self.position.ndim not in (1, 2):
            raise ValueError("Position and velocity must be shape (3,) or (N, 3)")
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
    
    @property
    def kinetic_energy(self) -> float | np.ndarray:
        """Calculate kinetic energy: KE = ½mv²"""
        v_squared = np.sum(self.velocity**2, axis=-1)
        return 0.5 * self.mass * v_squared
    
    @property
    def momentum(self) -> np.ndarray:
        """Calculate momentum: p = mv"""
        return self.mass * self.velocity
    
    def gyroradius(self, B_magnitude: float) -> float | np.ndarray:
        """
        Calculate Larmor gyroradius: r_L = mv_perp / (|q|B)
        
        Args:
            B_magnitude: Magnetic field strength [T]
            
        Returns:
            Gyroradius [m]
        """
        v_perp = np.linalg.norm(self.velocity, axis=-1)  # Approximation
        return self.mass * v_perp / (np.abs(self.charge) * B_magnitude)
    
    def gyrofrequency(self, B_magnitude: float) -> float | np.ndarray:
        """
        Calculate cyclotron frequency: ω_c = |q|B / m
        
        Args:
            B_magnitude: Magnetic field strength [T]
            
        Returns:
            Gyrofrequency [rad/s]
        """
        return np.abs(self.charge) * B_magnitude / self.mass


@dataclass
class FieldConfiguration:
    """
    Container for electromagnetic field data on a grid.
    
    This class provides interpolation methods to get field values
    at arbitrary positions within the simulation domain.
    
    Attributes:
        E_x, E_y, E_z: Electric field components on grid [V/m]
        B_x, B_y, B_z: Magnetic field components on grid [T]
        x_grid, y_grid, z_grid: Coordinate arrays for the fields
    """
    E_x: Optional[np.ndarray] = None
    E_y: Optional[np.ndarray] = None
    E_z: Optional[np.ndarray] = None
    B_x: Optional[np.ndarray] = None
    B_y: Optional[np.ndarray] = None
    B_z: Optional[np.ndarray] = None
    x_grid: Optional[np.ndarray] = None
    y_grid: Optional[np.ndarray] = None
    z_grid: Optional[np.ndarray] = None
    
    def get_fields_at_point(self, x: float, y: float, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate E and B fields at a specific point.
        
        Uses trilinear interpolation for smooth field values.
        
        Args:
            x, y, z: Position coordinates [m]
            
        Returns:
            Tuple of (E, B) vectors at the point
        """
        if any(grid is None for grid in [self.x_grid, self.y_grid, self.z_grid]):
            raise ValueError("Grid coordinates not set")
        
        # Simple nearest-neighbor interpolation (can be upgraded to trilinear)
        ix = np.argmin(np.abs(self.x_grid - x))
        iy = np.argmin(np.abs(self.y_grid - y))
        iz = np.argmin(np.abs(self.z_grid - z))
        
        E = np.array([
            self.E_x[ix, iy, iz] if self.E_x is not None else 0.0,
            self.E_y[ix, iy, iz] if self.E_y is not None else 0.0,
            self.E_z[ix, iy, iz] if self.E_z is not None else 0.0
        ])
        
        B = np.array([
            self.B_x[ix, iy, iz] if self.B_x is not None else 0.0,
            self.B_y[ix, iy, iz] if self.B_y is not None else 0.0,
            self.B_z[ix, iy, iz] if self.B_z is not None else 0.0
        ])
        
        return E, B


class LorentzSolver:
    """
    Solves the Lorentz force equation for charged particle trajectories.
    
    The equation of motion is:
        d𝐫/dt = 𝐯
        d𝐯/dt = (q/m)(𝐄 + 𝐯 × 𝐁)
    
    where:
        𝐫 = position vector
        𝐯 = velocity vector
        q = particle charge
        m = particle mass
        𝐄 = electric field
        𝐁 = magnetic field
    
    Uses scipy.integrate.solve_ivp with adaptive step size for accuracy.
    Supports multiple integration methods ('RK45', 'DOP853', 'BDF').
    
    Attributes:
        method: Integration method name
        rtol, atol: Relative and absolute tolerances
        max_step: Maximum integration step size [s]
    """
    
    def __init__(self, method: str = 'RK45', 
                 rtol: float = 1e-6, 
                 atol: float = 1e-9,
                 max_step: Optional[float] = None):
        """
        Initialize the Lorentz force solver.
        
        Args:
            method: ODE solver method ('RK45', 'DOP853', 'BDF', etc.)
            rtol: Relative tolerance for adaptive stepping
            atol: Absolute tolerance for adaptive stepping
            max_step: Maximum step size (None for automatic)
        """
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self._field_func: Optional[Callable] = None
    
    def _lorentz_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side of the Lorentz force ODE.
        
        State vector: [x, y, z, vx, vy, vz]
        
        The Lorentz force equation:
            𝐅 = q(𝐄 + 𝐯 × 𝐁)
            d𝐯/dt = 𝐅/m = (q/m)(𝐄 + 𝐯 × 𝐁)
        
        Args:
            t: Current time [s]
            state: State vector [x, y, z, vx, vy, vz]
            
        Returns:
            Time derivative of state [vx, vy, vz, ax, ay, az]
        """
        # Extract position and velocity
        pos = state[:3]  # [x, y, z]
        vel = state[3:]  # [vx, vy, vz]
        
        # Get fields at current position
        E, B = self._field_func(t, pos)
        
        # Calculate Lorentz force: F = q(E + v × B)
        v_cross_B = np.cross(vel, B)
        acceleration = (self._charge / self._mass) * (E + v_cross_B)
        
        # Return derivatives: [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
        return np.concatenate([vel, acceleration])
    
    def set_field_function(self, field_func: Callable[[float, np.ndarray], 
                                                       Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Set the function that provides E and B fields.
        
        Args:
            field_func: Function with signature func(t, position) -> (E, B)
                       where E and B are 3D vectors
        """
        self._field_func = field_func
    
    def solve(self, particle: ParticleState, 
              t_span: Tuple[float, float],
              t_eval: Optional[np.ndarray] = None,
              events: Optional[List[Callable]] = None) -> Dict:
        """
        Solve the particle trajectory.
        
        Args:
            particle: Initial particle state
            t_span: Time interval (t_start, t_end) [s]
            t_eval: Optional array of times for output evaluation
            events: Optional list of event functions for termination
            
        Returns:
            Dictionary containing:
                - 'success': Boolean indicating successful integration
                - 'trajectory': ParticleState with full trajectory
                - 'time': Time array
                - 'message': Solver status message
        """
        if self._field_func is None:
            raise ValueError("Field function not set. Call set_field_function() first.")
        
        # Store particle properties for use in ODE function
        self._charge = particle.charge
        self._mass = particle.mass
        
        # Initial state vector: [x, y, z, vx, vy, vz]
        y0 = np.concatenate([particle.position, particle.velocity])
        
        # Solve ODE
        solution = solve_ivp(
            fun=self._lorentz_equations,
            t_span=t_span,
            y0=y0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step if self.max_step is not None else np.inf,
            events=events
        )
        
        # Extract results
        if solution.success:
            positions = solution.y[:3, :].T  # Shape: (n_points, 3)
            velocities = solution.y[3:, :].T  # Shape: (n_points, 3)
            
            trajectory = ParticleState(
                position=positions,
                velocity=velocities,
                mass=particle.mass,
                charge=particle.charge,
                label=f"{particle.label}_trajectory"
            )
        else:
            trajectory = None
            warnings.warn(f"Integration failed: {solution.message}")
        
        return {
            'success': solution.success,
            'trajectory': trajectory,
            'time': solution.t,
            'message': solution.message,
            'nfev': solution.nfev,
            'events': solution.t_events
        }
    
    def solve_multiple(self, particles: List[ParticleState],
                      t_span: Tuple[float, float],
                      t_eval: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Solve trajectories for multiple particles.
        
        Particles are solved independently (no particle-particle interactions).
        For collective effects, see PlasmaRing class.
        
        Args:
            particles: List of initial particle states
            t_span: Time interval (t_start, t_end)
            t_eval: Optional evaluation times
            
        Returns:
            List of result dictionaries for each particle
        """
        results = []
        for particle in particles:
            result = self.solve(particle, t_span, t_eval)
            results.append(result)
        return results


@dataclass
class FRCParameters:
    """
    Parameters defining the Field-Reversed Configuration plasma.
    
    The FRC is characterized by:
    1. Closed poloidal field lines created by plasma currents
    2. A separatrix dividing closed and open field regions
    3. High plasma beta (β ≈ 1) in the core
    
    Rigid-Rotor Model:
        Assumes plasma rotates as a rigid body with angular frequency ω.
        This creates a diamagnetic current that reverses the external field.
        
        Current density: j_θ = n e ω r
        Pressure profile: p(r) = p₀ exp(-r²/2σ²)
    
    Attributes:
        separatrix_radius: Radius of the separatrix [m]
        length: FRC axial length [m]
        plasma_density: Peak electron density [m⁻³]
        temperature: Plasma temperature [eV]
        rotation_frequency: Angular rotation frequency [rad/s]
        ion_species: Ion type ('proton', 'deuteron', 'triton', etc.)
    """
    separatrix_radius: float = 0.2      # [m]
    length: float = 1.0                 # [m]
    plasma_density: float = 1e20        # [m⁻³]
    temperature: float = 1000           # [eV]
    rotation_frequency: float = 1e6     # [rad/s]
    ion_species: str = "proton"
    
    @property
    def ion_mass(self) -> float:
        """Get ion mass based on species."""
        masses = {
            'proton': PROTON_MASS,
            'deuteron': 2.014 * PROTON_MASS,
            'triton': 3.016 * PROTON_MASS,
            'electron': ELECTRON_MASS
        }
        return masses.get(self.ion_species, PROTON_MASS)
    
    def plasma_beta(self, B_external: float = 0.5) -> float:
        """
        Calculate plasma beta: β = p_plasma / p_magnetic
        
        Plasma beta measures the ratio of thermal pressure to magnetic pressure.
        FRCs typically operate at β ≈ 1 (high-beta configuration).
        
        β = (2 μ₀ n k_B T) / B²
        
        Args:
            B_external: External magnetic field at separatrix [T]. Default 0.5 T.
            
        Returns:
            Plasma beta (dimensionless)
        """
        from scipy.constants import k
        T_Kelvin = self.temperature * 11604.5  # Convert eV to K
        p_thermal = 2 * self.plasma_density * k * T_Kelvin  # Factor of 2 for ions + electrons
        p_magnetic = B_external**2 / (2 * MU_0)
        
        return p_thermal / p_magnetic if p_magnetic > 0 else np.inf


class PlasmaRing:
    """
    Represents the FRC plasma ring with internal magnetic field structure.
    
    The FRC is a compact toroid with:
    1. Poloidal magnetic field generated by azimuthal plasma currents
    2. Toroidal field typically negligible (unlike tokamaks)
    3. Field reversal at the separatrix boundary
    
    Magnetic Field Model (Rigid-Rotor):
        The plasma current creates an internal field that opposes the external
        mirror field, creating a null point and closed field lines.
        
        B_internal(r) = B₀ * (1 - (r/R_s)²) for r < R_s
        where R_s is the separatrix radius.
    
    Attributes:
        params: FRCParameters object
        vessel: Reference to VacuumVessel for grid access
        _internal_field_cache: Cached internal field calculations
    """
    
    def __init__(self, params: FRCParameters):
        """
        Initialize the FRC plasma ring.
        
        Args:
            params: FRCParameters with plasma characteristics
        """
        self.params = params
        self._internal_field_cache: Dict = {}
    
    def calculate_internal_field(self, r: np.ndarray, z: np.ndarray,
                                 B_external: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the internal magnetic field from plasma currents.
        
        Uses the rigid-rotor model where plasma rotation creates a diamagnetic
        current that generates the reversed field.
        
        The total field is: B_total = B_external + B_internal
        At the separatrix: B_total = 0 (field null)
        
        Model:
            B_z_internal(r) = -B_external * (1 - 2*(r/R_s)²) for r < R_s
            B_r_internal ≈ 0 (in thin-ring approximation)
        
        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            B_external: External field magnitude at separatrix [T]
            
        Returns:
            Tuple of (B_r, B_theta, B_z) internal field components [T]
        """
        R_s = self.params.separatrix_radius
        
        # Initialize field arrays
        B_r = np.zeros_like(r)
        B_theta = np.zeros_like(r)
        
        # Internal field model: parabolic profile that reverses external field
        # Inside separatrix (r < R_s): field is reversed
        # Outside separatrix: field decays rapidly
        
        inside_separatrix = r < R_s
        
        # Parabolic profile for B_z
        # B_z = B_ext * (2*(r/R_s)² - 1) gives:
        #   -B_ext at r=0 (full reversal)
        #   0 at r=R_s (separatrix)
        #   +B_ext at r>R_s (matches external)
        
        B_z = np.zeros_like(r)
        B_z[inside_separatrix] = B_external * (2 * (r[inside_separatrix] / R_s)**2 - 1)
        
        # Smooth transition outside separatrix (exponential decay)
        outside_separatrix = ~inside_separatrix
        if np.any(outside_separatrix):
            decay_length = 0.1 * R_s  # Characteristic decay scale
            delta_r = r[outside_separatrix] - R_s
            B_z[outside_separatrix] = B_external * np.exp(-delta_r / decay_length)
        
        # Axial variation (simple Gaussian envelope)
        L = self.params.length
        z_envelope = np.exp(-(z / (L/2))**2)
        B_z *= z_envelope
        
        return B_r, B_theta, B_z
    
    def get_separatrix_mask(self, vessel) -> np.ndarray:
        """
        Create a boolean mask identifying points inside the separatrix.
        
        The separatrix is the boundary between closed field lines (inside)
        and open field lines (outside).
        
        Args:
            vessel: VacuumVessel object
            
        Returns:
            Boolean array: True inside separatrix, False outside
        """
        R_s = self.params.separatrix_radius
        L = self.params.length
        
        # Ellipsoidal separatrix surface
        # (r/R_s)² + (z/(L/2))² = 1
        ellipticity = (vessel.R_grid / R_s)**2 + (vessel.Z_grid / (L/2))**2
        
        return ellipticity <= 1.0
    
    def calculate_pressure_profile(self, vessel) -> np.ndarray:
        """
        Calculate plasma pressure profile using rigid-rotor model.
        
        Pressure profile: p(r) = p₀ exp(-r²/2σ²)
        where σ ≈ R_s/2 for typical FRC
        
        From pressure balance: p + B²/(2μ₀) = constant
        
        Args:
            vessel: VacuumVessel object
            
        Returns:
            Pressure array [Pa]
        """
        from scipy.constants import k
        
        R_s = self.params.separatrix_radius
        sigma = R_s / 2  # Pressure scale length
        
        # Peak pressure from temperature and density
        T_Kelvin = self.params.temperature * 11604.5  # eV to K
        p0 = 2 * self.params.plasma_density * k * T_Kelvin
        
        # Exponential pressure profile
        pressure = p0 * np.exp(-(vessel.R_grid**2) / (2 * sigma**2))
        
        # Apply separatrix cutoff
        sep_mask = self.get_separatrix_mask(vessel)
        pressure = np.where(sep_mask, pressure, 0.0)
        
        return pressure
    
    def calculate_current_density(self, vessel, B_external: float) -> np.ndarray:
        """
        Calculate the azimuthal current density j_θ from ∇ × B = μ₀j.
        
        In cylindrical coordinates with axisymmetry:
            j_θ = (1/μ₀) * (∂B_r/∂z - ∂B_z/∂r)
        
        For the rigid-rotor model, this simplifies to:
            j_θ = n e ω r
        
        Args:
            vessel: VacuumVessel object
            B_external: External field [T]
            
        Returns:
            Azimuthal current density j_θ [A/m²]
        """
        # Rigid-rotor current: j_θ = n e ω r
        j_theta = (self.params.plasma_density * ELEMENTARY_CHARGE * 
                   self.params.rotation_frequency * vessel.R_grid)
        
        # Apply separatrix mask
        sep_mask = self.get_separatrix_mask(vessel)
        j_theta = np.where(sep_mask, j_theta, 0.0)
        
        return j_theta
    
    def get_total_field(self, vessel, coil_system, 
                       t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the total magnetic field including external coils and plasma.
        
        Superposition: B_total = B_coils + B_plasma_internal
        
        Args:
            vessel: VacuumVessel object
            coil_system: CoilSystem with external coils
            t: Time [s]
            
        Returns:
            Tuple of (B_r, B_theta, B_z) total field [T]
        """
        # Get external field from coils
        B_ext_r, B_ext_theta, B_ext_z = coil_system.calculate_total_field(vessel, t)
        
        # Estimate external field magnitude at separatrix
        # (simplified: use average on-axis field)
        B_ext_mag = np.mean(np.sqrt(B_ext_r**2 + B_ext_z**2))
        
        # Get internal plasma field
        B_int_r, B_int_theta, B_int_z = self.calculate_internal_field(
            vessel.R_grid, vessel.Z_grid, B_ext_mag
        )
        
        # Superpose fields
        B_r_total = B_ext_r + B_int_r
        B_theta_total = B_ext_theta + B_int_theta
        B_z_total = B_ext_z + B_int_z
        
        return B_r_total, B_theta_total, B_z_total


# Utility functions
def create_frc_particles(n_particles: int = 100,
                         frc_params: Optional[FRCParameters] = None,
                         species: str = 'proton') -> List[ParticleState]:
    """
    Create a population of test particles distributed in the FRC.
    
    Particles are initialized with:
    - Positions: Uniformly distributed within separatrix
    - Velocities: Maxwellian distribution at plasma temperature
    
    Args:
        n_particles: Number of particles to create
        frc_params: FRC parameters (uses defaults if None)
        species: Particle species ('proton', 'electron', etc.)
        
    Returns:
        List of ParticleState objects
    """
    if frc_params is None:
        frc_params = FRCParameters()
    
    particles = []
    R_s = frc_params.separatrix_radius
    L = frc_params.length
    
    # Thermal velocity: v_th = sqrt(kT/m)
    from scipy.constants import k
    T_Kelvin = frc_params.temperature * 11604.5
    
    if species == 'proton':
        mass = PROTON_MASS
        charge = ELEMENTARY_CHARGE
    elif species == 'electron':
        mass = ELECTRON_MASS
        charge = -ELEMENTARY_CHARGE
    else:
        mass = frc_params.ion_mass
        charge = ELEMENTARY_CHARGE
    
    v_th = np.sqrt(k * T_Kelvin / mass)
    
    for i in range(n_particles):
        # Random position within separatrix (ellipsoidal)
        # Use rejection sampling
        while True:
            r = np.random.uniform(0, R_s)
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.uniform(-L/2, L/2)
            
            # Check if inside separatrix
            if (r/R_s)**2 + (z/(L/2))**2 <= 1.0:
                break
        
        # Convert to Cartesian
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Maxwellian velocity distribution
        velocity = np.random.normal(0, v_th, 3)
        
        particle = ParticleState(
            position=np.array([x, y, z]),
            velocity=velocity,
            mass=mass,
            charge=charge,
            label=f"{species}_{i}"
        )
        particles.append(particle)
    
    return particles


# Example usage and testing
if __name__ == "__main__":
    print("FRC Physics Engine - Module Test")
    print("=" * 50)
    
    # Test particle creation
    particles = create_frc_particles(n_particles=10, species='proton')
    print(f"Created {len(particles)} test particles")
    print(f"Sample particle energy: {particles[0].kinetic_energy:.2e} J")
    
    # Test FRC parameters
    frc_params = FRCParameters(
        separatrix_radius=0.2,
        length=1.0,
        plasma_density=1e20,
        temperature=1000  # eV
    )
    
    B_test = 0.5  # Tesla
    beta = frc_params.plasma_beta(B_test)
    print(f"Plasma beta at B={B_test} T: {beta:.3f}")
    
    # Test PlasmaRing
    plasma = PlasmaRing(frc_params)
    print(f"FRC separatrix radius: {frc_params.separatrix_radius} m")
    print(f"FRC length: {frc_params.length} m")
    
    # Test Lorentz solver with simple uniform field
    def uniform_field(t, pos):
        E = np.array([0.0, 0.0, 0.0])
        B = np.array([0.0, 0.0, 1.0])  # 1 T in z-direction
        return E, B
    
    solver = LorentzSolver(method='RK45', rtol=1e-8, atol=1e-10)
    solver.set_field_function(uniform_field)
    
    # Create test particle (proton)
    test_particle = ParticleState(
        position=np.array([0.1, 0.0, 0.0]),  # 10 cm from axis
        velocity=np.array([0.0, 1e5, 0.0]),  # 100 km/s in y-direction
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
        label="test_proton"
    )
    
    # Solve for 1 microsecond
    t_span = (0.0, 1e-6)
    t_eval = np.linspace(0, 1e-6, 100)
    
    result = solver.solve(test_particle, t_span, t_eval=t_eval)
    
    if result['success']:
        traj = result['trajectory']
        final_pos = traj.position[-1]
        print(f"\nTest particle trajectory:")
        print(f"  Initial position: {test_particle.position} m")
        print(f"  Final position: {final_pos} m")
        print(f"  Gyroradius (theoretical): {test_particle.gyroradius(1.0):.4f} m")
        print(f"  Energy conserved: {np.allclose(traj.kinetic_energy, traj.kinetic_energy[0], rtol=1e-6)}")
    else:
        print(f"Solver failed: {result['message']}")
