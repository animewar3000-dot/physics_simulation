"""
main.py - Main Simulation Runner for FRC Nuclear Reactor Simulation

This script connects the grid, coils, plasma, and solver into an active
simulation loop. It demonstrates the full workflow of setting up and
running an FRC simulation with telemetry monitoring.

Usage:
    python main.py
    
Or in Google Colab:
    from main import run_simulation
    run_simulation()
"""

import numpy as np
import matplotlib.pyplot as plt

# Import from Phase 1 modules
from environment import VacuumVessel, GridConfig, CoilSystem, CoilParameters, MagneticCoil
from physics import LorentzSolver, FRCParameters, PlasmaRing, create_frc_particles, ParticleState
from dashboard import TelemetryDashboard, TelemetryData


def run_simulation(
    num_particles: int = 100,
    sample_particles: int = 5,
    t_end: float = 1e-6,
    num_timepoints: int = 100,
    show_dashboard: bool = True
) -> tuple:
    """
    Execute a complete FRC simulation run.
    
    This function orchestrates the entire simulation workflow:
    1. Initialize vacuum vessel and magnetic grid
    2. Setup external coil system (mirror/pinch configuration)
    3. Initialize FRC plasma equilibrium
    4. Generate particle ensemble
    5. Run particle trajectory simulation
    6. Collect telemetry data
    7. Display results on dashboard
    
    Args:
        num_particles: Total number of particles to simulate
        sample_particles: Number of particles to track individually
        t_end: Simulation end time [s]
        num_timepoints: Number of time points for output
        show_dashboard: Whether to display the telemetry dashboard
        
    Returns:
        Tuple of (vessel, coils, plasma, solver, trajectories, dashboard)
        
    Raises:
        ValueError: If parameters are out of valid range
    """
    
    # =========================================================================
    # Step 1/4: Initialize Vacuum Vessel & Magnetic Grid
    # =========================================================================
    print("[1/4] Initializing Vacuum Vessel & Magnetic Grid...")
    
    # Configure grid with cylindrical coordinates (r, θ, z)
    # r_max=0.5m, z_max=1.0m provides adequate domain for typical FRC
    grid_cfg = GridConfig(
        r_max=0.5,      # Maximum radial extent [m]
        z_max=1.0,      # Maximum axial extent [m]
        r_points=100,   # Radial resolution
        z_points=200,   # Axial resolution
        theta_points=32,  # Azimuthal resolution (for 3D if needed)
        use_cylindrical=True  # Use axisymmetric cylindrical coordinates
    )
    
    # Create vacuum vessel with computational grid
    vessel = VacuumVessel(grid_cfg)
    
    print(f"   - Grid dimensions: r=[{grid_cfg.r_min}, {grid_cfg.r_max}] m, "
          f"z=[{grid_cfg.z_min}, {grid_cfg.z_max}] m")
    print(f"   - Grid resolution: {grid_cfg.r_points} × {grid_cfg.theta_points} × {grid_cfg.z_points}")
    
    # =========================================================================
    # Step 2/4: Setup External Coil System (Mirror/Pinch Configuration)
    # =========================================================================
    print("\n[2/4] Initializing External Magnetic Coil System...")
    
    # Create coil system for generating external magnetic fields
    # Mirror coils create magnetic mirrors at ends to confine plasma axially
    coils = CoilSystem()
    
    # Add two mirror coils at z = ±0.5m with opposing currents for mirror effect
    # Current of 1 MA creates strong confining field (~several Tesla)
    coil_top = MagneticCoil(
        CoilParameters(
            radius=0.4,           # Coil radius [m]
            z_position=0.5,       # Axial position [m]
            current=1e6,          # Coil current [A]
            n_turns=10          # Number of turns
        )
    )
    
    coil_bottom = MagneticCoil(
        CoilParameters(
            radius=0.4,           # Coil radius [m]
            z_position=-0.5,      # Axial position [m]
            current=1e6,          # Coil current [A] (same direction for mirror)
            n_turns=10          # Number of turns
        )
    )
    
    coils.add_coil(coil_top)
    coils.add_coil(coil_bottom)
    
    print(f"   - Added {len(coils.coils)} mirror coils")
    print(f"   - Coil currents: {[c.params.current for c in coils.coils]} A")
    
    # Calculate external magnetic field on grid
    print("   - Computing external B-field distribution...")
    B_ext = coils.calculate_total_field(vessel)
    B_ext_magnitude = np.sqrt(B_ext[0]**2 + B_ext[1]**2 + B_ext[2]**2)
    print(f"   - External field range: {B_ext_magnitude.min():.4f} to {B_ext_magnitude.max():.4f} T")
    
    # =========================================================================
    # Step 3/4: Initialize FRC Plasma Equilibrium
    # =========================================================================
    print("\n[3/4] Initializing FRC Plasma Equilibrium...")
    
    # Define FRC plasma parameters using rigid-rotor model
    # Separatrix radius defines boundary between closed/open field lines
    # Temperature in eV (1000 eV ≈ 11.6 million K)
    frc_params = FRCParameters(
        separatrix_radius=0.2,    # FRC core radius [m]
        temperature=1000,         # Plasma temperature [eV]
        plasma_density=1e20,      # Plasma density [m⁻³]
        rotation_frequency=1e6,   # Ion rotation frequency [rad/s]
        ion_species="deuteron"    # Deuterium ions
    )
    
    # Create plasma ring with internal self-generated magnetic field
    # The FRC has reversed internal field that cancels external field at separatrix
    plasma = PlasmaRing(frc_params)
    
    print(f"   - Separatrix radius: {frc_params.separatrix_radius} m")
    print(f"   - Plasma temperature: {frc_params.temperature} eV")
    print(f"   - Plasma density: {frc_params.plasma_density:.2e} m⁻³")
    
    # Calculate plasma's internal magnetic field (reversed field)
    print("   - Computing internal plasma B-field (rigid-rotor model)...")
    # Get average external field at separatrix for the rigid-rotor model
    B_ext_avg = float(np.mean(B_ext_magnitude))
    B_plasma = plasma.calculate_internal_field(
        r=vessel.R_grid,
        z=vessel.Z_grid,
        B_external=B_ext_avg
    )
    if B_plasma is not None:
        B_plasma_mag = np.sqrt(B_plasma[0]**2 + B_plasma[1]**2 + B_plasma[2]**2)
        print(f"   - Internal field range: {B_plasma_mag.min():.4f} to {B_plasma_mag.max():.4f} T")
    
    # Calculate total magnetic field (external + internal)
    B_total = tuple(
        B_ext[i] + (B_plasma[i] if B_plasma is not None else 0) 
        for i in range(3)
    )
    B_total_mag = np.sqrt(B_total[0]**2 + B_total[1]**2 + B_total[2]**2)
    print(f"   - Total field range: {B_total_mag.min():.4f} to {B_total_mag.max():.4f} T")
    
    # =========================================================================
    # Step 4/4: Generate Particles & Configure Solver
    # =========================================================================
    print("\n[4/4] Generating Particles & Configuring Solver...")
    
    # Generate particle ensemble with Maxwellian velocity distribution
    # Particles initialized within separatrix radius with thermal velocities
    particles = create_frc_particles(
        n_particles=num_particles,
        frc_params=frc_params,
        species='deuteron'  # For reproducibility
    )
    
    print(f"   - Generated {len(particles)} particles")
    print(f"   - Particle species: Deuterium ions (mass={frc_params.ion_mass:.2e} kg)")
    
    # Initialize Lorentz force solver for particle trajectory integration
    # Uses scipy.integrate.solve_ivp with RK45 method
    solver = LorentzSolver(
        rtol=1e-8,  # Relative tolerance for adaptive stepping
        atol=1e-10  # Absolute tolerance
    )
    
    # Define simulation time span
    t_span = (0.0, t_end)  # Start and end time [s]
    t_eval = np.linspace(t_span[0], t_span[1], num_timepoints)  # Output times
    
    print(f"   - Time span: {t_span[0]:.2e} to {t_span[1]:.2e} s ({t_end*1e6:.2f} μs)")
    print(f"   - Output timepoints: {num_timepoints}")
    
    # =========================================================================
    # Execute Simulation: Particle Trajectory Integration
    # =========================================================================
    print("\nExecuting particle trajectory simulation...")
    
    # Track sample particles individually for visualization
    # Full simulation would process all particles (can be parallelized)
    trajectories = []
    confined_count = 0
    
    # Create field configuration for solver interpolation
    from physics import FieldConfiguration
    field_config = FieldConfiguration(
        B_x=B_total[0],
        B_y=B_total[1],
        B_z=B_total[2],
        E_x=np.zeros_like(B_total[0]),  # No external E-field initially
        E_y=np.zeros_like(B_total[1]),
        E_z=np.zeros_like(B_total[2]),
        x_grid=vessel.R_grid * np.cos(vessel.Theta_grid),
        y_grid=vessel.R_grid * np.sin(vessel.Theta_grid),
        z_grid=vessel.Z_grid
    )
    
    # Set up field function for the solver
    def get_fields(t: float, position: np.ndarray) -> tuple:
        """Get E and B fields at a given position."""
        return field_config.get_fields_at_point(position[0], position[1], position[2])
    
    solver.set_field_function(get_fields)
    
    # Integrate trajectories for sample particles
    for idx, particle in enumerate(particles[:sample_particles]):
        print(f"   - Tracking particle {idx+1}/{sample_particles}...")
        
        # Solve Lorentz force equation: F = q(E + v × B)
        traj = solver.solve(
            particle=particle,
            t_span=t_span,
            t_eval=t_eval
        )
        
        if traj is not None:
            trajectories.append(traj)
            
            # Check if particle remains confined (within separatrix)
            final_pos = traj.y[:, -1]  # Final position
            r_final = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
            if r_final < frc_params.separatrix_radius:
                confined_count += 1
    
    # Estimate confinement fraction for all particles
    # (Simplified - full simulation would track all particles)
    confinement_fraction = confined_count / min(sample_particles, len(particles))
    print(f"\n   - Confined particles: {confined_count}/{sample_particles} "
          f"({confinement_fraction*100:.1f}%)")
    
    print("\n✓ Simulation completed successfully!")
    
    # =========================================================================
    # Telemetry & Dashboard Display
    # =========================================================================
    dashboard = None
    
    if show_dashboard:
        print("\nGenerating telemetry dashboard...")
        
        # Aggregate simulation data into telemetry container
        telemetry = TelemetryData(
            time=t_end,
            B_field=B_total,
            E_field=(np.zeros_like(B_total[0]), 
                     np.zeros_like(B_total[1]), 
                     np.zeros_like(B_total[2])),
            plasma_density=plasma.get_density_profile(vessel),
            plasma_temperature=plasma.get_temperature_profile(vessel),
            pressure_profile=plasma.get_pressure_profile(vessel),
            current_density=plasma.get_current_density(vessel),
            coil_currents=[c.params.current for c in coils.coils],
            input_power=coils.calculate_input_power(),
            recovered_power=0.0,  # Will be populated by DEC module in Phase 2
            particle_count=len(particles),
            confined_particle_count=int(confined_count),
            energy_balance={
                'magnetic_energy': 0.5 * np.sum(B_total_mag**2) / MU_0,
                'thermal_energy': 1.5 * frc_params.density * frc_params.temperature * 1.6e-19,
                'kinetic_energy': sum(p.kinetic_energy for p in particles)
            }
        )
        
        # Create and update dashboard
        dashboard = TelemetryDashboard()
        dashboard.update(telemetry, vessel)
        
        # Display plots
        plt.tight_layout()
        plt.show()
        
        print(f"   - Net power: {telemetry.net_power:.2e} W")
        print(f"   - Gain factor: {telemetry.gain_factor:.2f}")
    
    return vessel, coils, plasma, solver, trajectories, dashboard


def run_parameter_scan():
    """
    Run a parameter scan to study confinement vs. coil current.
    
    This function demonstrates how to use the simulation framework
    for parametric studies and optimization.
    """
    print("=" * 60)
    print("FRC Parameter Scan: Confinement vs. Coil Current")
    print("=" * 60)
    
    currents = np.linspace(0.5e6, 2.0e6, 5)  # 0.5 to 2.0 MA
    confinement_results = []
    
    for I in currents:
        print(f"\nRunning simulation with I_coil = {I*1e-6:.1f} MA...")
        
        # Setup vessel and coils
        vessel = VacuumVessel(GridConfig(r_max=0.5, z_max=1.0))
        coils = CoilSystem()
        coils.add_coil(MagneticCoil(CoilParameters(radius=0.4, z_position=0.5, current=I)))
        coils.add_coil(MagneticCoil(CoilParameters(radius=0.4, z_position=-0.5, current=I)))
        
        # Setup plasma
        plasma = PlasmaRing(FRCParameters(separatrix_radius=0.2, temperature=1000))
        
        # Generate and track particles
        particles = create_frc_particles(num_particles=50, params=FRCParameters(separatrix_radius=0.2, temperature=1000))
        solver = LorentzSolver()
        
        # Get fields
        B_ext = coils.calculate_field_on_grid(vessel)
        B_plasma = plasma.calculate_internal_field(vessel)
        B_total = tuple(B_ext[i] + (B_plasma[i] if B_plasma is not None else 0) for i in range(3))
        
        from physics import FieldConfiguration
        field_config = FieldConfiguration(
            B_x=B_total[0], B_y=B_total[1], B_z=B_total[2],
            E_x=np.zeros_like(B_total[0]), E_y=np.zeros_like(B_total[1]), E_z=np.zeros_like(B_total[2]),
            x_grid=vessel.R_grid * np.cos(vessel.Theta_grid),
            y_grid=vessel.R_grid * np.sin(vessel.Theta_grid),
            z_grid=vessel.Z_grid
        )
        
        # Track particles
        confined = 0
        t_span = (0.0, 1e-6)
        t_eval = np.linspace(0, 1e-6, 50)
        
        for p in particles[:10]:
            traj = solver.solve(p, field_config, t_span, t_eval)
            if traj is not None:
                r_final = np.sqrt(traj.y[0, -1]**2 + traj.y[1, -1]**2)
                if r_final < 0.2:
                    confined += 1
        
        confinement_frac = confined / min(10, len(particles))
        confinement_results.append(confinement_frac)
        print(f"   Confinement fraction: {confinement_frac*100:.1f}%")
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(currents * 1e-6, confinement_results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Coil Current [MA]', fontsize=12)
    plt.ylabel('Confinement Fraction', fontsize=12)
    plt.title('FRC Confinement vs. Coil Current', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return currents, confinement_results


if __name__ == "__main__":
    # Run main simulation
    print("=" * 60)
    print("Field-Reversed Configuration (FRC) Nuclear Reactor Simulation")
    print("Phase 1: Grid, Coils, Plasma Equilibrium, and Particle Dynamics")
    print("=" * 60)
    print()
    
    vessel, coils, plasma, solver, trajectories, dashboard = run_simulation(
        num_particles=100,
        sample_particles=5,
        t_end=1e-6,
        num_timepoints=100,
        show_dashboard=True
    )
    
    # Optional: Run parameter scan
    # Uncomment to study confinement vs. coil current
    # currents, results = run_parameter_scan()
