"""
dashboard.py - Telemetry & Control Interface for FRC Simulation

This module provides real-time visualization and monitoring of the FRC simulation.
It displays:
1. Axial magnetic field strength maps
2. Core plasma density and temperature profiles
3. Net electrical output (P_recovered - P_input)

The dashboard is designed to read state from the physics engine without
modifying it directly (read-only telemetry pattern).

Uses matplotlib for 2D plots and optional plotly for interactive visualization.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

# Plotly remains optional; importing it is deferred until an interactive
# dashboard is requested so matplotlib-only use does not require Plotly.
PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


@dataclass
class TelemetryData:
    """
    Container for simulation telemetry data.
    
    This class aggregates all measurable quantities from the simulation
    for display on the dashboard.
    
    Attributes:
        time: Current simulation time [s]
        B_field: Magnetic field components (B_r, B_theta, B_z) [T]
        E_field: Electric field components (E_r, E_theta, E_z) [V/m]
        plasma_density: Plasma density profile [m⁻³]
        plasma_temperature: Plasma temperature profile [eV]
        pressure_profile: Plasma pressure profile [Pa]
        current_density: Current density profile [A/m²]
        coil_currents: Array of coil currents [A]
        input_power: Total input power to coils [W]
        recovered_power: Power recovered from DEC system [W]
        particle_count: Number of confined particles
        energy_balance: Dictionary with energy components
    """
    time: float = 0.0
    
    # Field data (3D arrays)
    B_field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    E_field: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    
    # Plasma profiles (3D arrays)
    plasma_density: Optional[np.ndarray] = None
    plasma_temperature: Optional[np.ndarray] = None
    pressure_profile: Optional[np.ndarray] = None
    current_density: Optional[np.ndarray] = None
    
    # Coil data
    coil_currents: List[float] = field(default_factory=list)
    
    # Power measurements
    input_power: float = 0.0
    recovered_power: float = 0.0
    
    # Particle data
    particle_count: int = 0
    confined_particle_count: int = 0
    
    # Energy tracking
    energy_balance: Dict[str, float] = field(default_factory=dict)
    
    @property
    def net_power(self) -> float:
        """Calculate net electrical output: P_net = P_recovered - P_input"""
        return self.recovered_power - self.input_power
    
    @property
    def gain_factor(self) -> float:
        """Calculate power gain: Q = P_recovered / P_input"""
        if self.input_power > 0:
            return self.recovered_power / self.input_power
        return 0.0
    
    def get_field_magnitude(self) -> Optional[np.ndarray]:
        """Calculate |B| from field components."""
        if self.B_field is None:
            return None
        B_r, B_theta, B_z = self.B_field
        return np.sqrt(B_r**2 + B_theta**2 + B_z**2)
    
    def get_field_components_along_axis(
        self,
        axis: str = 'z',
        r_coords: Optional[np.ndarray] = None,
        z_coords: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract field components along a specific axis.
        
        Args:
            axis: 'z' for an on-axis axial profile, 'r' for a midplane
                radial profile
            r_coords: Optional radial coordinate array for an 'r' profile
            z_coords: Optional axial coordinate array for a 'z' profile
            
        Returns:
            Dictionary with coordinate array and field components
        """
        if self.B_field is None:
            return {}
        
        B_r, _, B_z = (np.asarray(component) for component in self.B_field)
        if B_r.shape != B_z.shape:
            raise ValueError("B_r and B_z must have matching shapes")
        if axis not in {'r', 'z'}:
            raise ValueError("axis must be 'r' or 'z'")

        def coordinates(values: Optional[np.ndarray], size: int) -> np.ndarray:
            if values is None:
                return np.arange(size)
            values = np.asarray(values)
            if values.ndim != 1 or values.size != size:
                raise ValueError("Coordinate array length must match the field profile")
            return values

        if B_z.ndim == 3:
            # Field arrays use the simulation grid ordering (theta, r, z).
            if axis == 'z':
                profile_r, profile_z = B_r[0, 0, :], B_z[0, 0, :]
                return {'z': coordinates(z_coords, B_z.shape[2]), 'B_z': profile_z, 'B_r': profile_r}

            midplane = B_z.shape[2] // 2
            profile_r, profile_z = B_r[0, :, midplane], B_z[0, :, midplane]
            return {'r': coordinates(r_coords, B_z.shape[1]), 'B_z': profile_z, 'B_r': profile_r}

        if B_z.ndim == 2:
            if axis == 'z':
                profile_r, profile_z = B_r[0, :], B_z[0, :]
                return {'z': coordinates(z_coords, B_z.shape[1]), 'B_z': profile_z, 'B_r': profile_r}

            midplane = B_z.shape[1] // 2
            profile_r, profile_z = B_r[:, midplane], B_z[:, midplane]
            return {'r': coordinates(r_coords, B_z.shape[0]), 'B_z': profile_z, 'B_r': profile_r}

        if B_z.ndim == 1:
            return {
                axis: coordinates(z_coords if axis == 'z' else r_coords, B_z.size),
                'B_z': B_z,
                'B_r': B_r,
            }

        raise ValueError("Field components must be one-, two-, or three-dimensional")


class TelemetryDashboard:
    """
    Real-time visualization dashboard for FRC simulation.
    
    Provides live-updating plots for:
    1. Magnetic field topology and strength
    2. Plasma density and temperature profiles
    3. Power balance and energy metrics
    4. Particle confinement statistics
    
    The dashboard operates in read-only mode, observing simulation state
    without modifying it.
    
    Attributes:
        fig: Main figure object
        axes: Dictionary of subplot axes
        telemetry_history: List of TelemetryData snapshots
        update_interval: Time between updates [ms]
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 10),
                 use_plotly: bool = False):
        """
        Initialize the telemetry dashboard.
        
        Args:
            figsize: Figure size (width, height) in inches
            use_plotly: If True and available, use plotly for interactive plots
        """
        self.figsize = figsize
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        self.telemetry_history: List[TelemetryData] = []
        self.update_interval = 100  # ms
        
        # Initialize figure and axes
        self._setup_figure()
        
        # Storage for plot objects
        self.plot_objects: Dict[str, Any] = {}
    
    def _setup_figure(self) -> None:
        """Create the figure layout with subplots."""
        if self.use_plotly:
            self._setup_plotly_figure()
        else:
            self._setup_matplotlib_figure()
    
    def _setup_matplotlib_figure(self) -> None:
        """Set up matplotlib figure with subplots."""
        self.fig = plt.figure(figsize=self.figsize)
        
        # Create grid layout: 2x2 with some merging
        # Top-left: B-field map
        # Top-right: Plasma profiles
        # Bottom-left: Power balance
        # Bottom-right: Particle confinement
        
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        self.axes = {
            'bfield_map': self.fig.add_subplot(gs[0, 0]),
            'plasma_profiles': self.fig.add_subplot(gs[0, 1]),
            'power_balance': self.fig.add_subplot(gs[1, 0]),
            'confinement': self.fig.add_subplot(gs[1, 1])
        }
        
        # Set titles
        self.axes['bfield_map'].set_title('Axial Magnetic Field Strength |B| [T]')
        self.axes['plasma_profiles'].set_title('Plasma Density & Temperature Profiles')
        self.axes['power_balance'].set_title('Power Balance (Input vs Recovered)')
        self.axes['confinement'].set_title('Particle Confinement Statistics')
    
    def _setup_plotly_figure(self) -> None:
        """Set up plotly interactive figure."""
        if not PLOTLY_AVAILABLE:
            raise RuntimeError("Plotly is required for an interactive dashboard")
        from plotly.subplots import make_subplots

        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Axial Magnetic Field Strength |B| [T]',
                'Plasma Density & Temperature Profiles',
                'Power Balance (Input vs Recovered)',
                'Particle Confinement Statistics'
            )
        )
    
    def update_bfield_map(self, telemetry: TelemetryData, 
                         vessel_coords: Optional[Dict] = None) -> None:
        """
        Update the magnetic field strength map.
        
        Displays a 2D color map of |B| in the r-z plane.
        Overlays field lines if available.
        
        Args:
            telemetry: Current telemetry data
            vessel_coords: Dictionary with 'R_grid', 'Z_grid' arrays
        """
        ax = self.axes['bfield_map']
        ax.clear()
        
        B_mag = telemetry.get_field_magnitude()
        
        if B_mag is None:
            ax.text(0.5, 0.5, 'No field data', transform=ax.transAxes, ha='center')
            return
        
        if vessel_coords is not None:
            R = vessel_coords.get('R_grid', None)
            Z = vessel_coords.get('Z_grid', None)
            
            if R is not None and Z is not None:
                # Take mid-theta slice for visualization
                theta_idx = 0
                R_slice = R[theta_idx, :, :]
                Z_slice = Z[theta_idx, :, :]
                B_slice = B_mag[theta_idx, :, :]
                
                # Create contour plot
                contour = ax.contourf(R_slice, Z_slice, B_slice, 
                                     levels=50, cmap='viridis')
                plt.colorbar(contour, ax=ax, label='|B| [T]')
                
                # Overlay field contours only when their levels occur in the data.
                finite_values = B_slice[np.isfinite(B_slice)]
                contour_levels = [
                    level for level in (0.1, 0.5, 1.0)
                    if finite_values.size and finite_values.min() <= level <= finite_values.max()
                ]
                if contour_levels:
                    ax.contour(R_slice, Z_slice, B_slice,
                              levels=contour_levels, colors='white',
                              linewidths=0.5, linestyles='dashed')
        
        ax.set_xlabel('Radius r [m]')
        ax.set_ylabel('Axial position z [m]')
        ax.set_aspect('equal')
    
    def update_plasma_profiles(self, telemetry: TelemetryData,
                              vessel_coords: Optional[Dict] = None) -> None:
        """
        Update plasma density and temperature profiles.
        
        Shows radial profiles at z=0 (midplane).
        
        Args:
            telemetry: Current telemetry data
            vessel_coords: Grid coordinates
        """
        ax = self.axes['plasma_profiles']
        ax.clear()
        
        # Get radial coordinate
        if vessel_coords is not None:
            r = vessel_coords.get('r_grid', np.linspace(0, 1, 50))
        else:
            r = np.linspace(0, 1, 50)
        r = np.asarray(r)
        if r.ndim != 1 or r.size == 0:
            raise ValueError("r_grid must be a non-empty one-dimensional array")

        def radial_profile(values: np.ndarray) -> np.ndarray:
            """Extract a theta=0, z=0 radial profile from grid data."""
            values = np.asarray(values)
            if values.ndim == 3:
                return values[0, :, values.shape[2] // 2]
            if values.ndim == 2:
                return values[:, values.shape[1] // 2]
            if values.ndim == 1:
                return values
            raise ValueError("Plasma profiles must be one-, two-, or three-dimensional")

        def profile_coordinates(profile_size: int) -> np.ndarray:
            if r.size == profile_size:
                return r
            return np.linspace(r[0], r[-1], profile_size)
        
        # Plot density profile
        if telemetry.plasma_density is not None:
            density_profile = radial_profile(telemetry.plasma_density)
            
            ax.semilogy(profile_coordinates(len(density_profile)), density_profile,
                       'b-', label='Density n [m⁻³]', linewidth=2)
        
        # Plot temperature profile
        if telemetry.plasma_temperature is not None:
            temp_profile = radial_profile(telemetry.plasma_temperature)
            
            ax.plot(profile_coordinates(len(temp_profile)), temp_profile,
                   'r-', label='Temperature T [eV]', linewidth=2)
        
        ax.set_xlabel('Radius r [m]')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Core Plasma Profiles (z=0)')
    
    def update_power_balance(self, telemetry: TelemetryData) -> None:
        """
        Update power balance display.
        
        Shows time history of input power, recovered power, and net output.
        
        Args:
            telemetry: Current telemetry data
        """
        ax = self.axes['power_balance']
        ax.clear()
        
        # Extract time series from history
        times = [t.time for t in self.telemetry_history]
        p_input = [t.input_power for t in self.telemetry_history]
        p_recovered = [t.recovered_power for t in self.telemetry_history]
        p_net = [t.net_power for t in self.telemetry_history]
        
        if len(times) > 0:
            ax.plot(times, p_input, 'r-', label='P_input', linewidth=2)
            ax.plot(times, p_recovered, 'g-', label='P_recovered', linewidth=2)
            ax.plot(times, p_net, 'b--', label='P_net', linewidth=2)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Power [W]')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Log scaling is valid only when every plotted value is positive.
            all_powers = p_input + p_recovered + p_net
            if all(value > 0 for value in all_powers) and max(all_powers) > 10 * min(all_powers):
                ax.set_yscale('log')
        
        # Display current values as text
        current_text = (
            f"Current Values:\n"
            f"P_input: {telemetry.input_power:.2e} W\n"
            f"P_recovered: {telemetry.recovered_power:.2e} W\n"
            f"P_net: {telemetry.net_power:.2e} W\n"
            f"Gain Q: {telemetry.gain_factor:.2f}"
        )
        ax.text(0.02, 0.98, current_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update_confinement_stats(self, telemetry: TelemetryData) -> None:
        """
        Update particle confinement statistics.
        
        Shows confined fraction and energy distribution.
        
        Args:
            telemetry: Current telemetry data
        """
        ax = self.axes['confinement']
        ax.clear()
        
        # Calculate confinement fraction
        if telemetry.particle_count > 0:
            confined_fraction = telemetry.confined_particle_count / telemetry.particle_count
        else:
            confined_fraction = 0.0
        
        # Create bar chart
        categories = ['Total', 'Confined', 'Lost']
        counts = [
            telemetry.particle_count,
            telemetry.confined_particle_count,
            telemetry.particle_count - telemetry.confined_particle_count
        ]
        
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = ax.bar(categories, counts, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count:,}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
        
        ax.set_ylabel('Particle Count')
        ax.set_title(f'Confinement: {confined_fraction:.1%}')
        
        # Display energy balance if available
        if telemetry.energy_balance:
            energy_text = "Energy Components:\n"
            for key, value in telemetry.energy_balance.items():
                energy_text += f"{key}: {value:.2e} J\n"
            
            ax.text(0.02, 0.02, energy_text.rstrip(), transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
    
    def update(self, telemetry: TelemetryData, 
              vessel_coords: Optional[Dict] = None) -> None:
        """
        Update all dashboard panels with new telemetry data.
        
        Args:
            telemetry: Current telemetry snapshot
            vessel_coords: Optional grid coordinates for spatial plots
        """
        # Add to history
        self.telemetry_history.append(telemetry)
        
        # Limit history length for performance
        if len(self.telemetry_history) > 1000:
            self.telemetry_history = self.telemetry_history[-1000:]
        
        # Update each panel
        self.update_bfield_map(telemetry, vessel_coords)
        self.update_plasma_profiles(telemetry, vessel_coords)
        self.update_power_balance(telemetry)
        self.update_confinement_stats(telemetry)
        
        # Add main title with current time
        if self.use_plotly:
            self.fig.update_layout(title_text=f"FRC Simulation Dashboard - t = {telemetry.time:.6f} s")
        else:
            self.fig.suptitle(f"FRC Simulation Dashboard - t = {telemetry.time:.6f} s", 
                            fontsize=14, fontweight='bold')
    
    def show(self) -> None:
        """Display the dashboard."""
        if self.use_plotly:
            self.fig.show()
        else:
            plt.tight_layout()
            plt.show()
    
    def create_animation(self, telemetry_sequence: List[TelemetryData],
                        vessel_coords: Optional[Dict] = None,
                        interval: int = 100) -> Optional[FuncAnimation]:
        """
        Create an animation from a sequence of telemetry data.
        
        Args:
            telemetry_sequence: List of TelemetryData snapshots
            vessel_coords: Grid coordinates
            interval: Time between frames [ms]
            
        Returns:
            FuncAnimation object if using matplotlib, None otherwise
        """
        if self.use_plotly:
            warnings.warn("Animation not yet implemented for plotly backend.")
            return None
        
        def animate(frame: int) -> None:
            self.update(telemetry_sequence[frame], vessel_coords)
        
        anim = FuncAnimation(
            self.fig, animate,
            frames=len(telemetry_sequence),
            interval=interval,
            blit=False,
            repeat=True
        )
        
        return anim


class ControlPanel:
    """
    Interactive control panel for adjusting simulation parameters.
    
    Provides sliders and input fields for:
    - Coil currents
    - Plasma parameters
    - Diagnostic settings
    
    This is a placeholder for future interactive control implementation.
    In Google Colab, this would use ipywidgets.
    """
    
    def __init__(self):
        """Initialize the control panel."""
        self.parameters: Dict[str, Any] = {
            'coil_current': 1e6,  # A
            'plasma_density': 1e20,  # m⁻³
            'plasma_temperature': 1000,  # eV
            'time_step': 1e-9,  # s
        }
        
        self.callbacks: Dict[str, callable] = {}
    
    def add_parameter(self, name: str, value: Any, 
                     min_val: float = None, max_val: float = None,
                     callback: callable = None) -> None:
        """
        Add a controllable parameter.
        
        Args:
            name: Parameter name
            value: Initial value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            callback: Function to call when parameter changes
        """
        self.parameters[name] = value
        
        if callback is not None:
            self.callbacks[name] = callback
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Update a parameter value.
        
        Args:
            name: Parameter name
            value: New value
        """
        if name in self.parameters:
            old_value = self.parameters[name]
            self.parameters[name] = value
            
            # Call callback if registered
            if name in self.callbacks:
                self.callbacks[name](old_value, value)
        else:
            raise KeyError(f"Parameter '{name}' not found")
    
    def get_parameter(self, name: str) -> Any:
        """Get current value of a parameter."""
        return self.parameters.get(name)
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameter values."""
        return self.parameters.copy()


# Example usage and testing
if __name__ == "__main__":
    print("FRC Telemetry Dashboard - Module Test")
    print("=" * 50)
    
    # Create sample telemetry data
    telemetry = TelemetryData(
        time=0.0,
        B_field=(
            np.random.randn(16, 50, 100) * 0.1,  # B_r
            np.zeros((16, 50, 100)),             # B_theta
            np.random.randn(16, 50, 100) * 0.5   # B_z
        ),
        plasma_density=np.exp(-np.linspace(0, 3, 50)**2) * 1e20,
        plasma_temperature=np.ones(50) * 1000,
        input_power=5e6,      # 5 MW input
        recovered_power=2e6,  # 2 MW recovered
        particle_count=10000,
        confined_particle_count=8500,
        energy_balance={
            'magnetic': 1e5,
            'thermal': 5e5,
            'kinetic': 2e4
        }
    )
    
    print(f"Net Power: {telemetry.net_power:.2e} W")
    print(f"Gain Factor: {telemetry.gain_factor:.2f}")
    print(f"Confinement: {telemetry.confined_particle_count/telemetry.particle_count:.1%}")
    
    # Create dashboard
    dashboard = TelemetryDashboard(use_plotly=False)
    
    # Create mock vessel coordinates
    r = np.linspace(0, 0.5, 50)
    z = np.linspace(-1, 1, 100)
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    vessel_coords = {
        'R_grid': R[np.newaxis, :, :],
        'Z_grid': Z[np.newaxis, :, :],
        'r_grid': r
    }
    
    # Update dashboard
    dashboard.update(telemetry, vessel_coords)
    
    print("\nDashboard created successfully!")
    print("Call dashboard.show() to display (requires GUI backend)")
    
    # Test control panel
    control = ControlPanel()
    control.add_parameter('test_param', 1.0, 0.0, 10.0)
    control.set_parameter('test_param', 5.0)
    print(f"\nControl panel test: {control.get_parameter('test_param')}")
