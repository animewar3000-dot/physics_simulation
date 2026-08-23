"""
environment.py - Spatial Grid & Hardware Architecture for FRC Simulation

This module defines the vacuum vessel geometry and magnetic coil systems
that generate the external magnetic fields for the Field-Reversed Configuration.

Physical Constants:
    μ₀ (mu_0): Permeability of free space = 4π × 10⁻⁷ T·m/A
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.special import ellipk, ellipe


# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space [T·m/A]


@dataclass
class GridConfig:
    """Configuration for the computational grid."""
    r_min: float = 0.0      # Minimum radial coordinate [m]
    r_max: float = 1.0      # Maximum radial coordinate [m]
    r_points: int = 100     # Number of radial grid points
    
    theta_min: float = 0.0  # Minimum azimuthal angle [rad]
    theta_max: float = 2 * np.pi  # Maximum azimuthal angle [rad]
    theta_points: int = 32  # Number of azimuthal grid points
    
    z_min: float = -1.0     # Minimum axial coordinate [m]
    z_max: float = 1.0      # Maximum axial coordinate [m]
    z_points: int = 200     # Number of axial grid points
    
    use_cylindrical: bool = True  # Use cylindrical coordinates if True, else Cartesian


class VacuumVessel:
    """
    Represents the vacuum vessel containing the FRC plasma.
    
    Establishes a cylindrical coordinate grid (r, θ, z) for axisymmetric simulations,
    with optional support for full 3D Cartesian grids.
    
    The grid is used to:
    1. Discretize the spatial domain for field calculations
    2. Store plasma properties (density, temperature, etc.)
    3. Compute magnetic and electric field distributions
    
    Attributes:
        config: GridConfig object containing grid parameters
        r_grid: Radial coordinate array [m]
        theta_grid: Azimuthal coordinate array [rad]
        z_grid: Axial coordinate array [m]
        R_grid: 3D array of radial coordinates at each grid point
        Theta_grid: 3D array of azimuthal coordinates at each grid point
        Z_grid: 3D array of axial coordinates at each grid point
        x_grid, y_grid, z_grid_cart: Cartesian coordinate grids (if needed)
    """
    
    def __init__(self, config: Optional[GridConfig] = None):
        """
        Initialize the vacuum vessel grid.
        
        Args:
            config: GridConfig object. If None, uses default values.
        """
        self.config = config if config is not None else GridConfig()
        self._build_grid()
    
    def _build_grid(self) -> None:
        """
        Build the computational grid based on configuration.
        
        For cylindrical coordinates, creates meshgrid for (r, θ, z).
        For Cartesian, converts cylindrical bounds to Cartesian domain.
        """
        # Create 1D coordinate arrays
        self.r_grid = np.linspace(
            self.config.r_min, 
            self.config.r_max, 
            self.config.r_points
        )
        self.theta_grid = np.linspace(
            self.config.theta_min,
            self.config.theta_max,
            self.config.theta_points,
            endpoint=False  # Exclude endpoint for periodic boundary
        )
        self.z_grid = np.linspace(
            self.config.z_min,
            self.config.z_max,
            self.config.z_points
        )
        
        if self.config.use_cylindrical:
            # Create 3D meshgrid for cylindrical coordinates
            # Shape: (theta_points, r_points, z_points)
            self.Theta_grid, self.R_grid, self.Z_grid = np.meshgrid(
                self.theta_grid, 
                self.r_grid, 
                self.z_grid,
                indexing='ij'
            )
            
            # Convert to Cartesian for visualization and some calculations
            self.x_grid = self.R_grid * np.cos(self.Theta_grid)
            self.y_grid = self.R_grid * np.sin(self.Theta_grid)
            self.z_grid_cart = self.Z_grid
        else:
            # Cartesian grid setup
            max_radius = self.config.r_max
            self.x_grid = np.linspace(-max_radius, max_radius, self.config.r_points)
            self.y_grid = np.linspace(-max_radius, max_radius, self.config.r_points)
            self.z_grid_cart = np.linspace(
                self.config.z_min,
                self.config.z_max,
                self.config.z_points
            )
            
            X, Y, Z = np.meshgrid(
                self.x_grid, 
                self.y_grid, 
                self.z_grid_cart,
                indexing='ij'
            )
            
            # Derive cylindrical coordinates from Cartesian
            self.R_grid = np.sqrt(X**2 + Y**2)
            self.Theta_grid = np.arctan2(Y, X)
            self.Z_grid = Z
    
    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the grid."""
        if self.config.use_cylindrical:
            return (
                len(self.theta_grid),
                len(self.r_grid),
                len(self.z_grid)
            )
        else:
            return (
                len(self.x_grid),
                len(self.y_grid),
                len(self.z_grid_cart)
            )
    
    def get_volume_elements(self) -> np.ndarray:
        """
        Calculate volume elements dV for integration.
        
        In cylindrical coordinates: dV = r dr dθ dz
        Returns array of volume elements at each grid point.
        """
        if self.config.use_cylindrical:
            dr = self.r_grid[1] - self.r_grid[0] if len(self.r_grid) > 1 else 1.0
            dtheta = self.theta_grid[1] - self.theta_grid[0] if len(self.theta_grid) > 1 else 1.0
            dz = self.z_grid[1] - self.z_grid[0] if len(self.z_grid) > 1 else 1.0
            
            # Volume element varies with radius in cylindrical coords
            dV = self.R_grid * dr * dtheta * dz
        else:
            dx = self.x_grid[1] - self.x_grid[0] if len(self.x_grid) > 1 else 1.0
            dy = self.y_grid[1] - self.y_grid[0] if len(self.y_grid) > 1 else 1.0
            dz = self.z_grid_cart[1] - self.z_grid_cart[0] if len(self.z_grid_cart) > 1 else 1.0
            
            dV = np.ones_like(self.R_grid) * dx * dy * dz
        
        return dV
    
    def mask_inside_vessel(self, data: np.ndarray) -> np.ndarray:
        """
        Apply vessel boundary mask to data array.
        
        Sets values outside the physical vessel to zero or NaN.
        Default implementation assumes cylindrical vessel with radius r_max.
        """
        mask = self.R_grid <= self.config.r_max
        return np.where(mask, data, np.nan)


@dataclass
class CoilParameters:
    """Parameters defining a magnetic coil."""
    radius: float           # Coil radius [m]
    z_position: float       # Axial position of coil center [m]
    current: float          # Current through coil [A] (can be time-dependent)
    n_turns: int = 1        # Number of turns
    coil_type: str = "circular"  # "circular" or "helical"


class MagneticCoil:
    """
    Represents a magnetic field coil using the Biot-Savart law.
    
    Calculates the magnetic field B at arbitrary points in space due to
    current-carrying coils. For circular loops, uses the analytical solution
    involving complete elliptic integrals.
    
    The magnetic field from a circular loop at position (r, z) in cylindrical
    coordinates is given by:
    
    B_r = (μ₀ I z / (2π r √((R+r)² + z²))) * [-K(k²) + ((R²+r²+z²)/((R-r)²+z²)) * E(k²)]
    B_z = (μ₀ I / (2π √((R+r)² + z²))) * [K(k²) + ((R²-r²-z²)/((R-r)²+z²)) * E(k²)]
    
    where:
        R = coil radius
        I = current
        k² = 4Rr / ((R+r)² + z²)
        K(k²) = complete elliptic integral of first kind
        E(k²) = complete elliptic integral of second kind
    
    Attributes:
        params: CoilParameters object containing coil specifications
        current_function: Optional callable for time-dependent current I(t)
    """
    
    def __init__(self, params: CoilParameters, 
                 current_function: Optional[callable] = None):
        """
        Initialize a magnetic coil.
        
        Args:
            params: CoilParameters with radius, position, current, etc.
            current_function: Optional function I(t) for time-varying current.
                            If None, uses constant current from params.
        """
        self.params = params
        self.current_function = current_function
    
    def get_current(self, t: float = 0.0) -> float:
        """
        Get the coil current at time t.
        
        Args:
            t: Time [s]
            
        Returns:
            Current value [A]
        """
        if self.current_function is not None:
            return self.current_function(t)
        return self.params.current
    
    def _calculate_field_on_axis(self, r: np.ndarray, z: np.ndarray, 
                                  t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate magnetic field using elliptic integral formulation.
        
        This is the analytical solution for a circular current loop.
        Handles the singularity at r=0 by using the on-axis formula.
        
        Args:
            r: Radial coordinates [m]
            z: Axial coordinates relative to coil center [m]
            t: Time for current evaluation [s]
            
        Returns:
            Tuple of (B_r, B_z) arrays [T]
        """
        I = self.get_current(t) * self.params.n_turns
        R = self.params.radius
        
        # Relative axial position from coil center
        z_rel = z - self.params.z_position
        
        # Precompute common terms
        R_plus_r = R + r
        R_minus_r = R - r
        denom_base = R_plus_r**2 + z_rel**2
        
        # Avoid division by zero at r=0, R=r, z=0
        epsilon = 1e-12
        denom_base = np.maximum(denom_base, epsilon)
        
        # Modulus squared for elliptic integrals
        k_squared = 4 * R * r / denom_base
        k_squared = np.clip(k_squared, 0, 1 - epsilon)  # Ensure k² < 1
        
        # Complete elliptic integrals K(k²) and E(k²)
        K = ellipk(k_squared)
        E = ellipe(k_squared)
        
        # Common factor
        sqrt_denom = np.sqrt(denom_base)
        factor = MU_0 * I / (2 * np.pi * sqrt_denom)
        
        # Handle r=0 separately (on-axis field)
        on_axis = r < epsilon
        
        # Radial component B_r
        # B_r = (μ₀ I z / (2π r √denom)) * [-K + ((R²+r²+z²)/denom_minus) * E]
        B_r = np.zeros_like(r)
        off_axis = ~on_axis
        if np.any(off_axis):
            denom_minus = R_minus_r**2 + z_rel**2
            denom_minus = np.maximum(denom_minus, epsilon)
            
            term1 = -K[off_axis]
            term2 = ((R**2 + r[off_axis]**2 + z_rel[off_axis]**2) / denom_minus[off_axis]) * E[off_axis]
            
            B_r[off_axis] = (MU_0 * I * z_rel[off_axis] / (2 * np.pi * r[off_axis] * sqrt_denom[off_axis])) * (term1 + term2)
        
        # Axial component B_z
        # B_z = (μ₀ I / (2π √denom)) * [K + ((R²-r²-z²)/denom_minus) * E]
        term1_z = K
        term2_z = ((R**2 - r**2 - z_rel**2) / np.maximum(R_minus_r**2 + z_rel**2, epsilon)) * E
        B_z = factor * (term1_z + term2_z)
        
        # On-axis limit: B_r = 0, B_z = μ₀ I R² / (2(R²+z²)^(3/2))
        if np.any(on_axis):
            B_z[on_axis] = MU_0 * I * R**2 / (2 * (R**2 + z_rel[on_axis]**2)**1.5)
        
        return B_r, B_z
    
    def calculate_field(self, vessel: VacuumVessel, 
                       t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the complete magnetic field vector on the grid.
        
        Computes B = (B_r, B_θ, B_z) at all grid points. For a single circular
        coil in an axisymmetric configuration, B_θ = 0.
        
        Args:
            vessel: VacuumVessel object containing the grid
            t: Time for current evaluation [s]
            
        Returns:
            Tuple of (B_r, B_theta, B_z) arrays [T]
        """
        if vessel.config.use_cylindrical:
            r = vessel.R_grid
            z = vessel.Z_grid
            
            B_r, B_z = self._calculate_field_on_axis(r, z, t)
            B_theta = np.zeros_like(B_r)  # Axisymmetric, no toroidal field
            
        else:
            # Cartesian grid - need to transform results
            r = vessel.R_grid
            z = vessel.Z_grid
            
            B_r_cyl, B_z_cyl = self._calculate_field_on_axis(r, z, t)
            
            # Transform to Cartesian components
            # B_x = B_r cos(θ) - B_θ sin(θ)
            # B_y = B_r sin(θ) + B_θ cos(θ)
            cos_theta = np.cos(vessel.Theta_grid)
            sin_theta = np.sin(vessel.Theta_grid)
            
            B_x = B_r_cyl * cos_theta
            B_y = B_r_cyl * sin_theta
            B_z = B_z_cyl
            
            # Return as cylindrical components for consistency
            B_r = B_r_cyl
            B_theta = np.zeros_like(B_r)
            B_z = B_z_cyl
        
        return B_r, B_theta, B_z
    
    def calculate_field_cartesian(self, vessel: VacuumVessel,
                                  t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate magnetic field in Cartesian components.
        
        Useful for visualization and coupling with Cartesian solvers.
        
        Args:
            vessel: VacuumVessel object
            t: Time [s]
            
        Returns:
            Tuple of (B_x, B_y, B_z) arrays [T]
        """
        B_r, B_theta, B_z = self.calculate_field(vessel, t)
        
        # Cylindrical to Cartesian transformation
        cos_theta = np.cos(vessel.Theta_grid)
        sin_theta = np.sin(vessel.Theta_grid)
        
        B_x = B_r * cos_theta - B_theta * sin_theta
        B_y = B_r * sin_theta + B_theta * cos_theta
        
        return B_x, B_y, B_z


class CoilSystem:
    """
    Manages multiple magnetic coils as a system.
    
    Allows superposition of fields from multiple coils (central mirror coils,
    field-reversal coils, etc.) to create complex magnetic configurations.
    
    Attributes:
        coils: List of MagneticCoil objects
    """
    
    def __init__(self):
        """Initialize an empty coil system."""
        self.coils: List[MagneticCoil] = []
    
    def add_coil(self, coil: MagneticCoil) -> None:
        """
        Add a coil to the system.
        
        Args:
            coil: MagneticCoil object to add
        """
        self.coils.append(coil)
    
    def remove_coil(self, index: int) -> None:
        """
        Remove a coil from the system.
        
        Args:
            index: Index of coil to remove
        """
        if 0 <= index < len(self.coils):
            self.coils.pop(index)
    
    def calculate_total_field(self, vessel: VacuumVessel,
                             t: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the total magnetic field from all coils.
        
        Uses superposition principle: B_total = Σ B_i
        
        Args:
            vessel: VacuumVessel object containing the grid
            t: Time for field calculation [s]
            
        Returns:
            Tuple of (B_r_total, B_theta_total, B_z_total) arrays [T]
        """
        B_r_total = np.zeros_like(vessel.R_grid)
        B_theta_total = np.zeros_like(vessel.R_grid)
        B_z_total = np.zeros_like(vessel.R_grid)
        
        for coil in self.coils:
            B_r, B_theta, B_z = coil.calculate_field(vessel, t)
            B_r_total += B_r
            B_theta_total += B_theta
            B_z_total += B_z
        
        return B_r_total, B_theta_total, B_z_total
    
    def calculate_field_magnitude(self, vessel: VacuumVessel,
                                  t: float = 0.0) -> np.ndarray:
        """
        Calculate the magnitude of the total magnetic field |B|.
        
        Args:
            vessel: VacuumVessel object
            t: Time [s]
            
        Returns:
            |B| = √(B_r² + B_θ² + B_z²) array [T]
        """
        B_r, B_theta, B_z = self.calculate_total_field(vessel, t)
        return np.sqrt(B_r**2 + B_theta**2 + B_z**2)


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test configuration
    print("FRC Simulation Environment - Module Test")
    print("=" * 50)
    
    # Create vacuum vessel
    config = GridConfig(
        r_max=0.5,
        r_points=50,
        z_max=1.0,
        z_points=100,
        theta_points=16
    )
    vessel = VacuumVessel(config)
    print(f"Grid shape: {vessel.get_grid_shape()}")
    
    # Create coil system for FRC
    coil_system = CoilSystem()
    
    # Add mirror coils (typical FRC configuration)
    mirror_params = CoilParameters(
        radius=0.4,      # 40 cm coil radius
        z_position=0.5,  # 50 cm from center
        current=1e6,     # 1 MA current
        n_turns=10
    )
    coil_system.add_coil(MagneticCoil(mirror_params))
    
    # Add opposite mirror coil
    mirror_params2 = CoilParameters(
        radius=0.4,
        z_position=-0.5,
        current=1e6,
        n_turns=10
    )
    coil_system.add_coil(MagneticCoil(mirror_params2))
    
    # Calculate field at t=0
    B_mag = coil_system.calculate_field_magnitude(vessel, t=0.0)
    
    print(f"Maximum |B|: {np.nanmax(B_mag):.4f} T")
    print(f"Minimum |B|: {np.nanmin(B_mag):.4f} T")
    print(f"Mean |B|: {np.nanmean(B_mag):.4f} T")
    
    # Find field null (where FRC forms)
    null_mask = B_mag < 0.01  # Less than 10 mT
    null_fraction = np.sum(null_mask) / null_mask.size
    print(f"Volume fraction with |B| < 10 mT: {null_fraction:.2%}")
