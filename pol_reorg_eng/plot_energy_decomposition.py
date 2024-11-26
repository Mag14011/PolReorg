import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Dict, List
import os

def plot_energy_decomposition(
    # [Previous parameters remain the same]
    reactant_coulombic: float,
    reactant_coulombic_var: float,
    reactant_donor_polarization: float,
    reactant_donor_polarization_var: float,
    reactant_acceptor_polarization: float,
    reactant_acceptor_polarization_var: float,
    reactant_total: float,
    reactant_total_var: float,
    product_coulombic: float,
    product_coulombic_var: float,
    product_donor_polarization: float,
    product_donor_polarization_var: float,
    product_acceptor_polarization: float,
    product_acceptor_polarization_var: float,
    product_total: float,
    product_total_var: float,
    reorg_unpolarized: float,
    reorg_polarized: float,
    output_path: str,
    field_magnitude: Optional[float] = None
) -> str:
    """
    Create energy decomposition plot showing all components with error bars.
    
    Args:
        # Reactant state energies
        reactant_coulombic: Coulombic energy of reactant state
        reactant_coulombic_var: Variance in reactant Coulombic energy
        reactant_donor_polarization: Polarization energy of donor in reactant state
        reactant_donor_polarization_var: Variance in reactant donor polarization
        reactant_acceptor_polarization: Polarization energy of acceptor in reactant state
        reactant_acceptor_polarization_var: Variance in reactant acceptor polarization
        reactant_total: Total energy of reactant state (Coulombic + both polarizations)
        reactant_total_var: Variance in reactant total energy
        
        # Product state energies (same structure as reactant)
        [product state parameters]
        
        # Reorganization energies
        reorg_unpolarized: Reorganization energy without polarization
        reorg_polarized: Reorganization energy including polarization
        
        # Output
        output_path: Directory to save plot
        field_magnitude: Optional field magnitude for title
        
    Returns:
        Path to saved plot file
    """

    # Print validation data [Previous validation prints remain the same]
    print("\nValidating Energy Decomposition Data:")
    print("\nReactant State:")
    print(f"  Coulombic:             {reactant_coulombic:.3f} ± {np.sqrt(reactant_coulombic_var):.3f} eV")
    print(f"  Donor Polarization:    {reactant_donor_polarization:.3f} ± {np.sqrt(reactant_donor_polarization_var):.3f} eV")
    print(f"  Acceptor Polarization: {reactant_acceptor_polarization:.3f} ± {np.sqrt(reactant_acceptor_polarization_var):.3f} eV")
    print(f"  Total Energy:          {reactant_total:.3f} ± {np.sqrt(reactant_total_var):.3f} eV")

    print("\nProduct State:")
    print(f"  Coulombic:             {product_coulombic:.3f} ± {np.sqrt(product_coulombic_var):.3f} eV")
    print(f"  Donor Polarization:    {product_donor_polarization:.3f} ± {np.sqrt(product_donor_polarization_var):.3f} eV")
    print(f"  Acceptor Polarization: {product_acceptor_polarization:.3f} ± {np.sqrt(product_acceptor_polarization_var):.3f} eV")
    print(f"  Total Energy:          {product_total:.3f} ± {np.sqrt(product_total_var):.3f} eV")

    print("\nReorganization Energies:")
    print(f"  Unpolarized: {reorg_unpolarized:.3f} eV")
    print(f"  Polarized:   {reorg_polarized:.3f} eV")

    # Create plot
    width_inches = 6.6  # Double-column width
    height_inches = width_inches / 1.618  # Golden ratio
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # Define x positions for each group - Note reordered positions for donor/acceptor
    reactant_positions = np.array([0, 1, 2, 3])  # Coulombic, Donor-Pol, Acc-Pol, Total
    product_positions = np.array([5, 6, 7, 8])   # Coulombic, Acc-Pol, Donor-Pol, Total
    reorg_positions = np.array([10, 11])         # Unpolarized, Polarized

    all_positions = np.concatenate([reactant_positions, product_positions, reorg_positions])
    width = 0.8

    # Define colors
    GRAY = '#D3D3D3'    # Coulombic and unpolarized reorg
    RED = '#FF6B6B'     # Donor polarization
    BLUE = '#4A90E2'    # Acceptor polarization
    PURPLE = '#9370DB'  # Total and polarized reorg

    # Prepare data arrays - Note reordered data for product state
    energy_data = [
        # Reactant state
        reactant_coulombic,
        reactant_donor_polarization,
        reactant_acceptor_polarization,
        reactant_total,
        # Product state (note reordered polarization energies)
        product_coulombic,
        product_acceptor_polarization,  # Swapped order
        product_donor_polarization,     # Swapped order
        product_total,
        # Reorganization
        reorg_unpolarized,
        reorg_polarized
    ]

    # Prepare error bars - Note reordered errors for product state
    error_bars = [
        # Reactant state
        np.sqrt(reactant_coulombic_var),
        np.sqrt(reactant_donor_polarization_var),
        np.sqrt(reactant_acceptor_polarization_var),
        np.sqrt(reactant_total_var),
        # Product state (note reordered variances)
        np.sqrt(product_coulombic_var),
        np.sqrt(product_acceptor_polarization_var),  # Swapped order
        np.sqrt(product_donor_polarization_var),     # Swapped order
        np.sqrt(product_total_var),
        # Reorganization (no errors)
        0, 0
    ]

    # Color scheme - Note reordered colors for product state
    colors = [
        # Reactant state
        GRAY,    # Coulombic
        RED,     # Donor Pol
        BLUE,    # Acceptor Pol
        PURPLE,  # Total
        # Product state
        GRAY,    # Coulombic
        BLUE,    # Acceptor Pol (swapped)
        RED,     # Donor Pol (swapped)
        PURPLE,  # Total
        # Reorganization
        GRAY,    # Unpolarized (matches Coulombic)
        PURPLE   # Polarized (matches Total)
    ]

    # Create bars
    bars = ax.bar(all_positions, energy_data, width,
                 color=colors, edgecolor='black', linewidth=1,
                 yerr=error_bars, capsize=3)

    # Customize plot
    ax.set_ylabel('Energy (eV)', fontsize=10)

    # Set x-axis labels - Note reordered labels for product state
    labels = [
        'Coul', 'Don-Pol', 'Acc-Pol', 'Total',  # Reactant
        'Coul', 'Acc-Pol', 'Don-Pol', 'Total',  # Product (note reordered)
        'Unpol', 'Pol'                          # Reorganization
    ]
    ax.set_xticks(all_positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, edgecolor='black', label='Coulombic/Unpol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor='black', label='Donor Pol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor='black', label='Acceptor Pol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=PURPLE, edgecolor='black', label='Total/Pol')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='best')

    # Add group labels
    y_max = ax.get_ylim()[1]
    ax.text(np.mean(reactant_positions), y_max, 'Reactant State',
            ha='center', va='bottom')
    ax.text(np.mean(product_positions), y_max, 'Product State',
            ha='center', va='bottom')
    ax.text(np.mean(reorg_positions), y_max, 'Reorg',
            ha='center', va='bottom')

    # Add group separators
    ax.axvline(x=4, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=9, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=8, direction='in')
    ax.tick_params(axis='both', which='minor', direction='in')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Keep all spines visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # Add field magnitude to title if provided
    if field_magnitude is not None:
        ax.set_title(f'Field Magnitude: {field_magnitude:.2f} V/Å',
                    fontsize=10, pad=10)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_path,
        f"energy_decomposition{'_' + str(field_magnitude) if field_magnitude else ''}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return output_file

def generate_energy_decomposition_animation(
    results: List[Dict],
    output_path: str,
    fps: int = 30
) -> str:
    """
    Generate an animation showing how energy decomposition changes with field magnitude.
    """
    # Set up figure with more height to accommodate labels
    width_inches = 6.6
    height_inches = width_inches / 1.3  # Adjusted ratio for more height
    fig, ax = plt.subplots(figsize=(width_inches, height_inches))

    # Define colors and positions (same as before)
    GRAY = '#D3D3D3'
    RED = '#FF6B6B'
    BLUE = '#4A90E2'
    PURPLE = '#9370DB'

    reactant_positions = np.array([0, 1, 2, 3])
    product_positions = np.array([5, 6, 7, 8])
    reorg_positions = np.array([10, 11])
    all_positions = np.concatenate([reactant_positions, product_positions, reorg_positions])
    width = 0.8

    # Initialize bars with zeros
    colors = [
        GRAY, RED, BLUE, PURPLE,
        GRAY, BLUE, RED, PURPLE,
        GRAY, PURPLE
    ]
    bars = ax.bar(all_positions, np.zeros(len(all_positions)), width,
                 color=colors, edgecolor='black', linewidth=1)

    # Convert bars to a list for easier concatenation
    bars_list = list(bars)

    # Rest of the setup code remains the same
    ax.set_ylabel('Energy (eV)', fontsize=10)
    labels = [
        'Coul', 'Don-Pol', 'Acc-Pol', 'Total',
        'Coul', 'Acc-Pol', 'Don-Pol', 'Total',
        'Unpol', 'Pol'
    ]
    ax.set_xticks(all_positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, edgecolor='black', label='Coulombic/Unpol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor='black', label='Donor Pol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, edgecolor='black', label='Acceptor Pol'),
        plt.Rectangle((0, 0), 1, 1, facecolor=PURPLE, edgecolor='black', label='Total/Pol')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

    # Add separators
    ax.axvline(x=4, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=9, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Calculate initial y-limits from first frame
    first_frame = results[0]
    initial_values = [
        first_frame['reactant_coulombic'],
        first_frame['reactant_donor_polarization'],
        first_frame['reactant_acceptor_polarization'],
        first_frame['reactant_total'],
        first_frame['product_coulombic'],
        first_frame['product_acceptor_polarization'],
        first_frame['product_donor_polarization'],
        first_frame['product_total'],
        first_frame['reorg_unpolarized'],
        first_frame['reorg_polarized']
    ]
    max_val = max(abs(min(initial_values)), abs(max(initial_values)))
    y_margin = max_val * 0.2
    ax.set_ylim(-max_val - y_margin, max_val + y_margin)

    def update(frame):
        data = results[frame]

        # Prepare energy values
        energy_values = [
            data['reactant_coulombic'],
            data['reactant_donor_polarization'],
            data['reactant_acceptor_polarization'],
            data['reactant_total'],
            data['product_coulombic'],
            data['product_acceptor_polarization'],
            data['product_donor_polarization'],
            data['product_total'],
            data['reorg_unpolarized'],
            data['reorg_polarized']
        ]

        # Prepare error values
        error_values = [
            np.sqrt(data['reactant_coulombic_var']),
            np.sqrt(data['reactant_donor_polarization_var']),
            np.sqrt(data['reactant_acceptor_polarization_var']),
            np.sqrt(data['reactant_total_var']),
            np.sqrt(data['product_coulombic_var']),
            np.sqrt(data['product_acceptor_polarization_var']),
            np.sqrt(data['product_donor_polarization_var']),
            np.sqrt(data['product_total_var']),
            0, 0  # No errors for reorg energies
        ]

        # Update bars
        for bar, val in zip(bars, energy_values):
            bar.set_height(val)

        # Remove old error bars
        for artist in ax.lines[:]:
            artist.remove()

        # Add new error bars
        cap_width = 0.05
        error_lines = []
        for x, val, err in zip(all_positions, energy_values, error_values):
            if err > 0:
                # Vertical line
                line = ax.vlines(x, val - err, val + err, color='black')
                # Horizontal caps
                cap1 = ax.hlines(val - err, x - cap_width, x + cap_width, color='black')
                cap2 = ax.hlines(val + err, x - cap_width, x + cap_width, color='black')
                error_lines.extend([line, cap1, cap2])

        # Update title
        field_magnitude = data.get('donor_field', None)
        if field_magnitude is not None:
            ax.set_title(f'Field Magnitude: {field_magnitude:.2f} V/Å',
                        fontsize=10, pad=10)

        # Return the combined list of artists that need updating
        return bars_list  # Only return bars since error bars are redrawn each frame

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(results),
                        interval=1000/fps, blit=True)

    # Adjust layout before saving
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Make room for both labels and title

    # Save animation
    output_file = os.path.join(output_path, "energy_decomposition.mp4")
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close()

    return output_file
