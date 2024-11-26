import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def read_energy_data(filename):
    """
    Read and parse the energy analysis output file.
    
    Args:
        filename: Path to the formatted output file
    
    Returns:
        metadata: Dict containing analysis parameters
        data: DataFrame containing the numerical data
    """
    metadata = {}
    data_lines = []
    data_section_found = False

    with open(filename, 'r') as f:
        content = f.readlines()

    # Read through file line by line
    for i, line in enumerate(content):
        line = line.strip()
        
        # Extract field type from metadata
        if line.startswith('Field Type:'):
            metadata['Field Type'] = line.split(':')[1].strip()
            
        # Look for data section marker
        if '# Data:' in line:
            data_section_found = True
            # Skip the "# Data:" line and the column numbers line
            data_start = i + 2
            break

    if not data_section_found:
        raise ValueError("Could not find '# Data:' marker in file")

    # Read the data lines
    for line in content[data_start:]:
        if line.strip() and not line.startswith('#'):  # Skip empty lines and comments
            values = [x for x in line.split() if x]
            if len(values) == 28:  # Updated to expect 28 columns
                data_lines.append(values)

    if not data_lines:
        raise ValueError("No data found in file")

    # Convert to DataFrame with correct column indexing
    data = pd.DataFrame(data_lines, columns=[f"col_{i+1}" for i in range(28)])
    # Convert numeric columns
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='ignore')

    return metadata, data

def create_field_subplot(ax, data):
    """Create subplot showing field magnitudes vs filter value for MD fields."""
    markers = {
        'reactant_donor': {'marker': 'o', 'color': 'blue', 'filled': True, 'label': 'R.D'},
        'product_acceptor': {'marker': 'o', 'color': 'blue', 'filled': False, 'label': 'P.A'},
        'reactant_acceptor': {'marker': 's', 'color': 'red', 'filled': True, 'label': 'R.A'},
        'product_donor': {'marker': 's', 'color': 'red', 'filled': False, 'label': 'P.D'}
    }
    
    for field_type, props in markers.items():
        if field_type == 'reactant_donor':
            x, y, yerr = data['col_1'], data['col_1'], data['col_2']
        elif field_type == 'reactant_acceptor':
            x, y, yerr = data['col_1'], data['col_3'], data['col_4']
        elif field_type == 'product_donor':
            x, y, yerr = data['col_1'], data['col_5'], data['col_6']
        else:  # product_acceptor
            x, y, yerr = data['col_1'], data['col_7'], data['col_8']
            
        ax.errorbar(x, y, yerr=yerr,
                   marker=props['marker'],
                   color=props['color'],
                   markerfacecolor=props['color'] if props['filled'] else 'white',
                   markeredgecolor='black',
                   label=props['label'],
                   linestyle=':',
                   capsize=3)
    
    ax.set_xlabel('Filter (V/Å)')
    ax.set_ylabel('E-Field (V/Å)')
    ax.legend(ncol=2, fontsize='small')
    ax.tick_params(direction='in', which='both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def create_polarization_subplot(ax, data, is_synthetic):
    """Create subplot showing polarization energies vs respective fields."""
    markers = {
        'reactant_donor': {'marker': 'o', 'color': 'blue', 'filled': True, 'label': 'R.D'},
        'product_acceptor': {'marker': 'o', 'color': 'blue', 'filled': False, 'label': 'P.A'},
        'reactant_acceptor': {'marker': 's', 'color': 'red', 'filled': True, 'label': 'R.A'},
        'product_donor': {'marker': 's', 'color': 'red', 'filled': False, 'label': 'P.D'}
    }
    
    for field_type, props in markers.items():
        if field_type == 'reactant_donor':
            x, y, yerr = data['col_1'], data['col_13'], np.sqrt(data['col_14'])
        elif field_type == 'reactant_acceptor':
            x, y, yerr = data['col_3'], data['col_15'], np.sqrt(data['col_16'])
        elif field_type == 'product_donor':
            x, y, yerr = data['col_5'], data['col_21'], np.sqrt(data['col_22'])
        else:  # product_acceptor
            x, y, yerr = data['col_7'], data['col_23'], np.sqrt(data['col_24'])
            
        ax.errorbar(x, y, yerr=yerr,
                   marker=props['marker'],
                   color=props['color'],
                   markerfacecolor=props['color'] if props['filled'] else 'white',
                   markeredgecolor='black',
                   label=props['label'],
                   linestyle=':',
                   capsize=3)
    
    ax.set_xlabel('E-Field (V/Å)')
    ax.set_ylabel('Pol. Energy (eV)')
    ax.legend(ncol=2, fontsize='small')
    ax.tick_params(direction='in', which='both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def create_reorg_subplot(ax, data, is_synthetic):
    """Create subplot showing reorganization energies."""
    if is_synthetic:
        markers = {
            'reactant_donor': {'marker': 'o', 'filled': True, 'label': 'R.D'},
            'product_acceptor': {'marker': 'o', 'filled': False, 'label': 'P.A'},
            'reactant_acceptor': {'marker': 's', 'filled': True, 'label': 'R.A'},
            'product_donor': {'marker': 's', 'filled': False, 'label': 'P.D'}
        }
        
        for field_type, props in markers.items():
            if field_type == 'reactant_donor':
                x = data['col_1']
            elif field_type == 'reactant_acceptor':
                x = data['col_3']
            elif field_type == 'product_donor':
                x = data['col_5']
            else:  # product_acceptor
                x = data['col_7']
            
            # Unpolarized (blue) and Polarized (red)
            for reorg_type, color, col_idx in [('U', 'blue', 'col_27'), ('P', 'red', 'col_28')]:
                ax.plot(x, data[col_idx],
                       marker=props['marker'],
                       color=color,
                       markerfacecolor=color if props['filled'] else 'white',
                       markeredgecolor='black',
                       label=f'{reorg_type}-{props["label"]}',
                       linestyle=':')
        
        ax.set_xlabel('E-Field (V/Å)')
        ax.legend(ncol=2, fontsize='small')
    else:
        x = data['col_1']
        ax.plot(x, data['col_27'], 'o:', label='Unpol', color='blue', markeredgecolor='black')
        ax.plot(x, data['col_28'], 's:', label='Pol', color='red', markeredgecolor='black')
        ax.set_xlabel('Filter (V/Å)')
        ax.legend(fontsize='small')
    
    ax.set_ylabel('Reorg. Energy (eV)')
    ax.tick_params(direction='in', which='both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def plot_efield_dependent_reorganization_energy(input_file, output_dir):
    """Main plotting function."""
    metadata, data = read_energy_data(input_file)
    is_synthetic = metadata.get('Field Type', '').strip().lower() == 'synthetic'
    
    if not is_synthetic:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.5))
        plt.subplots_adjust(wspace=0.4)
        
        create_field_subplot(ax1, data)
        create_polarization_subplot(ax2, data, is_synthetic)
        create_reorg_subplot(ax3, data, is_synthetic)
        
        ax1.set_title('(A)')
        ax2.set_title('(B)')
        ax3.set_title('(C)')
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.5))
        plt.subplots_adjust(wspace=0.4)
        
        create_polarization_subplot(ax1, data, is_synthetic)
        create_reorg_subplot(ax2, data, is_synthetic)
        
        ax1.set_title('(A)')
        ax2.set_title('(B)')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 
                              "synthetic_field_analysis.png" if is_synthetic else "md_field_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python plot_efield_dependent_reorganization_energy.py input_file output_dir")
        sys.exit(1)
    
    output_file = plot_efield_dependent_reorganization_energy(sys.argv[1], sys.argv[2])
    print(f"Plot saved as: {output_file}")

