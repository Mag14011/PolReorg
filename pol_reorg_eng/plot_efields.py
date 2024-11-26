import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def read_field_data(filename):
    """Read the frame-by-frame field data from file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the start of frame-by-frame data
    start_marker = "=== Frame-by-Frame Results ==="
    header_line = None
    data_start = None
    
    for i, line in enumerate(lines):
        if start_marker in line:
            header_line = i + 1
            data_start = i + 3  # Skip the header and delimiter lines
            break
    
    if data_start is None:
        raise ValueError("Could not find frame-by-frame data section in file")
    
    # Extract the data lines
    data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
    
    # Convert to DataFrame
    data = pd.DataFrame([
        line.split() for line in data_lines
    ], columns=['Frame', 'E_x_d', 'E_y_d', 'E_z_d', 'U_d', 
                'E_x_a', 'E_y_a', 'E_z_a', 'U_a', 
                'U_pol', 'U_Coul', 'U_total'])
    
    # Convert numeric columns
    numeric_columns = ['Frame', 'E_x_d', 'E_y_d', 'E_z_d', 'U_d',
                      'E_x_a', 'E_y_a', 'E_z_a', 'U_a',
                      'U_pol', 'U_Coul', 'U_total']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col])
    
    # Convert frame to time in ns
    data['Time'] = data['Frame'] * 20 / 1000
    
    return data

def calculate_magnitudes(data):
    """Calculate field magnitudes for donor and acceptor."""
    data['E_d_mag'] = np.sqrt(data['E_x_d']**2 + data['E_y_d']**2 + data['E_z_d']**2)
    data['E_a_mag'] = np.sqrt(data['E_x_a']**2 + data['E_y_a']**2 + data['E_z_a']**2)
    return data

def plot_unit_sphere(ax, x, y, z, color):
    """Plot normalized vectors on a unit sphere."""
    # Normalize vectors
    mag = np.sqrt(x**2 + y**2 + z**2)
    x_norm = x / mag
    y_norm = y / mag
    z_norm = z / mag
    
    # Plot points
    ax.scatter(x_norm, y_norm, z_norm, c=color, alpha=0.5, s=1)
    
    # Set up sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = 1 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 1 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot wireframe sphere
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1)
    
    # Set equal aspect ratio and limits
    ax.set_box_aspect([1,1,1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def set_inward_ticks(ax):
    """Helper function to set ticks inward for a given axis."""
    ax.tick_params(direction='in', which='both')  # both major and minor ticks
    ax.tick_params(axis='both', which='both', direction='in')

def create_field_analysis_plot(filename, output_filename="field_analysis.png"):
    """Create the complete figure with all panels."""
    # Set default tick direction for all plots
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # Read and process data
    try:
        data = read_field_data(filename)
        data = calculate_magnitudes(data)
    except Exception as e:
        print(f"Error reading data: {e}")
        raise
    
    # Create figure and grid
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 0.3], height_ratios=[1, 1],
                         hspace=0.2, wspace=0)
    
    # Create sphere subplots separately
    gs_sphere_d = gs[0, 0].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.3)
    gs_sphere_a = gs[1, 0].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.3)
    
    # Colors
    donor_color = 'red'
    acceptor_color = 'blue'
    
    # Calculate y-axis limits for time series and histograms
    max_mag = max(data['E_d_mag'].max(), data['E_a_mag'].max())
    y_lim = (0, np.ceil(max_mag * 10) / 10)
    
    # Donor panels
    # Unit sphere
    ax_sphere_d = fig.add_subplot(gs_sphere_d[0], projection='3d')
    plot_unit_sphere(ax_sphere_d, data['E_x_d'], data['E_y_d'], data['E_z_d'], donor_color)
    ax_sphere_d.set_title('Donor\nField Directions')
    
    # Time series
    ax_time_d = fig.add_subplot(gs_sphere_d[1])
    ax_time_d.plot(data['Time'], data['E_d_mag'], color=donor_color, linewidth=0.5)
    ax_time_d.set_ylabel('Magnitude (V/Å)')
    ax_time_d.set_title('Donor Field Magnitude')
    ax_time_d.set_ylim(y_lim)
    ax_time_d.set_xticklabels([])
    set_inward_ticks(ax_time_d)
    
    # Histogram
    ax_hist_d = fig.add_subplot(gs[0, 1])
    ax_hist_d.hist(data['E_d_mag'], bins=30, orientation='horizontal', color=donor_color)
    ax_hist_d.set_ylim(y_lim)
    ax_hist_d.set_xticklabels([])
    ax_hist_d.set_yticklabels([])
    set_inward_ticks(ax_hist_d)
    
    # Acceptor panels
    # Unit sphere
    ax_sphere_a = fig.add_subplot(gs_sphere_a[0], projection='3d')
    plot_unit_sphere(ax_sphere_a, data['E_x_a'], data['E_y_a'], data['E_z_a'], acceptor_color)
    ax_sphere_a.set_title('Acceptor\nField Directions')
    
    # Time series
    ax_time_a = fig.add_subplot(gs_sphere_a[1])
    ax_time_a.plot(data['Time'], data['E_a_mag'], color=acceptor_color, linewidth=0.5)
    ax_time_a.set_xlabel('Time (ns)')
    ax_time_a.set_ylabel('Magnitude (V/Å)')
    ax_time_a.set_title('Acceptor Field Magnitude')
    ax_time_a.set_ylim(y_lim)
    set_inward_ticks(ax_time_a)
    
    # Histogram
    ax_hist_a = fig.add_subplot(gs[1, 1])
    ax_hist_a.hist(data['E_a_mag'], bins=30, orientation='horizontal', color=acceptor_color)
    ax_hist_a.set_ylim(y_lim)
    ax_hist_a.set_xticklabels([])
    ax_hist_a.set_yticklabels([])
    set_inward_ticks(ax_hist_a)
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "field_analysis.png"
        create_field_analysis_plot(input_file, output_file)
    else:
        print("Usage: python script.py input_file [output_file]")
