import os
import matplotlib.pyplot as plt

def read_rmsd(input_filepath, frame_indices=None):
    """
    Reads frame index and RMSD from a file.
    
    Args:
        input_filepath (str): Path to RMSD file
        frame_indices (array-like, optional): Specific frame indices to include
    
    Returns:
        tuple: Lists of time points and corresponding RMSD values for filtered frames
    """
    frames = []
    rmsds = []
    with open(input_filepath, 'r') as file:
        for line in file:
            parts = line.split(':')
            frame_num = int(parts[0].split()[1])  # Extract frame number
            time = (frame_num * 20) / 1000.0      # Convert to ns
            rmsd = float(parts[1].split('=')[1])
            
            # Only include filtered frames if frame_indices is provided
            if frame_indices is None or frame_num in frame_indices:
                frames.append(time)
                rmsds.append(rmsd)
    return frames, rmsds

def plot_rmsd_data(ax, data, colors, labels, x_min, x_max, x_increment, y_min, y_max, y_increment):
    """
    Plots RMSD data on the provided axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axis to plot on
        data (list): List of (frames, rmsds) tuples
        colors (list): List of colors for each plot
        labels (list): List of labels for each plot
        x_min, x_max (float): X-axis limits
        x_increment (float): X-axis tick increment
        y_min, y_max (float): Y-axis limits
        y_increment (float): Y-axis tick increment
    """
    for (frames, rmsds), color, label in zip(data, colors[:len(data)], labels):
        ax.plot(frames, rmsds, label=label, color=color, alpha=0.7, linewidth=1.0)

    extended_x_max = x_max + (x_increment / 2)
    extended_y_max = y_max + (y_increment / 2)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (Å)')
    ax.legend()
    ax.tick_params(direction='in')
    ax.set_xlim(x_min, extended_x_max)
    ax.set_xticks([x_min + i * x_increment for i in range(int((x_max - x_min) / x_increment) + 1)])
    ax.set_ylim(y_min, extended_y_max)
    ax.set_yticks([y_min + i * y_increment for i in range(int((y_max - y_min) / y_increment) + 1)])

def plot_coordinate_transformation_rmsd(folder_path, filename, frame_indices=None):
    """
    Main function to plot RMSD data for reactants and products.
    
    Args:
        folder_path (str): Path to folder containing RMSD files
        filename (str): Output filename for the plot
        frame_indices (array-like, optional): Specific frame indices to include in plot
    """
    # Define input files
    files_reactant = [
        f'{folder_path}/txt/reactant_donor_red_transdRMSD.txt',
        f'{folder_path}/txt/reactant_donor_ox_transdRMSD.txt',
        f'{folder_path}/txt/reactant_acceptor_ox_transdRMSD.txt',
        f'{folder_path}/txt/reactant_acceptor_red_transdRMSD.txt',
    ]

    files_product = [
        f'{folder_path}/txt/product_donor_red_transdRMSD.txt',
        f'{folder_path}/txt/product_donor_ox_transdRMSD.txt',
        f'{folder_path}/txt/product_acceptor_ox_transdRMSD.txt',
        f'{folder_path}/txt/product_acceptor_red_transdRMSD.txt',
    ]

    # Plot settings
    x_min, x_max = 0, 280
    x_increment = 50
    y_min, y_max = 0, 1.50
    y_increment = 0.30

    colors = ['red', 'blue', 'cyan', 'orange']
    labels = ['Donor Red', 'Donor Ox', 'Acceptor Ox', 'Acceptor Red']

    # Read the RMSD data with frame filtering
    data_reactant = [read_rmsd(file, frame_indices) for file in files_reactant]
    data_product = [read_rmsd(file, frame_indices) for file in files_product]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # Plot reactant data
    plot_rmsd_data(axs[0], data_reactant, colors, labels, 
                  x_min, x_max, x_increment, y_min, y_max, y_increment)

    # Plot product data
    plot_rmsd_data(axs[1], data_product, colors, labels,
                  x_min, x_max, x_increment, y_min, y_max, y_increment)

    plt.tight_layout()
    
    # Save the figure
    file_path = os.path.join(f"{folder_path}/png", filename)
    plt.savefig(file_path, dpi=600)
    print(f' **Figure saved to: {file_path}')

# Example usage:
if __name__ == "__main__":
    # Example with frame filtering
    frame_indices = [i for i in range(0, 100, 2)]  # Example: plot every second frame up to 100
    plot_coordinate_transformation_rmsd('path_to_folder', 'output_filename.png', frame_indices)
    
    # Example without frame filtering (plots all frames)
    plot_coordinate_transformation_rmsd('path_to_folder', 'output_filename.png')
