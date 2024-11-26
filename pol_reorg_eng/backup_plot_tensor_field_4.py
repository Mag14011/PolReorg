import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import os
from tqdm import tqdm
import read_xyz as rxyz

def get_fe_position(xyz_frame: Tuple[List[str], np.ndarray]) -> np.ndarray:
    """
    Extract Fe atom position from a frame of XYZ coordinates.
    
    Args:
        xyz_frame: Tuple of (atom_symbols, coordinates)
        
    Returns:
        numpy array of Fe coordinates [x, y, z]
    
    Raises:
        ValueError: If not exactly one Fe atom is found
    """
    symbols, coords = xyz_frame
    fe_indices = [i for i, symbol in enumerate(symbols) if symbol.lower() == 'fe']
    if len(fe_indices) != 1:
        raise ValueError(f"Found {len(fe_indices)} Fe atoms, expected exactly 1")
    return coords[fe_indices[0]]

def plot_ellipsoid(ax: Axes3D, 
                   tensor: np.ndarray, 
                   center: np.ndarray,
                   color: str,
                   alpha: float = 0.5,
                   scale: float = 3.0) -> None:
    """
    Plot polarizability tensor as an ellipsoid centered at Fe position.
    """
    # Apply scaling to the tensor itself, not just as a multiplication
    scaled_tensor = tensor * scale
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(scaled_tensor)
    
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Transform unit sphere to ellipsoid using the scaled eigenvalues directly
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i,j], y[i,j], z[i,j]])
            # Remove sqrt to allow for larger scaling
            transformed = np.dot(eigenvecs, point * eigenvals)
            x[i,j] = transformed[0] + center[0]
            y[i,j] = transformed[1] + center[1]
            z[i,j] = transformed[2] + center[2]
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_efield_vector(ax: Axes3D,
                      fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      center: np.ndarray,
                      color: str,
                      std: Optional[float] = None,
                      scale: float = 1.0) -> None:
    """
    Plot electric field vector with optional variance cone at Fe position.
    The variance cone size is determined solely by the standard deviation parameter.
    """
    Ex, Ey, Ez = fields
    # Scale the field components while preserving relative magnitudes
    Ex, Ey, Ez = Ex * scale, Ey * scale, Ez * scale
    
    # Plot main vector without normalization to preserve relative magnitudes
    ax.quiver(center[0], center[1], center[2],
              Ex, Ey, Ez,
              color=color,
              arrow_length_ratio=0.2,
              linewidth=3)
    
    if std is not None and std > 0:
        # Create cone of variance based only on the std parameter
        theta = np.linspace(0, 2*np.pi, 20)
        r = std * scale  # Cone size depends only on std and scale
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros_like(theta)
        
        # Transform cone to align with field direction
        magnitude = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        if magnitude > 0:
            direction = np.array([Ex, Ey, Ez]) / magnitude
            z_axis = direction
            x_axis = np.cross(z_axis, [0, 0, 1])
            if np.all(x_axis == 0):
                x_axis = np.cross(z_axis, [0, 1, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            transform = np.vstack((x_axis, y_axis, z_axis))
            
            points = np.vstack((x, y, z))
            transformed = np.dot(transform, points)
            ax.plot_surface(transformed[0] + center[0],
                          transformed[1] + center[1],
                          transformed[2] + center[2],
                          color=color,
                          alpha=0.2)

def generate_tensor_field_animation(
    # Reactant state data
    reactant_donor_tensors: np.ndarray,
    reactant_acceptor_tensors: np.ndarray,
    reactant_donor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    reactant_acceptor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    # Product state data
    product_donor_tensors: np.ndarray,
    product_acceptor_tensors: np.ndarray,
    product_donor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    product_acceptor_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    # Trajectory data for Fe positions
    reactant_donor_xyz: str,
    reactant_acceptor_xyz: str,
    product_donor_xyz: str,
    product_acceptor_xyz: str,
    # Other parameters
    output_path: str,
    fps: int = 15,
    frame_interval: Optional[int] = None,
    first_last_only: bool = False,
    tensor_scale: float = 3.0,
    field_scale: float = 1.0
) -> str:
    """
    Generate animation showing tensor orientations and fields centered at Fe positions.
    
    Args:
        reactant/product_donor/acceptor_tensors: Tensors for each frame
        reactant/product_donor/acceptor_fields: (Ex, Ey, Ez) for each frame
        reactant/product_donor/acceptor_xyz: Paths to XYZ trajectory files
        output_path: Directory to save animation
        fps: Frames per second for animation
        frame_interval: If set, only plot every nth frame
        first_last_only: If True, only plot first and last frames
        tensor_scale: Scaling factor for tensor visualization
        field_scale: Scaling factor for field vector visualization
    
    Returns:
        Path to saved animation file
    """
    print("\nReading XYZ trajectories from:")
    print(f"Reactant Donor:    {os.path.abspath(reactant_donor_xyz)}")
    print(f"Reactant Acceptor: {os.path.abspath(reactant_acceptor_xyz)}")
    print(f"Product Donor:     {os.path.abspath(product_donor_xyz)}")
    print(f"Product Acceptor:  {os.path.abspath(product_acceptor_xyz)}")

    # Read XYZ trajectories
    reactant_donor_frames = rxyz.read_xyz(reactant_donor_xyz)
    reactant_acceptor_frames = rxyz.read_xyz(reactant_acceptor_xyz)
    product_donor_frames = rxyz.read_xyz(product_donor_xyz)
    product_acceptor_frames = rxyz.read_xyz(product_acceptor_xyz)
    print("XYZ trajectories loaded successfully")

    # Get frame counts
    reactant_frames = len(reactant_donor_frames)
    product_frames = len(product_donor_frames)
    total_frames = max(reactant_frames, product_frames)
    
    # Determine which frames to plot based on options
    if first_last_only:
        frames_to_plot = [0, total_frames - 1]
    elif frame_interval:
        frames_to_plot = list(range(0, total_frames, frame_interval))
        if (total_frames - 1) not in frames_to_plot:
            frames_to_plot.append(total_frames - 1)
    else:
        frames_to_plot = list(range(total_frames))
    
    # Calculate plot limits based on Fe positions
    def get_position_limits():
        all_positions = []
        for frames in [reactant_donor_frames, reactant_acceptor_frames,
                      product_donor_frames, product_acceptor_frames]:
            positions = np.array([get_fe_position(frame) for frame in frames])
            all_positions.extend(positions)
        all_positions = np.array(all_positions)
        min_pos = np.min(all_positions, axis=0)
        max_pos = np.max(all_positions, axis=0)
        center = (min_pos + max_pos) / 2
        
        # Calculate range based on actual positions plus padding
        max_range = np.max(max_pos - min_pos)
        # Add padding that's proportional to the actual molecular size
        plot_range = max_range * 1.2  # Reduced padding factor
        
        return center, plot_range

    # Get plot limits once
    center_pos, plot_range = get_position_limits()
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 6))
    
    # Create subplots for reactant and product states
    ax_reactant = fig.add_subplot(121, projection='3d')
    ax_product = fig.add_subplot(122, projection='3d')
    
    # Set consistent viewing angle
    ax_reactant.view_init(elev=20, azim=45)
    ax_product.view_init(elev=20, azim=45)

    def update(frame):
        ax_reactant.clear()
        ax_product.clear()

        # Set labels and limits
        for ax, title in [(ax_reactant, "Reactant State"),
                         (ax_product, "Product State")]:
            ax.set_xlabel("x (Å)")
            ax.set_ylabel("y (Å)")
            ax.set_zlabel("z (Å)")
            ax.set_title(title)

            # Set limits using the pre-calculated values
            padding = plot_range / 2
            ax.set_xlim(center_pos[0] - padding, center_pos[0] + padding)
            ax.set_ylim(center_pos[1] - padding, center_pos[1] + padding)
            ax.set_zlim(center_pos[2] - padding, center_pos[2] + padding)
            
            # Force aspect ratio to be equal
            ax.set_box_aspect([1, 1, 1])

        print(f"\n{'='*50}")
        if first_last_only:
            actual_frame = 0 if frame == 0 else total_frames - 1
            print(f"Frame {frame} (showing first/last frame)")
        elif frame_interval:
            actual_frame = frame                 
            print(f"Frame {frame} (every {frame_interval}th frame, showing frame {actual_frame})")
        else:
            actual_frame = frame
            print(f"Frame {frame} (showing all frames)")
        print(f"{'='*50}")

        # Plot reactant state if frame exists
        if frame < reactant_frames:
            # Get Fe positions for current frame
            reactant_donor_center = get_fe_position(reactant_donor_frames[frame])
            reactant_acceptor_center = get_fe_position(reactant_acceptor_frames[frame])

            # Calculate Fe-Fe distance for reactant state
            reactant_fe_fe_distance = np.linalg.norm(reactant_donor_center - reactant_acceptor_center)

            print("\nReactant State Positions:")
            print(f"Donor Fe:     [{reactant_donor_center[0]:10.6f} {reactant_donor_center[1]:10.6f} {reactant_donor_center[2]:10.6f}]")
            print(f"Acceptor Fe:  [{reactant_acceptor_center[0]:10.6f} {reactant_acceptor_center[1]:10.6f} {reactant_acceptor_center[2]:10.6f}]")
            print(f"Fe-Fe Distance: {reactant_fe_fe_distance:10.6f} Å")

            # Plot tensors and fields at Fe positions
            plot_ellipsoid(ax_reactant, reactant_donor_tensors[frame],
                          reactant_donor_center, 'blue', alpha=0.3, scale=tensor_scale)
            plot_ellipsoid(ax_reactant, reactant_acceptor_tensors[frame],
                          reactant_acceptor_center, 'red', alpha=0.3, scale=tensor_scale)

            plot_efield_vector(ax_reactant,
                             [f[frame] for f in reactant_donor_fields],
                             reactant_donor_center, '#00ff00',
                             np.std([np.linalg.norm([f[frame] for f in reactant_donor_fields])]),
                             scale=field_scale)

            plot_efield_vector(ax_reactant,
                             [f[frame] for f in reactant_acceptor_fields],
                             reactant_acceptor_center, '#00ff00',
                             np.std([np.linalg.norm([f[frame] for f in reactant_acceptor_fields])]),
                             scale=field_scale)

            # Add dotted line connecting Fe atoms with distance label
            ax_reactant.plot([reactant_donor_center[0], reactant_acceptor_center[0]],
                           [reactant_donor_center[1], reactant_acceptor_center[1]],
                           [reactant_donor_center[2], reactant_acceptor_center[2]],
                           'k:', alpha=1.0)

            # Calculate midpoint for label position
            midpoint = (reactant_donor_center + reactant_acceptor_center) / 2
            ax_reactant.text(midpoint[0], midpoint[1], midpoint[2],
                           f'{reactant_fe_fe_distance:.1f} Å',
                           color='black', fontsize=8,
                           horizontalalignment='center',
                           verticalalignment='bottom')

        else:
            print("\nReactant State: No more frames")
            ax_reactant.text(0, 0, 0, "No more frames",
                           ha='center', va='center')

        # Plot product state if frame exists
        if frame < product_frames:
            # Get Fe positions for current frame
            product_donor_center = get_fe_position(product_donor_frames[frame])
            product_acceptor_center = get_fe_position(product_acceptor_frames[frame])

            # Calculate Fe-Fe distance for product state
            product_fe_fe_distance = np.linalg.norm(product_donor_center - product_acceptor_center)

            print("\nProduct State Positions:")
            print(f"Donor Fe:     [{product_donor_center[0]:10.6f} {product_donor_center[1]:10.6f} {product_donor_center[2]:10.6f}]")
            print(f"Acceptor Fe:  [{product_acceptor_center[0]:10.6f} {product_acceptor_center[1]:10.6f} {product_acceptor_center[2]:10.6f}]")
            print(f"Fe-Fe Distance: {product_fe_fe_distance:10.6f} Å")

            # Plot tensors and fields at Fe positions
            plot_ellipsoid(ax_product, product_donor_tensors[frame],
                          product_donor_center, 'blue', alpha=0.3, scale=tensor_scale)
            plot_ellipsoid(ax_product, product_acceptor_tensors[frame],
                          product_acceptor_center, 'red', alpha=0.3, scale=tensor_scale)

            plot_efield_vector(ax_product,
                             [f[frame] for f in product_donor_fields],
                             product_donor_center, '#00ff00',
                             np.std([np.linalg.norm([f[frame] for f in product_donor_fields])]),
                             scale=field_scale)

            plot_efield_vector(ax_product,
                             [f[frame] for f in product_acceptor_fields],
                             product_acceptor_center, '#00ff00',
                             np.std([np.linalg.norm([f[frame] for f in product_acceptor_fields])]),
                             scale=field_scale)

            # Add dotted line connecting Fe atoms with distance label
            ax_product.plot([product_donor_center[0], product_acceptor_center[0]],
                          [product_donor_center[1], product_acceptor_center[1]],
                          [product_donor_center[2], product_acceptor_center[2]],
                          'k:', alpha=1.0)

            # Calculate midpoint for label position
            midpoint = (product_donor_center + product_acceptor_center) / 2
            ax_product.text(midpoint[0], midpoint[1], midpoint[2],
                          f'{product_fe_fe_distance:.1f} Å',
                          color='black', fontsize=8,
                          horizontalalignment='center',
                          verticalalignment='bottom')

        else:
            print("\nProduct State: No more frames")
            ax_product.text(0, 0, 0, "No more frames",
                          ha='center', va='center')

        # Add frame counter and additional info
        frame_info = (f"Frame {frame+1}/{total_frames}"
                     f"\nReactant Frames: {reactant_frames}, Product Frames: {product_frames}")
        if first_last_only:
            frame_info += "\nShowing first and last frames only"
        elif frame_interval:
            frame_info += f"\nShowing every {frame_interval}th frame"
        fig.suptitle(frame_info)

    # Create progress bar
    pbar = tqdm(total=len(frames_to_plot), desc="Generating animation")
    
    # Custom callback for progress updates
    def progress_callback(current_frame):
        pbar.update(1)
    
    # Create animation with progress tracking
    anim = FuncAnimation(fig, update, frames=frames_to_plot, interval=1000/fps)
    anim._start = lambda *args: progress_callback(None)
     
    # Generate output filename
    filename_parts = []
    if first_last_only:
        filename_parts.append("first_last_only")
    elif frame_interval:
        filename_parts.append(f"interval_{frame_interval}")
    
    output_file = os.path.join(
        output_path, 
        f"tensor_field_{'_'.join(filename_parts) if filename_parts else 'trajectory'}.mp4"
    )
    
    # Save animation
    print(f"\nSaving animation to: {output_file}")
    anim.save(output_file, writer='ffmpeg', fps=fps)
    pbar.close()
    plt.close()
    
    return output_file

# Example usage if running as main script
if __name__ == "__main__":
    # Create dummy data for testing
    n_frames = 100
    dummy_tensor = np.array([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]]] * n_frames)
    dummy_fields = (np.zeros(n_frames), np.zeros(n_frames), np.ones(n_frames))
    
    # Create dummy xyz files for testing
    def create_dummy_xyz(filename):
        with open(filename, 'w') as f:
            for _ in range(n_frames):
                f.write("2\nDummy molecule\nFe 0 0 0\nC 1 1 1\n")
    
    # Test files
    test_files = ["test_donor.xyz", "test_acceptor.xyz", 
                  "test_product_donor.xyz", "test_product_acceptor.xyz"]
    for file in test_files:
        create_dummy_xyz(file)
    
    # Test the animation with different options
    output_path = "."
    
    # Test with all frames
    generate_tensor_field_animation(
        dummy_tensor, dummy_tensor, dummy_fields, dummy_fields,
        dummy_tensor, dummy_tensor, dummy_fields, dummy_fields,
        "test_donor.xyz", "test_acceptor.xyz",
        "test_product_donor.xyz", "test_product_acceptor.xyz",
        output_path=output_path,
        tensor_scale=1.0,
        field_scale=1.0
    )
    
    # Test with first and last frames only
    generate_tensor_field_animation(
        dummy_tensor, dummy_tensor, dummy_fields, dummy_fields,
        dummy_tensor, dummy_tensor, dummy_fields, dummy_fields,
        "test_donor.xyz", "test_acceptor.xyz",
        "test_product_donor.xyz", "test_product_acceptor.xyz",
        output_path=output_path,
        first_last_only=True,
        tensor_scale=1.0,
        field_scale=1.0
    )
    
    # Clean up test files
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
