import os
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import angle_between_vectors as abv
import create_output_folder as cof

def plot_visualize_polarizability(folder_path,
    reactant_donor_avg_alpha_red, reactant_donor_avg_alpha_ox,
    reactant_acceptor_avg_alpha_ox, reactant_acceptor_avg_alpha_red,
    reactant_E_x_donor, reactant_E_y_donor, reactant_E_z_donor,
    reactant_E_x_acceptor, reactant_E_y_acceptor, reactant_E_z_acceptor,
    product_E_x_donor, product_E_y_donor, product_E_z_donor,
    product_E_x_acceptor, product_E_y_acceptor, product_E_z_acceptor):

    # Ensure there are electric field vectors to plot
    if not reactant_E_x_donor.size or not product_E_x_donor.size:
        print(" **No electric field vectors selected. Skipping plotting.**")
        return

    def convert_to_tensor(alpha):
        """
        Convert a 1D array of alpha components into a 3x3 tensor.
        The input alpha should be in the order [xx, xy, xz, yy, yz, zz].
        """
        tensor = np.array([
            [alpha[0], alpha[1], alpha[2]],
            [alpha[1], alpha[3], alpha[4]],
            [alpha[2], alpha[4], alpha[5]]
        ])
        return tensor

    def plot_ellipsoid(ax, tensor, center=[0, 0, 0], color='b', scale=0.20, alpha=0.05):
        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)

        # Handle negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, 0)

        # Radii corresponding to eigenvalues
        radii = np.sqrt(eigenvalues)

        # Create grid and multivariate normal
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        # Rotate data with eigenvectors
        for i in range(len(x)):
            for j in range(len(x[0])):
                [x[i, j], y[i, j], z[i, j]] = np.dot(np.array([x[i, j], y[i, j], z[i, j]]), eigenvectors) + center

        # Plot ellipsoid as a wireframe
        ax.plot_surface(x * scale, y * scale, z * scale, color=color, alpha=alpha, linewidth=0, antialiased=True)

    def plot_cone(ax, start, end, radius, color='b', resolution=100, alpha=0.5):
        """
        Plot a cone along a vector from start to end.
        """
        vector = end - start
        length = np.linalg.norm(vector)
        direction = vector / length

        # Create points on circle perpendicular to the vector
        theta = np.linspace(0, 2*np.pi, resolution)
        v = np.array([-direction[1], direction[0], 0])
        if np.allclose(v, 0):
            v = np.array([0, -direction[2], direction[1]])
        v = v / np.linalg.norm(v)
        w = np.cross(direction, v)

        # Create cone surface
        t = np.linspace(0, length, resolution)
        X, T = np.meshgrid(theta, t)
        R = radius * (1 - T/length)
        x = start[0] + T*direction[0] + R*(np.cos(X)*v[0] + np.sin(X)*w[0])
        y = start[1] + T*direction[1] + R*(np.cos(X)*v[1] + np.sin(X)*w[1])
        z = start[2] + T*direction[2] + R*(np.cos(X)*v[2] + np.sin(X)*w[2])

        # Plot the cone
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)

    # Convert 1D arrays to 3x3 tensors
#   reactant_donor_avg_alpha_red_tensor   = reactant_donor_avg_alpha_red 
#   reactant_acceptor_avg_alpha_ox_tensor = reactant_acceptor_avg_alpha_ox 
#   product_donor_avg_alpha_ox_tensor     = product_donor_avg_alpha_ox 
#   product_acceptor_avg_alpha_red_tensor = product_acceptor_avg_alpha_red 
    
    # Electric field vectors for donor and acceptor
    reactant_E_donor    = np.array([reactant_E_x_donor, reactant_E_y_donor, reactant_E_z_donor])
    reactant_E_acceptor = np.array([reactant_E_x_acceptor, reactant_E_y_acceptor, reactant_E_z_acceptor])
    product_E_donor     = np.array([product_E_x_donor, product_E_y_donor, product_E_z_donor])
    product_E_acceptor  = np.array([product_E_x_acceptor, product_E_y_acceptor, product_E_z_acceptor])

    reactant_donor_mag    = np.sqrt(reactant_E_x_donor**2    + reactant_E_y_donor**2    + reactant_E_z_donor**2)
    reactant_acceptor_mag = np.sqrt(reactant_E_x_acceptor**2 + reactant_E_y_acceptor**2 + reactant_E_z_acceptor**2)
    product_donor_mag     = np.sqrt(product_E_x_donor**2     + product_E_y_donor**2     + product_E_z_donor**2)
    product_acceptor_mag  = np.sqrt(product_E_x_acceptor**2  + product_E_y_acceptor**2  + product_E_z_acceptor**2)

    print("\n Reactant E donor magnitudes:")
    print("   Min:", np.min(reactant_donor_mag))
    print("   Max:", np.max(reactant_donor_mag))
    print("   Mean:", np.mean(reactant_donor_mag))
    print("\n Product E donor magnitudes:")
    print("   Min:", np.min(product_donor_mag))
    print("   Max:", np.max(product_donor_mag))
    print("   Mean:", np.mean(product_donor_mag))

    # Compute average and standard deviation of electric field vectors
    reactant_E_donor_avg    = np.array([np.mean(reactant_E_x_donor), np.mean(reactant_E_y_donor), np.mean(reactant_E_z_donor)])
    reactant_E_donor_std    = np.array([np.std(reactant_E_x_donor), np.std(reactant_E_y_donor), np.std(reactant_E_z_donor)])
    reactant_E_acceptor_avg = np.array([np.mean(reactant_E_x_acceptor), np.mean(reactant_E_y_acceptor), np.mean(reactant_E_z_acceptor)])
    reactant_E_acceptor_std = np.array([np.std(reactant_E_x_acceptor), np.std(reactant_E_y_acceptor), np.std(reactant_E_z_acceptor)])
    product_E_donor_avg     = np.array([np.mean(product_E_x_donor), np.mean(product_E_y_donor), np.mean(product_E_z_donor)])
    product_E_donor_std     = np.array([np.std(product_E_x_donor), np.std(product_E_y_donor), np.std(product_E_z_donor)])
    product_E_acceptor_avg  = np.array([np.mean(product_E_x_acceptor), np.mean(product_E_y_acceptor), np.mean(product_E_z_acceptor)])
    product_E_acceptor_std  = np.array([np.std(product_E_x_acceptor), np.std(product_E_y_acceptor), np.std(product_E_z_acceptor)])

    # Normalize average electric field vectors
    reactant_E_donor_avg_norm    = reactant_E_donor_avg / np.linalg.norm(reactant_E_donor_avg)
    reactant_E_acceptor_avg_norm = reactant_E_acceptor_avg / np.linalg.norm(reactant_E_acceptor_avg)
    product_E_donor_avg_norm     = product_E_donor_avg / np.linalg.norm(product_E_donor_avg)
    product_E_acceptor_avg_norm  = product_E_acceptor_avg / np.linalg.norm(product_E_acceptor_avg)

    donor_efield_rot    = abv.angle_between_vectors(reactant_E_donor_avg_norm, reactant_E_acceptor_avg_norm)
    acceptor_efield_rot = abv.angle_between_vectors(product_E_donor_avg_norm, product_E_acceptor_avg_norm)
    
    print(f"\n")
    print(f" reactant_E_donor_avg         = {reactant_E_donor_avg}")
    print(f" reactant_E_donor_avg_norm    = {reactant_E_donor_avg_norm}")
    print(f" reactant_E_donor_std         = {reactant_E_donor_std}")
    print(f" reactant_E_acceptor_avg      = {reactant_E_acceptor_avg}")
    print(f" reactant_E_acceptor_avg_norm = {reactant_E_acceptor_avg_norm}")
    print(f" reactant_E_acceptor_std      = {reactant_E_acceptor_std}")

    print(f" product_E_donor_avg          = {product_E_donor_avg}")
    print(f" product_E_donor_avg_norm     = {product_E_donor_avg_norm}")
    print(f" product_E_donor_std          = {product_E_donor_std}")
    print(f" product_E_acceptor_avg       = {product_E_acceptor_avg}")
    print(f" product_E_acceptor_avg_norm  = {product_E_acceptor_avg_norm}")
    print(f" product_E_acceptor_std       = {product_E_acceptor_std}")

    print(f" donor_efield_rot             = {donor_efield_rot}")
    print(f" acceptor_efield_rot          = {acceptor_efield_rot}")

    # Define the subset size
    reactant_subset_size = 10                      
    product_subset_size  = 10                      

    # Randomly sample indices
    reactant_random_indices = np.random.choice(reactant_E_donor.shape[1], reactant_subset_size, replace=False)
    product_random_indices  = np.random.choice(product_E_donor.shape[1], product_subset_size, replace=False)

    # Sample the vectors
    sampled_reactant_E_donor    = reactant_E_donor[:, reactant_random_indices]
    sampled_product_E_donor     = product_E_donor[:, product_random_indices]
    sampled_reactant_E_acceptor = reactant_E_acceptor[:, reactant_random_indices]
    sampled_product_E_acceptor  = product_E_acceptor[:, product_random_indices]

    # Normalize the sampled vectors
    reactant_E_donor_norm    = sampled_reactant_E_donor / np.mean(reactant_donor_mag)
    product_E_donor_norm     = sampled_product_E_donor / np.mean(product_donor_mag)
    reactant_E_acceptor_norm = sampled_reactant_E_acceptor / np.mean(reactant_acceptor_mag)
    product_E_acceptor_norm  = sampled_product_E_acceptor / np.mean(product_acceptor_mag)

    # Plot donor vectors
    fig1 = plt.figure(figsize=(3.3, 3.3))
    ax0 = fig1.add_subplot(111, projection='3d')
    ax0.view_init(elev=30, azim=30)

    for i in range(reactant_subset_size):
        ax0.quiver(0.0, 0.0, 0.0,
            reactant_E_donor_norm[0, i],
            reactant_E_donor_norm[1, i],
            reactant_E_donor_norm[2, i],
            length=1.0, normalize=False, color='red', arrow_length_ratio=0.1, linewidth=1)
    for i in range(reactant_subset_size):
        ax0.quiver(0.0, 0.0, 0.0,
            reactant_E_acceptor_norm[0, i],
            reactant_E_acceptor_norm[1, i],
            reactant_E_acceptor_norm[2, i],
            length=1.0, normalize=False, color='blue', arrow_length_ratio=0.1, linewidth=1)
    ax0.set_xlim([-1.05, 1.05]); ax0.set_ylim([-1.05, 1.05]); ax0.set_zlim([-1.05, 1.05])
    ax0.tick_params(axis='x', pad=0)
    ax0.tick_params(axis='y', pad=0)
    ax0.tick_params(axis='z', pad=0)
    ax0.set_xlabel('X', labelpad=0, fontsize=12)
    ax0.set_ylabel('Y', labelpad=0, fontsize=12)
    ax0.set_zlabel('Z', labelpad=-2, fontsize=12)

    cof.create_output_folder(folder_path)
    filename="Visualize_Reactant_State_Electric_Field_Vectors.png"
    file_path = os.path.join(folder_path, filename)

#   fig1.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(file_path, dpi=600)
    print(f' **Figure saved to: {file_path}')

    # Plot acceptor vectors
    fig2 = plt.figure(figsize=(3.3, 3.3))
    ax1 = fig2.add_subplot(111, projection='3d')
    ax1.view_init(elev=30, azim=30)

    for i in range(product_subset_size):
        ax1.quiver(0.0, 0.0, 0.0,
            product_E_donor_norm[0, i],
            product_E_donor_norm[1, i],
            product_E_donor_norm[2, i],
            length=1.0, normalize=False, color='red', arrow_length_ratio=0.1, linewidth=1)
    for i in range(product_subset_size):
        ax1.quiver(0.0, 0.0, 0.0,
            product_E_acceptor_norm[0, i],
            product_E_acceptor_norm[1, i],
            product_E_acceptor_norm[2, i],
            length=1.0, normalize=False, color='blue', arrow_length_ratio=0.1, linewidth=1)
    ax1.set_xlim([-1.05, 1.05]); ax1.set_ylim([-1.05, 1.05]); ax1.set_zlim([-1.05, 1.05])
    ax1.set_xlabel('X', labelpad=0, fontsize=12)
    ax1.set_ylabel('Y', labelpad=0, fontsize=12)
    ax1.set_zlabel('Z', labelpad=-2, fontsize=12)

    cof.create_output_folder(folder_path)
    filename="Visualize_Product_State_Electric_Field_Vectors.png"
    file_path = os.path.join(folder_path, filename)

    #fig2.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    plt.savefig(file_path, dpi=600)
    print(f' **Figure saved to: {file_path}')

    ########################################################
    #Plot Polarizability and Electric Fields
    fig2 = plt.figure(figsize=(7.0, 3.5))

    # Plot for reactant donor
    ax2 = fig2.add_subplot(121, projection='3d')
    ax2.view_init(elev=30, azim=30)
    ax3 = fig2.add_subplot(122, projection='3d')
    ax3.view_init(elev=30, azim=30)

    plot_ellipsoid(ax2, reactant_donor_avg_alpha_red, center=[0, 0, 0], color='r', scale=0.20, alpha=0.1)
    plot_ellipsoid(ax2, reactant_donor_avg_alpha_ox, center=[0, 0, 0], color='b', scale=0.20, alpha=0.1)
    plot_ellipsoid(ax3, reactant_acceptor_avg_alpha_ox, center=[0, 0, 0], color='b', scale=0.20, alpha=0.1)
    plot_ellipsoid(ax3, reactant_acceptor_avg_alpha_red, center=[0, 0, 0], color='r', scale=0.20, alpha=0.1)

    # Plot cones and vectors with higher opacity
    start = np.array([reactant_E_donor_avg_norm[0], reactant_E_donor_avg_norm[1], reactant_E_donor_avg_norm[2]]) 
    end   = np.array([0, 0, 0])
    plot_cone(ax2, start, end, radius=np.mean(reactant_E_donor_std), color='orange', alpha=0.7)
    ax2.quiver(0, 0, 0, *reactant_E_donor_avg_norm, length=1.0, color='k', arrow_length_ratio=0.2, linewidth=2)

    start = np.array([product_E_donor_avg_norm[0], product_E_donor_avg_norm[1], product_E_donor_avg_norm[2]]) 
    end   = np.array([0, 0, 0])
    plot_cone(ax2, start, end, radius=np.mean(product_E_donor_std), color='lime', alpha=0.7)
    ax2.quiver(0, 0, 0, *product_E_donor_avg_norm, length=1.0, color='k', arrow_length_ratio=0.2, linewidth=2)

    start = np.array([reactant_E_acceptor_avg_norm[0], reactant_E_acceptor_avg_norm[1], reactant_E_acceptor_avg_norm[2]]) 
    end   = np.array([0, 0, 0])
    plot_cone(ax3, start, end, radius=np.mean(reactant_E_acceptor_std), color='orange', alpha=0.7)
    ax3.quiver(0, 0, 0, *reactant_E_acceptor_avg_norm, length=1.0, color='k', arrow_length_ratio=0.2, linewidth=2)

    start = np.array([product_E_acceptor_avg_norm[0], product_E_acceptor_avg_norm[1], product_E_acceptor_avg_norm[2]]) 
    end   = np.array([0, 0, 0])
    plot_cone(ax3, start, end, radius=np.mean(product_E_acceptor_std), color='lime', alpha=0.7)
    ax3.quiver(0, 0, 0, *product_E_acceptor_avg_norm, length=1.0, color='k', arrow_length_ratio=0.2, linewidth=2)

    ax2.set_xlim([-2, 2]); ax2.set_ylim([-2, 2]); ax2.set_zlim([-2, 2])
    ax3.set_xlim([-2, 2]); ax3.set_ylim([-2, 2]); ax3.set_zlim([-2, 2])

    fontsize =  12  
    ax2.set_xlabel('X', labelpad=-14, fontsize=fontsize)
    ax2.set_ylabel('Y', labelpad=-14, fontsize=fontsize)
    ax2.set_zlabel('Z', labelpad=-16, fontsize=fontsize)
    ax3.set_xlabel('X', labelpad=-14, fontsize=fontsize)
    ax3.set_ylabel('Y', labelpad=-14, fontsize=fontsize)
    ax3.set_zlabel('Z', labelpad=-16, fontsize=fontsize)

    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])

    # Ensure the output folder exists
    cof.create_output_folder(folder_path)

    # Define the full file path
    filename="Visualize_Polarizability_Ellipsoid.png"
    file_path = os.path.join(folder_path, filename)

    # Save the figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=600)

    print(f' **Figure saved to: {file_path}')
