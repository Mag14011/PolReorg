import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy import stats

def read_efield_data(filename):
    """
    Read electric field data from file, skipping header lines starting with @ or #
    and the last 3 lines containing summary information
    """
    times, magnitudes, ex, ey, ez = [], [], [], [], []
    
    # First read all valid lines into a list
    valid_lines = []
    with open(filename, 'r') as f:
        for line in f:
            if not (line.startswith('@') or line.startswith('#')):
                valid_lines.append(line)
    
    # Process all lines except the last 3
    for line in valid_lines[:-3]:
        data = line.split()
        if len(data) == 5:
            times.append(float(data[0]))
            magnitudes.append(float(data[1]))
            ex.append(float(data[2]))
            ey.append(float(data[3]))
            ez.append(float(data[4]))
    
    return np.array(times), np.array(magnitudes), np.array(ex), np.array(ey), np.array(ez)

def print_distribution_stats(name, data):
    """
    Print statistical information about a distribution
    """
    print(f"\n{name} Statistics:")
    print("-" * (len(name) + 11))
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Median: {np.median(data):.3f}")
    print(f"Standard Deviation: {np.std(data):.3f}")
    print(f"Minimum: {np.min(data):.3f}")
    print(f"Maximum: {np.max(data):.3f}")
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")
    
    # Perform Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test p-value: {p_value:.3e}")
    print(f"Distribution is {'normally distributed' if p_value > 0.05 else 'not normally distributed'}")

def create_static_plots(times, magnitudes, ex, ey, ez, output_file=None):
    """
    Create static 2D plots of electric field data with histogram side panels
    """
    # Print statistical information
    print("\nDistribution Statistics")
    print("=====================")
    
    print_distribution_stats("Magnitude", magnitudes)
    print_distribution_stats("Ex Component", ex)
    print_distribution_stats("Ey Component", ey)
    print_distribution_stats("Ez Component", ez)
    
    # Calculate correlations between components
    print("\nComponent Correlations")
    print("====================")
    print(f"Ex-Ey correlation: {np.corrcoef(ex, ey)[0,1]:.3f}")
    print(f"Ex-Ez correlation: {np.corrcoef(ex, ez)[0,1]:.3f}")
    print(f"Ey-Ez correlation: {np.corrcoef(ey, ez)[0,1]:.3f}")
    
    # Print time series statistics
    print("\nTime Series Information")
    print("=====================")
    print(f"Time range: {times[0]:.1f} to {times[-1]:.1f} ps")
    print(f"Number of data points: {len(times)}")
    print(f"Time step: {np.mean(np.diff(times)):.2f} ps")
    
    # Convert time from ps to ns
    times_ns = times / 1000.0
    
    fig = plt.figure(figsize=(6.6, 6.6))
    
    # Create main grid
    gs = GridSpec(2, 2, figure=fig, 
                 width_ratios=[4, 1],
                 height_ratios=[1, 1],
                 left=0.1, right=0.95, 
                 top=0.95, bottom=0.1,
                 hspace=0, wspace=0)
    
    # Add a border around the entire figure
    fig.patch.set_linewidth(1)
    fig.patch.set_edgecolor('black')
    
    # Magnitude plot
    ax_mag = fig.add_subplot(gs[0, 0])
    ax_mag.plot(times_ns, magnitudes, 'k-', linewidth=1, label='Magnitude')
    ax_mag.set_ylabel('Magnitude (MV/cm)')
    ax_mag.legend(frameon=False)
    plt.setp(ax_mag.get_xticklabels(), visible=False)
    ax_mag.tick_params(axis='both', direction='in')
    
    # Magnitude histogram
    ax_mag_hist = fig.add_subplot(gs[0, 1], sharey=ax_mag)
    ax_mag_hist.hist(magnitudes, bins=50, orientation='horizontal', color='gray', alpha=0.7)
    plt.setp(ax_mag_hist.get_yticklabels(), visible=False)
    plt.setp(ax_mag_hist.get_xticklabels(), rotation=0)
    ax_mag_hist.set_xlabel('Count')
    ax_mag_hist.tick_params(axis='both', direction='in')
    
    # Components plot
    ax_comp = fig.add_subplot(gs[1, 0], sharex=ax_mag)
    ax_comp.plot(times_ns, ex, 'r-', label='E$_x$', linewidth=1)
    ax_comp.plot(times_ns, ey, 'g-', label='E$_y$', linewidth=1)
    ax_comp.plot(times_ns, ez, 'b-', label='E$_z$', linewidth=1)
    ax_comp.set_xlabel('Time (ns)')
    ax_comp.set_ylabel('Component (MV/cm)')
    ax_comp.legend(frameon=False)
    ax_comp.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
    ax_comp.tick_params(axis='both', direction='in')
    
    # Components histogram
    ax_comp_hist = fig.add_subplot(gs[1, 1], sharey=ax_comp)
    ax_comp_hist.hist(ex, bins=50, orientation='horizontal', color='red', alpha=0.3)
    ax_comp_hist.hist(ey, bins=50, orientation='horizontal', color='green', alpha=0.3)
    ax_comp_hist.hist(ez, bins=50, orientation='horizontal', color='blue', alpha=0.3)
    ax_comp_hist.set_xlabel('Count')
    plt.setp(ax_comp_hist.get_yticklabels(), visible=False)
    ax_comp_hist.tick_params(axis='both', direction='in')
    
    # Remove grid lines from all plots
    ax_mag.grid(False)
    ax_mag_hist.grid(False)
    ax_comp.grid(False)
    ax_comp_hist.grid(False)
    
    # Set spine visibility
    ax_mag.spines['right'].set_visible(False)
    ax_mag.spines['left'].set_visible(True)
    ax_mag.spines['top'].set_visible(True)
    ax_mag.spines['bottom'].set_visible(True)
    
    ax_mag_hist.spines['right'].set_visible(True)
    ax_mag_hist.spines['left'].set_visible(False)
    ax_mag_hist.spines['top'].set_visible(True)
    ax_mag_hist.spines['bottom'].set_visible(True)
    
    ax_comp.spines['right'].set_visible(False)
    ax_comp.spines['left'].set_visible(True)
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['bottom'].set_visible(True)
    
    ax_comp_hist.spines['right'].set_visible(True)
    ax_comp_hist.spines['left'].set_visible(False)
    ax_comp_hist.spines['top'].set_visible(False)
    ax_comp_hist.spines['bottom'].set_visible(True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', edgecolor='black')
        print(f"Static plots saved as {output_file}")
    else:
        plt.show()
    plt.close()

class EFieldAnimator:
    def __init__(self, times, ex, ey, ez):
        self.times = times / 1000.0  # Convert to ns
        self.ex = ex
        self.ey = ey
        self.ez = ez
        
        # Create figure and 3D axes
        self.fig = plt.figure(figsize=(6.6, 6.6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize the vector plot
        self.vector = None
        self.time_text = None
        
        # Calculate axis limits
        max_magnitude = max(np.max(np.abs(ex)), np.max(np.abs(ey)), np.max(np.abs(ez)))
        self.limit = max_magnitude * 1.2
        
    def init_plot(self):
        """Initialize the animation"""
        self.ax.set_xlim([-self.limit, self.limit])
        self.ax.set_ylim([-self.limit, self.limit])
        self.ax.set_zlim([-self.limit, self.limit])
        
        self.ax.set_xlabel('Ex (MV/cm)')
        self.ax.set_ylabel('Ey (MV/cm)')
        self.ax.set_zlabel('Ez (MV/cm)')
        
        # Add coordinate system arrows
        arrow_length = self.limit * 0.2
        self.ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', alpha=0.5)
        self.ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', alpha=0.5)
        self.ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', alpha=0.5)
        
        # Initialize vector and time text
        self.vector = self.ax.quiver(0, 0, 0, self.ex[0], self.ey[0], self.ez[0],
                                   color='purple', linewidth=2, alpha=0.8)
        self.time_text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes)
        
        return self.vector, self.time_text
        
    def update(self, frame):
        """Update function for animation"""
        if self.vector is not None:
            self.vector.remove()
        
        self.vector = self.ax.quiver(0, 0, 0, 
                                   self.ex[frame], self.ey[frame], self.ez[frame],
                                   color='purple', linewidth=2, alpha=0.8)
        
        self.time_text.set_text(f'Time: {self.times[frame]:.1f} ns')
        self.ax.view_init(elev=20, azim=frame)
        
        return self.vector, self.time_text

def create_animation(times, ex, ey, ez, output_file, fps=10):
    """
    Create and save a 3D animation of the electric field vector
    """
    animator = EFieldAnimator(times, ex, ey, ez)
    
    anim = FuncAnimation(animator.fig, animator.update,
                        init_func=animator.init_plot,
                        frames=len(times),
                        interval=1000/fps,
                        blit=True)
    
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_file, writer=writer)
    plt.close()
    print(f"Animation saved as {output_file}")

def main(input_file, static_plot_file=None, animation_file=None):
    """
    Main function to generate both static plots and animation
    """
    # Read data
    times, magnitudes, ex, ey, ez = read_efield_data(input_file)
    
    # Create static plots
    if static_plot_file:
        create_static_plots(times, magnitudes, ex, ey, ez, static_plot_file)
    
    # Create animation
    if animation_file:
        create_animation(times, ex, ey, ez, animation_file)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize electric field data')
    parser.add_argument('input_file', help='Input data file')
    parser.add_argument('--static', help='Output file for static plots (e.g., plots.png)')
    parser.add_argument('--animation', help='Output file for animation (e.g., animation.mp4)')
    
    args = parser.parse_args()
    
    if not (args.static or args.animation):
        print("Please specify at least one output file using --static and/or --animation")
        sys.exit(1)
    
    main(args.input_file, args.static, args.animation)
