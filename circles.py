import numpy as np
import matplotlib.pyplot as plt

def plot_phase_circles(angle1_left, angle2_left, angle1_right, angle2_right):
    """Plots two circles with phase vectors, a single legend, and 2pi indicator."""

    def plot_circle_with_vectors(ax, angle1, angle2, title, is_first_plot=False):
        """Helper function to plot a single circle with vectors."""

        # Circle
        circle = plt.Circle((0, 0), 1, color='black', fill=False)
        
        ax.add_artist(circle)

        # Center dot
        ax.plot(0, 0, 'k.')

        # Vectors
        x1, y1 = np.cos(angle1), np.sin(angle1)
        x2, y2 = np.cos(angle2), np.sin(angle2)

        # Plot vectors and only add labels to the first plot
        ax.arrow(0, 0, x1, y1, head_width=0.1, head_length=0.1, fc='red', ec='red',
                 label='Signal 1' if is_first_plot else None)
        ax.arrow(0, 0, x2, y2, head_width=0.1, head_length=0.1, fc='green', ec='green',
                 label='Signal 2' if is_first_plot else None)

        # Average vector
        avg_x = (x1 + x2) / 2
        avg_y = (y1 + y2) / 2
        ax.arrow(0, 0, avg_x, avg_y, head_width=0.1, head_length=0.1, fc='blue', ec='blue',
                 label='Average' if is_first_plot else None)

        # 2pi indicator (leftward radius)
        #ax.plot([-1.2, -1], [0, 0], 'k-', lw=0.5)  # Add a small line
        #ax.text(-1.3, 0, '2π (0)', ha='right', va='center', fontsize=8)  # Add text

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.grid(True)
    ax2.grid(True)
    plot_circle_with_vectors(ax1, angle1_left, angle2_left, 'Close Phase Angles', is_first_plot=True)  # Add labels here
    plot_circle_with_vectors(ax2, angle1_right, angle2_right, 'Distant Phase Angles')

    # Create a single legend for the whole figure
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=3)  # Adjust position as needed

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make space for the legend

    plt.show()

complex_list = [
    np.exp(1j * 0),  # Unit magnitude, angle pi/4
    np.exp(1j * np.pi / 4),  # Unit magnitude, angle pi/3
    #np.exp(1j * np.pi / 6),  # Unit magnitude, angle pi/6
    #np.exp(1j * np.pi / 2), # Unit magnitude, angle pi/2
]

average = np.mean(complex_list)

print(np.abs(average))

# Example usage:
angle1_left = 0 # radians
angle2_left = np.pi / 4  # radians
angle1_right = -0.2  # radians
angle2_right = 2.3 # radians

plot_phase_circles(angle1_left, angle2_left, angle1_right, angle2_right)