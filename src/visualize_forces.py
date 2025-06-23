import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import sys


def load_data():
    """Load vertex positions and force data from numpy files."""
    # Load vertex positions
    # vertices_path = "./out/sensor_vertices.npy"
    vertices_path = "./out/sensor_verts.npy"
    vertices = np.load(vertices_path)
    print(f"Loaded vertices with shape: {vertices.shape}")

    # Load force data
    forces_path = "./out/forces.npy"
    if not os.path.exists(forces_path):
        raise FileNotFoundError(f"Forces file not found: {forces_path}")

    forces = np.load(forces_path)
    print(f"Loaded forces with shape: {forces.shape}")

    # Handle force data shape: (timesteps, 1 object, n_verts, 3 xyz)
    if len(forces.shape) == 4:
        forces = forces[:, 0, :, :]  # Get first object
    elif len(forces.shape) == 3:
        pass  # Already in correct shape (timesteps, n_verts, 3)
    else:
        raise ValueError(f"Unexpected forces shape: {forces.shape}")

    print(f"Final forces shape: {forces.shape}")

    # Ensure vertices and forces have compatible dimensions
    if vertices.shape[0] != forces.shape[1]:
        min_verts = min(vertices.shape[0], forces.shape[1])
        vertices = vertices[:min_verts]
        forces = forces[:, :min_verts, :]
        print(f"Adjusted data to {min_verts} vertices")

    return vertices, forces


def calculate_force_magnitudes(forces):
    """Calculate force magnitudes for each vertex at each timestep."""
    # forces shape: (timesteps, n_verts, 3)
    force_magnitudes = np.linalg.norm(forces, axis=2)
    return force_magnitudes


class ForceVisualizer:
    def __init__(self, vertices, forces, show_vectors=False, vector_scale=1.0, vertex_mask="all"):
        self.vertices = vertices
        self.forces = forces
        self.force_magnitudes = calculate_force_magnitudes(forces)
        self.timesteps = forces.shape[0]
        self.current_timestep = 0
        self.show_vectors = show_vectors
        self.vector_scale = vector_scale
        self.vertex_mask = vertex_mask
        
        # Calculate vertex mask indices
        self.mask_indices = self._calculate_mask_indices()

        # Set up the figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        # Store quiver plot for vector updates
        self.quiver = None

        # Initialize the scatter plot
        self.setup_plot()

    def _calculate_mask_indices(self):
        """Calculate which vertices to show based on the current mask."""
        if self.vertex_mask == "all":
            return np.arange(len(self.vertices))
        elif self.vertex_mask == "+x":
            return np.where(self.vertices[:, 0] >= 0)[0]
        elif self.vertex_mask == "-x":
            return np.where(self.vertices[:, 0] < 0)[0]
        elif self.vertex_mask == "+y":
            return np.where(self.vertices[:, 1] >= 0)[0]
        elif self.vertex_mask == "-y":
            return np.where(self.vertices[:, 1] < 0)[0]
        elif self.vertex_mask == "+z":
            return np.where(self.vertices[:, 2] >= 0)[0]
        elif self.vertex_mask == "-z":
            return np.where(self.vertices[:, 2] < 0)[0]
        else:
            return np.arange(len(self.vertices))

    def set_vertex_mask(self, mask_type):
        """Set the vertex mask and update the plot."""
        self.vertex_mask = mask_type
        self.mask_indices = self._calculate_mask_indices()
        print(f"Showing {len(self.mask_indices)} vertices (mask: {mask_type})")
        self.setup_plot()
        self.update_plot(self.current_timestep)

    def setup_plot(self):
        """Set up the initial 3D plot."""
        # Clear the axes
        self.ax.clear()
        
        # Get masked vertices and forces
        masked_vertices = self.vertices[self.mask_indices]
        masked_magnitudes = self.force_magnitudes[:, self.mask_indices]
        
        # Get the range of force magnitudes for consistent colormap
        self.vmin = np.min(self.force_magnitudes)
        self.vmax = np.max(self.force_magnitudes)

        # Create initial scatter plot with masked vertices
        initial_colors = masked_magnitudes[0]
        self.scatter = self.ax.scatter(
            masked_vertices[:, 0],
            masked_vertices[:, 1],
            masked_vertices[:, 2],
            c=initial_colors,
            cmap="viridis",
            vmin=self.vmin,
            vmax=self.vmax,
            s=50,
            alpha=0.8,
        )

        # Add colorbar
        cbar = self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.5, aspect=20)
        cbar.set_label("Force Magnitude", rotation=270, labelpad=15)

        # Set labels and title
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.set_zlabel("Z Position")
        self.ax.set_title(
            f"Force Visualization - Timestep: {self.current_timestep}/{self.timesteps-1} (Mask: {self.vertex_mask})"
        )

        # Add force vectors if enabled
        if self.show_vectors:
            self.add_force_vectors(self.current_timestep)

        # Set equal aspect ratio based on all vertices (not just masked ones)
        max_range = (
            np.array(
                [
                    self.vertices[:, 0].max() - self.vertices[:, 0].min(),
                    self.vertices[:, 1].max() - self.vertices[:, 1].min(),
                    self.vertices[:, 2].max() - self.vertices[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (self.vertices[:, 0].max() + self.vertices[:, 0].min()) * 0.5
        mid_y = (self.vertices[:, 1].max() + self.vertices[:, 1].min()) * 0.5
        mid_z = (self.vertices[:, 2].max() + self.vertices[:, 2].min()) * 0.5

        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def add_force_vectors(self, timestep):
        """Add force vectors to the plot for a given timestep."""
        # Remove existing quiver plot if it exists
        if self.quiver is not None:
            self.quiver.remove()
        
        # Get masked vertices and forces for this timestep
        masked_vertices = self.vertices[self.mask_indices]
        current_forces = self.forces[timestep][self.mask_indices]
        current_magnitudes = self.force_magnitudes[timestep][self.mask_indices]
        
        # Scale the force vectors
        scaled_forces = current_forces * self.vector_scale
        
        # Create colormap for vectors based on magnitude
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        colors = plt.cm.viridis(norm(current_magnitudes))
        
        # Add quiver plot for force vectors
        self.quiver = self.ax.quiver(
            masked_vertices[:, 0], masked_vertices[:, 1], masked_vertices[:, 2],
            scaled_forces[:, 0], scaled_forces[:, 1], scaled_forces[:, 2],
            colors=colors,
            alpha=0.7,
            arrow_length_ratio=0.1,
            linewidth=1.5
        )

    def update_plot(self, timestep):
        """Update the plot for a given timestep."""
        self.current_timestep = timestep
        
        # Get masked colors for this timestep
        colors = self.force_magnitudes[timestep][self.mask_indices]

        # Update scatter plot colors
        self.scatter.set_array(colors)
        
        # Update force vectors if enabled
        if self.show_vectors:
            self.add_force_vectors(timestep)
        
        self.ax.set_title(
            f"Force Visualization - Timestep: {timestep}/{self.timesteps-1} (Mask: {self.vertex_mask})"
        )

        return (self.scatter,)

    def animate(self, frame):
        """Animation function for matplotlib animation."""
        return self.update_plot(frame)

    def create_animation(self, interval=100, save_path=None):
        """Create and optionally save an animation."""
        ani = animation.FuncAnimation(
            self.fig,
            self.animate,
            frames=self.timesteps,
            interval=interval,
            blit=False,
            repeat=True,
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer="pillow", fps=10)
            print("Animation saved!")

        return ani

    def show_interactive(self):
        """Show interactive plot with keyboard controls."""

        def on_key(event):
            if event.key == "right" or event.key == "n":
                self.current_timestep = min(
                    self.current_timestep + 1, self.timesteps - 1
                )
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "left" or event.key == "p":
                self.current_timestep = max(self.current_timestep - 1, 0)
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "home":
                self.current_timestep = 0
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "end":
                self.current_timestep = self.timesteps - 1
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "v":
                # Toggle vector display
                self.show_vectors = not self.show_vectors
                if not self.show_vectors and self.quiver is not None:
                    self.quiver.remove()
                    self.quiver = None
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
                print(f"Force vectors: {'ON' if self.show_vectors else 'OFF'}")
            elif event.key == "+" or event.key == "=":
                # Increase vector scale
                self.vector_scale *= 1.5
                if self.show_vectors:
                    self.update_plot(self.current_timestep)
                    self.fig.canvas.draw()
                print(f"Vector scale: {self.vector_scale:.2f}")
            elif event.key == "-":
                # Decrease vector scale
                self.vector_scale /= 1.5
                if self.show_vectors:
                    self.update_plot(self.current_timestep)
                    self.fig.canvas.draw()
                print(f"Vector scale: {self.vector_scale:.2f}")
            elif event.key == "1":
                # Show all vertices
                self.set_vertex_mask("all")
                self.fig.canvas.draw()
            elif event.key == "2":
                # Show +x vertices
                self.set_vertex_mask("+x")
                self.fig.canvas.draw()
            elif event.key == "3":
                # Show -x vertices
                self.set_vertex_mask("-x")
                self.fig.canvas.draw()
            elif event.key == "4":
                # Show +y vertices
                self.set_vertex_mask("+y")
                self.fig.canvas.draw()
            elif event.key == "5":
                # Show -y vertices
                self.set_vertex_mask("-y")
                self.fig.canvas.draw()
            elif event.key == "6":
                # Show +z vertices
                self.set_vertex_mask("+z")
                self.fig.canvas.draw()
            elif event.key == "7":
                # Show -z vertices
                self.set_vertex_mask("-z")
                self.fig.canvas.draw()

        self.fig.canvas.mpl_connect("key_press_event", on_key)

        # Add instructions
        instructions = """
        Interactive Controls:
        - Right Arrow / 'n': Next timestep
        - Left Arrow / 'p': Previous timestep
        - Home: Go to first timestep
        - End: Go to last timestep
        - 'v': Toggle force vectors on/off
        - '+' / '=': Increase vector scale
        - '-': Decrease vector scale
        
        Vertex Masking:
        - '1': Show all vertices
        - '2': Show +x vertices only
        - '3': Show -x vertices only
        - '4': Show +y vertices only
        - '5': Show -y vertices only
        - '6': Show +z vertices only
        - '7': Show -z vertices only
        """
        print(instructions)

        plt.show()


def print_data_stats(vertices, forces):
    """Print statistics about the loaded data."""
    force_magnitudes = calculate_force_magnitudes(forces)

    print("\n" + "=" * 50)
    print("DATA STATISTICS")
    print("=" * 50)
    print(f"Number of vertices: {vertices.shape[0]}")
    print(f"Number of timesteps: {forces.shape[0]}")
    print(f"Vertex positions range:")
    print(f"  X: [{vertices[:, 0].min():.3f}, {vertices[:, 0].max():.3f}]")
    print(f"  Y: [{vertices[:, 1].min():.3f}, {vertices[:, 1].max():.3f}]")
    print(f"  Z: [{vertices[:, 2].min():.3f}, {vertices[:, 2].max():.3f}]")
    print(
        f"Force magnitude range: [{force_magnitudes.min():.3f}, {force_magnitudes.max():.3f}]"
    )
    print(f"Average force magnitude: {force_magnitudes.mean():.3f}")
    print("=" * 50)


def main():
    """Main function to run the force visualization."""
    print("Loading force visualization data...")

    # Load data
    vertices, forces = load_data()

    # Print statistics
    print_data_stats(vertices, forces)

    # Ask user for vertex masking options
    print("\nVertex Masking Options:")
    print("1. all (default) - Show all vertices")
    print("2. +x - Show vertices with x >= 0")
    print("3. -x - Show vertices with x < 0")
    print("4. +y - Show vertices with y >= 0")
    print("5. -y - Show vertices with y < 0")
    print("6. +z - Show vertices with z >= 0")
    print("7. -z - Show vertices with z < 0")
    
    mask_input = input("Show vertices only in (1-7, default=all): ").strip()
    mask_options = {
        "1": "all", "2": "+x", "3": "-x", "4": "+y", 
        "5": "-y", "6": "+z", "7": "-z", "": "all"
    }
    vertex_mask = mask_options.get(mask_input, "all")

    # Ask user for vector visualization options
    print("\nVector Visualization Options:")
    show_vectors_input = input("Show force vectors? (y/n): ").strip().lower()
    show_vectors = show_vectors_input in ['y', 'yes', '1', 'true']
    
    vector_scale = 0.01
    if show_vectors:
        try:
            scale_input = input(f"Enter vector scale factor (default {vector_scale}): ").strip()
            if scale_input:
                vector_scale = float(scale_input)
        except ValueError:
            print("Invalid scale factor, using default 1.0")

    # Create visualizer
    visualizer = ForceVisualizer(vertices, forces, show_vectors=show_vectors, 
                               vector_scale=vector_scale, vertex_mask=vertex_mask)

    # Ask user for visualization mode
    print("\nVisualization Options:")
    print("1. Interactive plot (use arrow keys to navigate)")
    print("2. Animated plot")
    print("3. Save animation as GIF")

    try:
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            visualizer.show_interactive()
        elif choice == "2":
            print("Creating animation...")
            ani = visualizer.create_animation(interval=200)
            plt.show()
        elif choice == "3":
            save_path = input("Enter save path (e.g., 'force_animation.gif'): ").strip()
            if not save_path:
                save_path = "force_animation.gif"
            ani = visualizer.create_animation(interval=200, save_path=save_path)
            plt.show()
        else:
            print("Invalid choice. Showing interactive plot by default.")
            visualizer.show_interactive()

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()
