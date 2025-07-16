import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
import argparse


def load_grid_data(file_path):
    """Load grid force data from CSV or numpy files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading grid data from {file_path}")
    
    if file_path.endswith('.csv'):
        # Load CSV data - you may need to adjust this based on your CSV format
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            # Extract grid data columns (adjust column names as needed)
            grid_data = df.values  # This will need to be reshaped based on your data format
            print(f"Loaded CSV data with shape: {grid_data.shape}")
            # You'll need to reshape this to [n_timesteps, x, y, z] format
        except ImportError:
            print("pandas not available, trying to load CSV with numpy...")
            grid_data = np.loadtxt(file_path, delimiter=',')
            print(f"Loaded CSV data with shape: {grid_data.shape}")
        
    elif file_path.endswith('.npy') or file_path.endswith('.npz'):
        if file_path.endswith('.npz'):
            npz_data = np.load(file_path)
            # Assume the first array in the npz file contains the grid data
            key = list(npz_data.keys())[0]
            grid_data = npz_data[key]
        else:
            grid_data = np.load(file_path)
        print(f"Loaded numpy data with shape: {grid_data.shape}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .csv, .npy, or .npz files.")
    
    # Handle different input formats
    if len(grid_data.shape) == 6:
        # Format: [n_timesteps, n_env, x, y, z, forcevector] - take first environment
        print(f"6D data detected: (n_timesteps, n_env, x, y, z, forcevector)")
        grid_data = grid_data[:, 0, :, :, :, :]  # Take first environment
        print(f"After selecting first environment: {grid_data.shape}")
    elif len(grid_data.shape) == 5:
        # Could be [n_timesteps, x, y, z, forcevector] or [n_timesteps, n_env, x, y, z]
        if grid_data.shape[-1] == 3:
            # [n_timesteps, x, y, z, forcevector] format (single environment)
            print(f"5D data detected: (n_timesteps, x, y, z, forcevector)")
        else:
            # [n_timesteps, n_env, x, y, z] format - take first environment
            print(f"5D data detected: (n_timesteps, n_env, x, y, z)")
            grid_data = grid_data[:, 0, :, :, :]
    elif len(grid_data.shape) == 4:
        # [n_timesteps, x, y, z] format (single environment, scalar forces)
        print(f"4D data detected: (n_timesteps, x, y, z)")
    elif len(grid_data.shape) == 2:
        # CSV data might need reshaping - this is a placeholder
        print("Warning: 2D data detected. You may need to reshape this data manually.")
        # For now, create a simple reshape as an example
        # You'll need to adjust this based on your actual CSV format
        if grid_data.shape[1] >= 32:  # Assuming 4x4x2 = 32 columns
            n_timesteps = grid_data.shape[0]
            grid_data = grid_data.reshape(n_timesteps, 4, 4, 2)
        else:
            raise ValueError(f"Cannot automatically reshape CSV data with shape {grid_data.shape}")
    else:
        raise ValueError(f"Unexpected grid data shape: {grid_data.shape}")
    
    print(f"Final grid data shape: {grid_data.shape}")
    return grid_data


def create_grid_coordinates(x_size, y_size, z_size):
    """Create 3D grid coordinates where each cell is 1 unit apart."""
    x_coords = np.arange(x_size)
    y_coords = np.arange(y_size)
    z_coords = np.arange(z_size)
    
    # Create meshgrid for all combinations
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Flatten to get coordinate arrays
    grid_positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return grid_positions, (X, Y, Z)


class GridForceVisualizer:
    def __init__(self, grid_data, show_vectors=False, vector_scale=1.0, layer_mask="all"):
        self.grid_data = grid_data  # Shape: [n_timesteps, x, y, z] or [n_timesteps, x, y, z, 3]
        
        # Check if we have force vectors (6th dimension) or scalar forces
        if len(grid_data.shape) == 5:
            self.has_force_vectors = True
            self.timesteps, self.x_size, self.y_size, self.z_size, self.force_dims = grid_data.shape
            print(f"Data has 3D force vectors: {self.force_dims} components")
        else:
            self.has_force_vectors = False
            self.timesteps, self.x_size, self.y_size, self.z_size = grid_data.shape
            print(f"Data has scalar forces")
            
        self.current_timestep = 0
        self.show_vectors = show_vectors
        self.vector_scale = vector_scale
        self.layer_mask = layer_mask
        
        # Create grid coordinates
        self.grid_positions, self.grid_meshes = create_grid_coordinates(
            self.x_size, self.y_size, self.z_size
        )
        
        # Calculate mask indices
        self.mask_indices = self._calculate_mask_indices()
        
        # Set up the figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        
        # Store quiver plot for vector updates
        self.quiver = None
        
        # Initialize the scatter plot
        self.setup_plot()

    def _calculate_mask_indices(self):
        """Calculate which grid cells to show based on the current mask."""
        if self.layer_mask == "all":
            return np.arange(len(self.grid_positions))
        elif self.layer_mask == "z0":
            return np.where(self.grid_positions[:, 2] == 0)[0]
        elif self.layer_mask == "z1":
            return np.where(self.grid_positions[:, 2] == 1)[0]
        elif self.layer_mask == "x_half":
            return np.where(self.grid_positions[:, 0] >= self.x_size // 2)[0]
        elif self.layer_mask == "y_half":
            return np.where(self.grid_positions[:, 1] >= self.y_size // 2)[0]
        else:
            return np.arange(len(self.grid_positions))

    def set_layer_mask(self, mask_type):
        """Set the layer mask and update the plot."""
        self.layer_mask = mask_type
        self.mask_indices = self._calculate_mask_indices()
        print(f"Showing {len(self.mask_indices)} grid cells (mask: {mask_type})")
        self.setup_plot()
        self.update_plot(self.current_timestep)

    def setup_plot(self):
        """Set up the initial 3D plot."""
        # Clear the axes
        self.ax.clear()
        
        # Get masked positions and forces
        masked_positions = self.grid_positions[self.mask_indices]
        
        # Get force data for current timestep and apply mask
        if self.has_force_vectors:
            # Calculate force magnitudes from 3D vectors
            force_magnitudes = np.linalg.norm(self.grid_data[0], axis=-1)
            flattened_forces = force_magnitudes.ravel()
        else:
            # Use scalar forces directly
            flattened_forces = self.grid_data[0].ravel()
            
        masked_forces = flattened_forces[self.mask_indices]
        
        # Get the range of force magnitudes for consistent colormap
        if self.has_force_vectors:
            all_magnitudes = np.linalg.norm(self.grid_data, axis=-1)
            self.vmin = np.min(all_magnitudes)
            self.vmax = np.max(all_magnitudes)
        else:
            self.vmin = np.min(self.grid_data)
            self.vmax = np.max(self.grid_data)
        
        # Create grid boxes and store them for updates
        self.create_grid_boxes(masked_positions, masked_forces)
        
        # Create a dummy scatter plot for the colorbar
        self.scatter = self.ax.scatter(
            masked_positions[:, 0] + 0.5,  # Center points in grid cells for colorbar
            masked_positions[:, 1] + 0.5, 
            masked_positions[:, 2] + 0.5,
            c=masked_forces,
            cmap="viridis",
            vmin=self.vmin,
            vmax=self.vmax,
            s=0,  # Make points invisible
            alpha=0
        )
        
        # Add colorbar (only once) - remove old colorbar first if it exists
        if hasattr(self, 'cbar') and self.cbar is not None:
            try:
                self.cbar.remove()
            except:
                pass
        self.cbar = self.fig.colorbar(self.scatter, ax=self.ax, shrink=0.5, aspect=20)
        self.cbar.set_label("Force Magnitude", rotation=270, labelpad=15)
        
        # Set labels and title
        self.ax.set_xlabel("X Grid Position")
        self.ax.set_ylabel("Y Grid Position") 
        self.ax.set_zlabel("Z Grid Position")
        self.ax.set_title(
            f"Grid Force Visualization - Timestep: {self.current_timestep}/{self.timesteps-1} (Mask: {self.layer_mask})"
        )
        
        # Add force vectors if enabled
        if self.show_vectors:
            self.add_force_vectors(self.current_timestep)
        
        # Set grid-based axis limits
        self.ax.set_xlim(0, self.x_size)
        self.ax.set_ylim(0, self.y_size)
        self.ax.set_zlim(0, self.z_size)
        
        # Set axis ticks to integer values only
        self.ax.set_xticks(range(0, self.x_size + 1))
        self.ax.set_yticks(range(0, self.y_size + 1))
        self.ax.set_zticks(range(0, self.z_size + 1))
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([self.x_size, self.y_size, self.z_size])

    def create_grid_boxes(self, masked_positions, masked_forces):
        """Create the 3D grid boxes with colors based on force magnitudes."""
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        cmap = plt.cm.viridis
        
        # Collect all box faces and colors
        all_faces = []
        all_colors = []
        
        for i, pos in enumerate(masked_positions):
            x, y, z = pos
            force_val = masked_forces[i]
            color = cmap(norm(force_val))
            
            # Create a cube from grid position to position+1 (not centered)
            # Grid cell (0,0,0) goes from (0,0,0) to (1,1,1)
            x_min, x_max = x, x + 1
            y_min, y_max = y, y + 1
            z_min, z_max = z, z + 1
            
            # Define the 8 vertices of a cube from (x,y,z) to (x+1,y+1,z+1)
            vertices = np.array([
                [x_min, y_min, z_min],  # 0
                [x_max, y_min, z_min],  # 1
                [x_max, y_max, z_min],  # 2
                [x_min, y_max, z_min],  # 3
                [x_min, y_min, z_max],  # 4
                [x_max, y_min, z_max],  # 5
                [x_max, y_max, z_max],  # 6
                [x_min, y_max, z_max],  # 7
            ])
            
            # Define the 6 quadrilateral faces of the cube
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[7], vertices[6], vertices[5]],  # top
                [vertices[0], vertices[4], vertices[5], vertices[1]],  # front
                [vertices[2], vertices[6], vertices[7], vertices[3]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[5], vertices[6], vertices[2]],  # right
            ]
            
            # Add faces and colors to collections
            for face in faces:
                all_faces.append(face)
                all_colors.append(color)
        
        # Store face structure for updates
        self.all_faces = all_faces
        self.faces_per_box = 6  # Each box has 6 faces
        
        # Create Poly3DCollection for all faces at once
        self.poly3d = Poly3DCollection(all_faces, alpha=0.6, facecolors=all_colors, 
                                      edgecolors='black', linewidths=0.5)
        self.ax.add_collection3d(self.poly3d)

    def add_force_vectors(self, timestep):
        """Add force vectors to the plot for a given timestep."""
        # Remove existing quiver plot if it exists
        if self.quiver is not None:
            self.quiver.remove()
        
        # Get masked positions for this timestep
        masked_positions = self.grid_positions[self.mask_indices]
        
        if self.has_force_vectors:
            # Use actual 3D force vectors
            force_vectors_3d = self.grid_data[timestep]  # Shape: (x, y, z, 3)
            
            # Flatten to get all force vectors and apply mask
            flattened_vectors = force_vectors_3d.reshape(-1, 3)  # Shape: (x*y*z, 3)
            masked_vectors = flattened_vectors[self.mask_indices]
            
            # Scale the vectors
            scaled_vectors = masked_vectors * self.vector_scale
            
            # Calculate magnitudes for coloring
            force_magnitudes = np.linalg.norm(masked_vectors, axis=1)
        else:
            # Create vertical force vectors from scalar forces
            flattened_forces = self.grid_data[timestep].ravel()
            masked_forces = flattened_forces[self.mask_indices]
            
            # Create vertical vectors (force magnitude in Z direction)
            scaled_vectors = np.zeros((len(masked_forces), 3))
            scaled_vectors[:, 2] = masked_forces * self.vector_scale  # Z-direction vectors
            
            # Use scalar forces for coloring
            force_magnitudes = masked_forces
        
        # Position vectors at center of grid cells
        vector_positions = masked_positions + 0.5
        
        # Create colormap for vectors based on magnitude
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        colors = plt.cm.viridis(norm(force_magnitudes))
        
        # Add quiver plot for force vectors
        self.quiver = self.ax.quiver(
            vector_positions[:, 0], vector_positions[:, 1], vector_positions[:, 2],
            scaled_vectors[:, 0], scaled_vectors[:, 1], scaled_vectors[:, 2],
            colors=colors,
            alpha=0.7,
            arrow_length_ratio=0.1,
            linewidth=2
        )

    def update_plot(self, timestep):
        """Update the plot for a given timestep - only update colors, don't recreate everything."""
        self.current_timestep = timestep
        
        # Get masked positions and forces for this timestep
        masked_positions = self.grid_positions[self.mask_indices]
        
        if self.has_force_vectors:
            # Calculate force magnitudes from 3D vectors
            force_magnitudes = np.linalg.norm(self.grid_data[timestep], axis=-1)
            flattened_forces = force_magnitudes.ravel()
        else:
            # Use scalar forces directly
            flattened_forces = self.grid_data[timestep].ravel()
            
        masked_forces = flattened_forces[self.mask_indices]
        
        # Update box colors without recreating the geometry
        norm = Normalize(vmin=self.vmin, vmax=self.vmax)
        cmap = plt.cm.viridis
        
        # Calculate new colors for each box
        new_colors = []
        for i, force_val in enumerate(masked_forces):
            color = cmap(norm(force_val))
            # Each box has 6 faces, so repeat the color 6 times
            for _ in range(self.faces_per_box):
                new_colors.append(color)
        
        # Update the Poly3DCollection colors
        self.poly3d.set_facecolors(new_colors)
        
        # Update scatter plot data for colorbar
        self.scatter.set_array(masked_forces)
        
        # Update force vectors if enabled
        if self.show_vectors:
            self.add_force_vectors(timestep)
        
        # Update title
        self.ax.set_title(
            f"Grid Force Visualization - Timestep: {timestep}/{self.timesteps-1} (Mask: {self.layer_mask})"
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
            if event.key == "right" or event.key == "a":
                self.current_timestep = min(
                    self.current_timestep + 1, self.timesteps - 1
                )
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "left" or event.key == "d":
                self.current_timestep = max(self.current_timestep - 1, 0)
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "w":
                self.current_timestep = 0
                self.update_plot(self.current_timestep)
                self.fig.canvas.draw()
            elif event.key == "s":
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
                # Show all grid cells
                self.set_layer_mask("all")
                self.fig.canvas.draw()
            elif event.key == "2":
                # Show bottom layer (z=0)
                self.set_layer_mask("z0")
                self.fig.canvas.draw()
            elif event.key == "3":
                # Show top layer (z=1)
                self.set_layer_mask("z1")
                self.fig.canvas.draw()
            elif event.key == "4":
                # Show x half
                self.set_layer_mask("x_half")
                self.fig.canvas.draw()
            elif event.key == "5":
                # Show y half
                self.set_layer_mask("y_half")
                self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect("key_press_event", on_key)
        
        # Add instructions
        instructions = """
        Interactive Controls:
        - Right Arrow / 'a': Next timestep
        - Left Arrow / 'd': Previous timestep
        - 'w': Go to first timestep
        - 's': Go to last timestep
        - 'v': Toggle force vectors on/off
        - '+' / '=': Increase vector scale
        - '-': Decrease vector scale
        
        Layer Masking:
        - '1': Show all grid cells
        - '2': Show bottom layer (z=0)
        - '3': Show top layer (z=1)
        - '4': Show x half
        - '5': Show y half
        """
        print(instructions)
        
        plt.show()


def print_grid_stats(grid_data):
    """Print statistics about the loaded grid data."""
    print("\n" + "=" * 50)
    print("GRID DATA STATISTICS")
    print("=" * 50)
    
    if len(grid_data.shape) == 5:
        # Has force vectors
        print(f"Grid shape: {grid_data.shape[1:4]}")
        print(f"Force vector dimensions: {grid_data.shape[4]}")
        print(f"Number of timesteps: {grid_data.shape[0]}")
        
        # Calculate force magnitudes for statistics
        force_magnitudes = np.linalg.norm(grid_data, axis=-1)
        print(f"Force magnitude range: [{force_magnitudes.min():.3f}, {force_magnitudes.max():.3f}]")
        print(f"Average force magnitude: {force_magnitudes.mean():.3f}")
        print(f"Non-zero cells per timestep: {np.count_nonzero(force_magnitudes, axis=(1,2,3)).mean():.1f}")
        
        # Statistics for individual force components
        print(f"Force X range: [{grid_data[..., 0].min():.3f}, {grid_data[..., 0].max():.3f}]")
        print(f"Force Y range: [{grid_data[..., 1].min():.3f}, {grid_data[..., 1].max():.3f}]")
        print(f"Force Z range: [{grid_data[..., 2].min():.3f}, {grid_data[..., 2].max():.3f}]")
    else:
        # Scalar forces
        print(f"Grid shape: {grid_data.shape[1:]} (x, y, z)")
        print(f"Number of timesteps: {grid_data.shape[0]}")
        print(f"Total grid cells: {np.prod(grid_data.shape[1:])}")
        print(f"Force range: [{grid_data.min():.3f}, {grid_data.max():.3f}]")
        print(f"Average force: {grid_data.mean():.3f}")
        print(f"Non-zero cells per timestep: {np.count_nonzero(grid_data, axis=(1,2,3)).mean():.1f}")
    
    print("=" * 50)


def main():
    """Main function to run the grid force visualization."""
    parser = argparse.ArgumentParser(description="Visualize grid sensor force data in 3D")
    parser.add_argument("path", help="Path to data file (.csv, .npy, or .npz)")
    parser.add_argument("--interactive", action="store_true", 
                       help="Show interactive plot (use arrow keys to navigate)")
    parser.add_argument("--animation", action="store_true", 
                       help="Show animated plot")
    parser.add_argument("--save", type=str, metavar="FILEPATH",
                       help="Save animation as GIF to specified filepath")
    parser.add_argument("--layer-mask", choices=["all", "z0", "z1", "x_half", "y_half"],
                       default="all", help="Layer masking option (default: all)")
    parser.add_argument("--show-vectors", action="store_true",
                       help="Show force vectors")
    parser.add_argument("--vector-scale", type=float, default=0.1,
                       help="Scale factor for force vectors (default: 0.1)")
    parser.add_argument("--interval", type=int, default=200,
                       help="Animation interval in milliseconds (default: 200)")
    
    args = parser.parse_args()
    
    print("Loading grid force visualization data...")
    
    # Load data
    try:
        grid_data = load_grid_data(args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Print statistics
    print_grid_stats(grid_data)
    
    # Create visualizer
    visualizer = GridForceVisualizer(
        grid_data, 
        show_vectors=args.show_vectors,
        vector_scale=args.vector_scale, 
        layer_mask=args.layer_mask
    )
    
    try:
        if args.save:
            # Save animation
            print(f"Creating and saving animation to {args.save}...")
            ani = visualizer.create_animation(interval=args.interval, save_path=args.save)
            # Don't show plot when saving
            print("Animation saved successfully!")
        elif args.animation:
            # Show animation
            print("Creating animation...")
            ani = visualizer.create_animation(interval=args.interval)
            plt.show()
        else:
            # Default to interactive (or if --interactive is specified)
            print("Starting interactive visualization...")
            visualizer.show_interactive()
            
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    except Exception as e:
        print(f"Error during visualization: {e}")


if __name__ == "__main__":
    main()