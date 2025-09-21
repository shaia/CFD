#!/usr/bin/env python3
"""
Advanced CFD Visualization Script
Creates complex and interactive visualizations from VTK output files
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import glob

def read_vtk_file(filename):
    """Read a VTK structured points file and extract data"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    dimensions = None
    origin = None
    spacing = None
    data_start = None

    for i, line in enumerate(lines):
        if line.startswith('DIMENSIONS'):
            dimensions = [int(x) for x in line.split()[1:4]]
        elif line.startswith('ORIGIN'):
            origin = [float(x) for x in line.split()[1:4]]
        elif line.startswith('SPACING'):
            spacing = [float(x) for x in line.split()[1:4]]
        elif line.startswith('POINT_DATA'):
            data_start = i + 1
            break

    if not all([dimensions, origin, spacing, data_start]):
        raise ValueError(f"Invalid VTK file format: {filename}")

    nx, ny = dimensions[0], dimensions[1]

    # Create coordinate arrays
    x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)
    X, Y = np.meshgrid(x, y)

    # Parse data fields
    data_fields = {}
    i = data_start

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('SCALARS'):
            field_name = line.split()[1]
            i += 2  # Skip LOOKUP_TABLE line

            # Read scalar data
            field_data = []
            while i < len(lines) and not lines[i].strip().startswith(('SCALARS', 'VECTORS')):
                values = lines[i].strip().split()
                field_data.extend([float(v) for v in values])
                i += 1

            # Reshape to grid
            field_data = np.array(field_data).reshape((ny, nx))
            data_fields[field_name] = field_data
        else:
            i += 1

    return X, Y, data_fields

def create_custom_colormap():
    """Create a custom colormap for better visualization"""
    colors = ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000']
    n_bins = 256
    return LinearSegmentedColormap.from_list('cfd_custom', colors, N=n_bins)

def create_multi_panel_animation(vtk_files, save_animation=True):
    """Create a comprehensive multi-panel animation"""

    # Read all files and extract data
    frames_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if 'u' in data_fields and 'v' in data_fields and 'p' in data_fields:
                velocity_mag = np.sqrt(data_fields['u']**2 + data_fields['v']**2)
                vorticity = np.gradient(data_fields['v'], axis=1) - np.gradient(data_fields['u'], axis=0)

                frames_data.append({
                    'u': data_fields['u'],
                    'v': data_fields['v'],
                    'p': data_fields['p'],
                    'velocity_mag': velocity_mag,
                    'vorticity': vorticity,
                    'T': data_fields.get('T', np.ones_like(velocity_mag) * 300)
                })

                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return

    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Velocity magnitude
    ax2 = fig.add_subplot(gs[0, 1])  # Pressure
    ax3 = fig.add_subplot(gs[0, 2])  # Vorticity
    ax4 = fig.add_subplot(gs[1, 0])  # Vector field
    ax5 = fig.add_subplot(gs[1, 1])  # Streamlines
    ax6 = fig.add_subplot(gs[1, 2])  # Temperature
    ax7 = fig.add_subplot(gs[2, :])  # Combined view with contours

    # Custom colormap
    custom_cmap = create_custom_colormap()

    # Calculate global min/max for consistent scaling
    vel_min = min(np.min(frame['velocity_mag']) for frame in frames_data)
    vel_max = max(np.max(frame['velocity_mag']) for frame in frames_data)
    p_min = min(np.min(frame['p']) for frame in frames_data)
    p_max = max(np.max(frame['p']) for frame in frames_data)
    vort_min = min(np.min(frame['vorticity']) for frame in frames_data)
    vort_max = max(np.max(frame['vorticity']) for frame in frames_data)

    # Initialize plots
    im1 = ax1.imshow(frames_data[0]['velocity_mag'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                     origin='lower', aspect='auto', vmin=vel_min, vmax=vel_max, cmap=custom_cmap)

    im2 = ax2.imshow(frames_data[0]['p'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                     origin='lower', aspect='auto', vmin=p_min, vmax=p_max, cmap='RdBu_r')

    im3 = ax3.imshow(frames_data[0]['vorticity'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                     origin='lower', aspect='auto', vmin=vort_min, vmax=vort_max, cmap='seismic')

    # Vector field (subsample for clarity)
    skip = 4
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]

    # Temperature with transparency
    im6 = ax6.imshow(frames_data[0]['T'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                     origin='lower', aspect='auto', cmap='plasma', alpha=0.8)

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Velocity Magnitude')

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Pressure')

    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Vorticity')

    cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8)
    cbar6.set_label('Temperature')

    # Set titles and labels
    ax1.set_title('Velocity Magnitude', fontsize=12, fontweight='bold')
    ax2.set_title('Pressure Field', fontsize=12, fontweight='bold')
    ax3.set_title('Vorticity', fontsize=12, fontweight='bold')
    ax4.set_title('Vector Field', fontsize=12, fontweight='bold')
    ax5.set_title('Streamlines', fontsize=12, fontweight='bold')
    ax6.set_title('Temperature', fontsize=12, fontweight='bold')
    ax7.set_title('Combined View with Contours', fontsize=14, fontweight='bold')

    # Add iteration counter
    time_text = fig.suptitle('', fontsize=16, fontweight='bold')

    def animate(frame):
        # Clear axes that need redrawing
        ax4.clear()
        ax5.clear()
        ax7.clear()

        data = frames_data[frame]

        # Update scalar fields
        im1.set_array(data['velocity_mag'])
        im2.set_array(data['p'])
        im3.set_array(data['vorticity'])
        im6.set_array(data['T'])

        # Vector field
        u_sub = data['u'][::skip, ::skip]
        v_sub = data['v'][::skip, ::skip]
        vel_sub = data['velocity_mag'][::skip, ::skip]

        ax4.quiver(X_sub, Y_sub, u_sub, v_sub, vel_sub, cmap=custom_cmap,
                   scale=10, width=0.002, alpha=0.8)
        ax4.set_xlim(X.min(), X.max())
        ax4.set_ylim(Y.min(), Y.max())
        ax4.set_aspect('equal')
        ax4.set_title('Vector Field', fontsize=12, fontweight='bold')

        # Streamlines with varying color
        ax5.streamplot(X, Y, data['u'], data['v'], color=data['velocity_mag'],
                       cmap=custom_cmap, density=2, linewidth=1.5, arrowsize=1.5)
        ax5.set_xlim(X.min(), X.max())
        ax5.set_ylim(Y.min(), Y.max())
        ax5.set_aspect('equal')
        ax5.set_title('Streamlines', fontsize=12, fontweight='bold')

        # Combined view with contours
        im7 = ax7.imshow(data['velocity_mag'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                         origin='lower', aspect='auto', vmin=vel_min, vmax=vel_max,
                         cmap=custom_cmap, alpha=0.7)

        # Add pressure contours
        contours = ax7.contour(X, Y, data['p'], levels=10, colors='white', linewidths=1, alpha=0.8)
        ax7.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

        # Add streamlines (remove alpha parameter for compatibility)
        ax7.streamplot(X, Y, data['u'], data['v'], color='black', density=1,
                       linewidth=0.8, arrowsize=1)

        # Highlight high vorticity regions
        vort_thresh = np.percentile(np.abs(data['vorticity']), 90)
        high_vort = np.abs(data['vorticity']) > vort_thresh
        ax7.contour(X, Y, high_vort, levels=[0.5], colors='red', linewidths=2)

        ax7.set_xlim(X.min(), X.max())
        ax7.set_ylim(Y.min(), Y.max())
        ax7.set_xlabel('X', fontsize=12)
        ax7.set_ylabel('Y', fontsize=12)
        ax7.set_title('Combined View: Velocity + Pressure Contours + Vorticity', fontsize=14, fontweight='bold')

        # Update iteration counter
        time_text.set_text(f'CFD Simulation - Iteration: {times[frame]} | Time Step: {frame+1}/{len(frames_data)}')

        return [im1, im2, im3, im6]

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                   interval=500, repeat=True)

    # Save animation
    if save_animation:
        output_file = 'visualization/visualization_output/cfd_advanced_animation.gif'
        print(f"Saving advanced animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=2, dpi=100)
        print(f"Advanced animation saved as {output_file}")

    plt.tight_layout()
    plt.show()

    return anim

def create_3d_surface_animation(vtk_files, field='velocity_mag', save_animation=True):
    """Create 3D surface animation"""
    from mpl_toolkits.mplot3d import Axes3D

    frames_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)

            if field == 'velocity_mag' and 'u' in data_fields and 'v' in data_fields:
                field_data = np.sqrt(data_fields['u']**2 + data_fields['v']**2)
            elif field in data_fields:
                field_data = data_fields[field]
            else:
                continue

            frames_data.append(field_data)
            iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
            times.append(iteration)

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return

    # Set up 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate global min/max
    vmin = min(np.min(frame) for frame in frames_data)
    vmax = max(np.max(frame) for frame in frames_data)

    def animate(frame):
        ax.clear()

        # Create 3D surface
        surf = ax.plot_surface(X, Y, frames_data[frame], cmap='viridis',
                              vmin=vmin, vmax=vmax, alpha=0.8, linewidth=0, antialiased=True)

        # Add contour lines at bottom
        ax.contour(X, Y, frames_data[frame], zdir='z', offset=vmin-0.1*(vmax-vmin),
                   cmap='viridis', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(field.replace('_', ' ').title())
        ax.set_title(f'3D Surface - {field.replace("_", " ").title()} - Iteration: {times[frame]}')

        # Set consistent limits
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_zlim(vmin, vmax)

        # Rotate view
        ax.view_init(elev=30, azim=frame*2)

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                   interval=200, repeat=True)

    if save_animation:
        output_file = f'visualization/visualization_output/cfd_3d_{field}.gif'
        print(f"Saving 3D animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=5)
        print(f"3D animation saved as {output_file}")

    plt.show()
    return anim

def create_particle_trace_animation(vtk_files, save_animation=True):
    """Create particle trace animation showing flow pathlines"""

    # Read all files
    frames_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if 'u' in data_fields and 'v' in data_fields:
                frames_data.append({
                    'u': data_fields['u'],
                    'v': data_fields['v'],
                    'X': X,
                    'Y': Y
                })
                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return

    # Initialize particles
    n_particles = 50
    particles_x = np.random.uniform(X.min(), X.min() + 0.1, n_particles)
    particles_y = np.random.uniform(Y.min(), Y.max(), n_particles)

    # Store particle histories
    particle_histories_x = [[] for _ in range(n_particles)]
    particle_histories_y = [[] for _ in range(n_particles)]

    fig, ax = plt.subplots(figsize=(15, 8))

    def animate(frame):
        ax.clear()

        data = frames_data[frame % len(frames_data)]

        # Background velocity magnitude
        velocity_mag = np.sqrt(data['u']**2 + data['v']**2)
        im = ax.imshow(velocity_mag, extent=[X.min(), X.max(), Y.min(), Y.max()],
                       origin='lower', aspect='auto', cmap='Blues', alpha=0.6)

        # Update particle positions
        dt = 0.01
        for i in range(n_particles):
            if 0 <= particles_x[i] < X.max() and 0 <= particles_y[i] < Y.max():
                # Interpolate velocity at particle position
                xi = int((particles_x[i] - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1))
                yi = int((particles_y[i] - Y.min()) / (Y.max() - Y.min()) * (Y.shape[0] - 1))

                xi = max(0, min(xi, data['u'].shape[1] - 1))
                yi = max(0, min(yi, data['u'].shape[0] - 1))

                u_interp = data['u'][yi, xi]
                v_interp = data['v'][yi, xi]

                # Update position
                particles_x[i] += u_interp * dt
                particles_y[i] += v_interp * dt

                # Store history
                particle_histories_x[i].append(particles_x[i])
                particle_histories_y[i].append(particles_y[i])

                # Limit history length
                if len(particle_histories_x[i]) > 20:
                    particle_histories_x[i] = particle_histories_x[i][-20:]
                    particle_histories_y[i] = particle_histories_y[i][-20:]

            # Reset particles that leave domain
            if particles_x[i] >= X.max() or particles_x[i] < X.min() or \
               particles_y[i] >= Y.max() or particles_y[i] < Y.min():
                particles_x[i] = X.min() + 0.01
                particles_y[i] = np.random.uniform(Y.min(), Y.max())
                particle_histories_x[i] = []
                particle_histories_y[i] = []

        # Draw particle traces
        for i in range(n_particles):
            if len(particle_histories_x[i]) > 1:
                # Color trace by age (newer = brighter)
                alphas = np.linspace(0.2, 1.0, len(particle_histories_x[i]))
                for j in range(len(particle_histories_x[i]) - 1):
                    ax.plot([particle_histories_x[i][j], particle_histories_x[i][j+1]],
                           [particle_histories_y[i][j], particle_histories_y[i][j+1]],
                           'r-', alpha=alphas[j], linewidth=2)

        # Draw current particle positions
        ax.scatter(particles_x, particles_y, c='red', s=30, alpha=0.8, edgecolors='white', linewidth=1)

        # Add streamlines for context
        ax.streamplot(data['X'], data['Y'], data['u'], data['v'],
                     color='gray', density=1, linewidth=0.5)

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Particle Traces in Flow Field - Iteration: {times[frame % len(times)]}')

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data)*3,
                                   interval=100, repeat=True)

    if save_animation:
        output_file = 'visualization/visualization_output/cfd_particle_traces.gif'
        print(f"Saving particle trace animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=10)
        print(f"Particle trace animation saved as {output_file}")

    plt.tight_layout()
    plt.show()
    return anim

def main():
    """Main function to create advanced CFD visualizations"""

    # Create output directory if it doesn't exist
    os.makedirs("visualization/visualization_output", exist_ok=True)

    # Find all VTK files
    vtk_pattern = "output/output_optimized_*.vtk"
    vtk_files = glob.glob(vtk_pattern)

    if not vtk_files:
        print(f"No VTK files found matching pattern: {vtk_pattern}")
        print("Make sure you have VTK files in the current directory!")
        return

    print(f"Found {len(vtk_files)} VTK files")

    # Create different advanced visualizations
    print("\n1. Creating multi-panel comprehensive animation...")
    create_multi_panel_animation(vtk_files)

    print("\n2. Creating 3D surface animation (velocity magnitude)...")
    create_3d_surface_animation(vtk_files, 'velocity_mag')

    print("\n3. Creating 3D surface animation (pressure)...")
    create_3d_surface_animation(vtk_files, 'p')

    print("\n4. Creating particle trace animation...")
    create_particle_trace_animation(vtk_files)

    print("\nAll advanced animations created!")

if __name__ == "__main__":
    main()