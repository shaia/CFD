#!/usr/bin/env python3
"""
Enhanced CFD Visualization Script - Simplified and Robust
Creates complex visualizations from VTK output files
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
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

def create_comprehensive_animation(vtk_files, save_animation=True):
    """Create a comprehensive 2x3 grid animation"""

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
                })

                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    if not frames_data:
        print("No valid data found!")
        return

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CFD Flow Analysis Dashboard', fontsize=16, fontweight='bold')

    # Calculate global min/max for consistent scaling
    vel_min = min(np.min(frame['velocity_mag']) for frame in frames_data)
    vel_max = max(np.max(frame['velocity_mag']) for frame in frames_data)
    p_min = min(np.min(frame['p']) for frame in frames_data)
    p_max = max(np.max(frame['p']) for frame in frames_data)
    vort_min = min(np.min(frame['vorticity']) for frame in frames_data)
    vort_max = max(np.max(frame['vorticity']) for frame in frames_data)

    # Custom colormap for velocity
    velocity_cmap = LinearSegmentedColormap.from_list(
        'velocity', ['#000080', '#0080FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
    )

    # Initialize static elements
    axes[0,0].set_title('Velocity Magnitude', fontweight='bold')
    axes[0,1].set_title('Pressure Field', fontweight='bold')
    axes[0,2].set_title('Vorticity', fontweight='bold')
    axes[1,0].set_title('Vector Field', fontweight='bold')
    axes[1,1].set_title('Streamlines', fontweight='bold')
    axes[1,2].set_title('Flow Analysis', fontweight='bold')

    def animate(frame):
        # Clear all axes
        for ax in axes.flat:
            ax.clear()

        data = frames_data[frame]

        # 1. Velocity Magnitude
        im1 = axes[0,0].imshow(data['velocity_mag'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                              origin='lower', aspect='auto', vmin=vel_min, vmax=vel_max, cmap=velocity_cmap)
        axes[0,0].set_title('Velocity Magnitude', fontweight='bold')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')

        # 2. Pressure Field with contours
        im2 = axes[0,1].imshow(data['p'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                              origin='lower', aspect='auto', vmin=p_min, vmax=p_max, cmap='RdBu_r')
        contours = axes[0,1].contour(X, Y, data['p'], levels=8, colors='black', linewidths=0.5)
        axes[0,1].clabel(contours, inline=True, fontsize=8)
        axes[0,1].set_title('Pressure Field', fontweight='bold')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Y')

        # 3. Vorticity
        im3 = axes[0,2].imshow(data['vorticity'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                              origin='lower', aspect='auto', vmin=vort_min, vmax=vort_max, cmap='seismic')
        axes[0,2].set_title('Vorticity', fontweight='bold')
        axes[0,2].set_xlabel('X')
        axes[0,2].set_ylabel('Y')

        # 4. Vector Field (subsampled)
        skip = 5
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        U_sub = data['u'][::skip, ::skip]
        V_sub = data['v'][::skip, ::skip]
        vel_sub = data['velocity_mag'][::skip, ::skip]

        q = axes[1,0].quiver(X_sub, Y_sub, U_sub, V_sub, vel_sub, cmap=velocity_cmap,
                            scale=15, width=0.003)
        axes[1,0].set_xlim(X.min(), X.max())
        axes[1,0].set_ylim(Y.min(), Y.max())
        axes[1,0].set_title('Vector Field', fontweight='bold')
        axes[1,0].set_xlabel('X')
        axes[1,0].set_ylabel('Y')

        # 5. Streamlines
        axes[1,1].streamplot(X, Y, data['u'], data['v'], color=data['velocity_mag'],
                            cmap=velocity_cmap, density=2, linewidth=1.5)
        axes[1,1].set_xlim(X.min(), X.max())
        axes[1,1].set_ylim(Y.min(), Y.max())
        axes[1,1].set_title('Streamlines', fontweight='bold')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Y')

        # 6. Combined Analysis
        # Background: velocity magnitude
        axes[1,2].imshow(data['velocity_mag'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', vmin=vel_min, vmax=vel_max,
                        cmap=velocity_cmap, alpha=0.8)

        # Overlay: pressure contours
        pressure_contours = axes[1,2].contour(X, Y, data['p'], levels=6, colors='white', linewidths=1.5)

        # Highlight vorticity regions
        vort_threshold = np.percentile(np.abs(data['vorticity']), 85)
        high_vort_mask = np.abs(data['vorticity']) > vort_threshold
        axes[1,2].contour(X, Y, high_vort_mask, levels=[0.5], colors='red', linewidths=2)

        axes[1,2].set_xlim(X.min(), X.max())
        axes[1,2].set_ylim(Y.min(), Y.max())
        axes[1,2].set_title('Combined Analysis', fontweight='bold')
        axes[1,2].set_xlabel('X')
        axes[1,2].set_ylabel('Y')

        # Add iteration info
        fig.suptitle(f'CFD Flow Analysis - Iteration: {times[frame]}', fontsize=16, fontweight='bold')

        plt.tight_layout()

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                  interval=600, repeat=True)

    # Save animation
    if save_animation:
        output_file = 'visualization/visualization_output/cfd_comprehensive_analysis.gif'
        print(f"Saving comprehensive animation to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=2)
            print(f"Animation saved as {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.show()
    return anim

def create_vorticity_analysis(vtk_files, save_animation=True):
    """Create focused vorticity analysis animation"""

    frames_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if 'u' in data_fields and 'v' in data_fields:
                # Calculate vorticity (curl of velocity)
                vorticity = np.gradient(data_fields['v'], axis=1) - np.gradient(data_fields['u'], axis=0)
                velocity_mag = np.sqrt(data_fields['u']**2 + data_fields['v']**2)

                frames_data.append({
                    'u': data_fields['u'],
                    'v': data_fields['v'],
                    'vorticity': vorticity,
                    'velocity_mag': velocity_mag
                })

                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)

        except Exception as e:
            continue

    if not frames_data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    vort_min = min(np.min(frame['vorticity']) for frame in frames_data)
    vort_max = max(np.max(frame['vorticity']) for frame in frames_data)

    def animate(frame):
        ax1.clear()
        ax2.clear()

        data = frames_data[frame]

        # Left: Vorticity field
        im1 = ax1.imshow(data['vorticity'], extent=[X.min(), X.max(), Y.min(), Y.max()],
                        origin='lower', aspect='auto', vmin=vort_min, vmax=vort_max, cmap='RdBu')
        ax1.set_title('Vorticity Field', fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Right: Streamlines colored by vorticity
        speed = data['velocity_mag']
        lw = 5 * speed / speed.max()  # Line width varies with speed

        ax2.streamplot(X, Y, data['u'], data['v'], color=data['vorticity'], linewidth=lw,
                      cmap='RdBu', density=2)
        ax2.set_title('Streamlines Colored by Vorticity', fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(Y.min(), Y.max())

        fig.suptitle(f'Vorticity Analysis - Iteration: {times[frame]}', fontsize=14, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames=len(frames_data),
                                  interval=400, repeat=True)

    if save_animation:
        output_file = 'visualization/visualization_output/cfd_vorticity_analysis.gif'
        print(f"Saving vorticity analysis to {output_file}...")
        try:
            anim.save(output_file, writer='pillow', fps=3)
            print(f"Vorticity analysis saved as {output_file}")
        except Exception as e:
            print(f"Error saving animation: {e}")

    plt.show()
    return anim

def create_statistical_analysis(vtk_files):
    """Create statistical analysis plots"""

    # Collect statistics over time
    times = []
    max_velocities = []
    avg_velocities = []
    max_vorticity = []
    pressure_range = []

    for filename in sorted(vtk_files):
        try:
            X, Y, data_fields = read_vtk_file(filename)
            if 'u' in data_fields and 'v' in data_fields and 'p' in data_fields:
                velocity_mag = np.sqrt(data_fields['u']**2 + data_fields['v']**2)
                vorticity = np.gradient(data_fields['v'], axis=1) - np.gradient(data_fields['u'], axis=0)

                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)

                max_velocities.append(np.max(velocity_mag))
                avg_velocities.append(np.mean(velocity_mag))
                max_vorticity.append(np.max(np.abs(vorticity)))
                pressure_range.append(np.max(data_fields['p']) - np.min(data_fields['p']))

        except Exception as e:
            continue

    if not times:
        return

    # Create statistical plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Velocity statistics
    axes[0,0].plot(times, max_velocities, 'r-o', label='Max Velocity', linewidth=2)
    axes[0,0].plot(times, avg_velocities, 'b-s', label='Avg Velocity', linewidth=2)
    axes[0,0].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Velocity')
    axes[0,0].set_title('Velocity Statistics', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Vorticity evolution
    axes[0,1].plot(times, max_vorticity, 'g-^', label='Max |Vorticity|', linewidth=2)
    axes[0,1].set_xlabel('Iteration')
    axes[0,1].set_ylabel('Vorticity')
    axes[0,1].set_title('Vorticity Evolution', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Pressure range
    axes[1,0].plot(times, pressure_range, 'm-d', label='Pressure Range', linewidth=2)
    axes[1,0].set_xlabel('Iteration')
    axes[1,0].set_ylabel('Pressure Range')
    axes[1,0].set_title('Pressure Variation', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Flow energy (approximation)
    flow_energy = [0.5 * avg_vel**2 for avg_vel in avg_velocities]
    axes[1,1].plot(times, flow_energy, 'c-p', label='Kinetic Energy Density', linewidth=2)
    axes[1,1].set_xlabel('Iteration')
    axes[1,1].set_ylabel('Energy')
    axes[1,1].set_title('Flow Energy Evolution', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('CFD Statistical Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('visualization/visualization_output/cfd_statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Statistical analysis saved as 'visualization/visualization_output/cfd_statistical_analysis.png'")

def main():
    """Main function to create enhanced CFD visualizations"""

    # Create output directory if it doesn't exist
    os.makedirs("visualization/visualization_output", exist_ok=True)

    # Find all VTK files
    vtk_pattern = "output/output_optimized_*.vtk"
    vtk_files = glob.glob(vtk_pattern)

    if not vtk_files:
        print(f"No VTK files found matching pattern: {vtk_pattern}")
        return

    print(f"Found {len(vtk_files)} VTK files")

    # Create enhanced visualizations
    print("\n1. Creating comprehensive flow analysis...")
    create_comprehensive_animation(vtk_files)

    print("\n2. Creating vorticity analysis...")
    create_vorticity_analysis(vtk_files)

    print("\n3. Creating statistical analysis...")
    create_statistical_analysis(vtk_files)

    print("\nAll enhanced visualizations complete!")

if __name__ == "__main__":
    main()