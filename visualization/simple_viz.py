#!/usr/bin/env python3
"""
Simple visualization script for CFD VTK files
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

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
        elif line.startswith('LOOKUP_TABLE default'):
            data_start = i + 1
            break

    if not all([dimensions, origin, spacing, data_start]):
        raise ValueError(f"Invalid VTK file format: {filename}")

    nx, ny = dimensions[0], dimensions[1]

    # Create coordinate arrays
    x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)
    X, Y = np.meshgrid(x, y)

    # Read data
    data = []
    for i in range(data_start, data_start + nx * ny):
        if i < len(lines):
            try:
                value = float(lines[i].strip())
                data.append(value)
            except ValueError:
                data.append(0.0)
        else:
            data.append(0.0)

    # Reshape to grid
    data = np.array(data).reshape(ny, nx)

    return X, Y, data

def visualize_vtk_files():
    """Create visualizations from VTK files"""

    # Look for VTK files
    vtk_files = glob.glob("output/minimal_step_*.vtk")

    if not vtk_files:
        print("No minimal_step VTK files found in output/")
        return

    vtk_files.sort()
    print(f"Found {len(vtk_files)} VTK files")

    # Create visualization output directory
    os.makedirs("visualization/visualization_output", exist_ok=True)

    # Process each file
    for i, filename in enumerate(vtk_files):
        print(f"Processing {filename}...")

        try:
            X, Y, data = read_vtk_file(filename)

            # Create figure
            plt.figure(figsize=(10, 6))

            # Create contour plot
            contour = plt.contourf(X, Y, data, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Velocity Magnitude')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'CFD Simulation - {os.path.basename(filename)}')
            plt.grid(True, alpha=0.3)

            # Save plot
            output_filename = f"visualization/visualization_output/plot_{i:02d}.png"
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_filename}")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print("\nVisualization complete!")
    print("Check visualization/visualization_output/ for PNG files")

if __name__ == "__main__":
    visualize_vtk_files()