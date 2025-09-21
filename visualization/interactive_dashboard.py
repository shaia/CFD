#!/usr/bin/env python3
"""
Interactive CFD Dashboard using Plotly
Creates web-based interactive visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
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
        elif line.startswith('POINT_DATA'):
            data_start = i + 1
            break

    if not all([dimensions, origin, spacing, data_start]):
        raise ValueError(f"Invalid VTK file format: {filename}")

    nx, ny = dimensions[0], dimensions[1]

    # Create coordinate arrays
    x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
    y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)

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

    return x, y, data_fields

def create_interactive_dashboard(vtk_files):
    """Create an interactive dashboard with multiple visualizations"""

    # Read all data
    all_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            x, y, data_fields = read_vtk_file(filename)
            if 'u' in data_fields and 'v' in data_fields and 'p' in data_fields:
                velocity_mag = np.sqrt(data_fields['u']**2 + data_fields['v']**2)
                vorticity = np.gradient(data_fields['v'], axis=1) - np.gradient(data_fields['u'], axis=0)

                all_data.append({
                    'x': x,
                    'y': y,
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

    if not all_data:
        print("No valid data found!")
        return

    # Create subplots
    fig = sp.make_subplots(
        rows=3, cols=3,
        subplot_titles=('Velocity Magnitude', 'Pressure Field', 'Vorticity',
                       'Vector Field', 'Streamlines (3D)', 'Temperature',
                       'Combined Analysis', 'Time Series', '3D Surface'),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "scatter3d"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "scatter"}, {"type": "surface"}]],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    # Add frames for animation
    frames = []

    for i, (data, time) in enumerate(zip(all_data, times)):
        X, Y = np.meshgrid(data['x'], data['y'])

        frame_data = []

        # 1. Velocity Magnitude (Heatmap)
        frame_data.append(go.Heatmap(
            z=data['velocity_mag'],
            x=data['x'],
            y=data['y'],
            colorscale='Viridis',
            showscale=False
        ))

        # 2. Pressure Field
        frame_data.append(go.Heatmap(
            z=data['p'],
            x=data['x'],
            y=data['y'],
            colorscale='RdBu',
            showscale=False
        ))

        # 3. Vorticity
        frame_data.append(go.Heatmap(
            z=data['vorticity'],
            x=data['x'],
            y=data['y'],
            colorscale='RdBu',
            showscale=False
        ))

        # 4. Vector Field (subsample)
        skip = 5
        X_sub = X[::skip, ::skip]
        Y_sub = Y[::skip, ::skip]
        U_sub = data['u'][::skip, ::skip]
        V_sub = data['v'][::skip, ::skip]

        # Create quiver plot using scatter
        frame_data.append(go.Scatter(
            x=X_sub.flatten(),
            y=Y_sub.flatten(),
            mode='markers',
            marker=dict(
                size=8,
                color=np.sqrt(U_sub**2 + V_sub**2).flatten(),
                colorscale='Viridis',
                showscale=False
            ),
            showlegend=False
        ))

        # 5. 3D Streamlines representation
        frame_data.append(go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=data['velocity_mag'].flatten(),
            mode='markers',
            marker=dict(
                size=2,
                color=data['velocity_mag'].flatten(),
                colorscale='Viridis',
                showscale=False
            ),
            showlegend=False
        ))

        # 6. Temperature
        frame_data.append(go.Heatmap(
            z=data['T'],
            x=data['x'],
            y=data['y'],
            colorscale='Plasma',
            showscale=False
        ))

        # 7. Combined Analysis (Velocity + Pressure contours)
        frame_data.append(go.Heatmap(
            z=data['velocity_mag'],
            x=data['x'],
            y=data['y'],
            colorscale='Viridis',
            opacity=0.7,
            showscale=False
        ))

        # 8. Time Series (placeholder)
        avg_velocity = np.mean(data['velocity_mag'])
        max_pressure = np.max(data['p'])
        frame_data.append(go.Scatter(
            x=times[:i+1],
            y=[np.mean(all_data[j]['velocity_mag']) for j in range(i+1)],
            mode='lines',
            name='Avg Velocity',
            showlegend=False
        ))

        # 9. 3D Surface
        frame_data.append(go.Surface(
            z=data['velocity_mag'],
            x=data['x'],
            y=data['y'],
            colorscale='Viridis',
            showscale=False
        ))

        frames.append(go.Frame(data=frame_data, name=str(i)))

    # Add initial traces
    initial_data = all_data[0]
    X, Y = np.meshgrid(initial_data['x'], initial_data['y'])

    # Row 1
    fig.add_trace(go.Heatmap(
        z=initial_data['velocity_mag'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='Viridis',
        name='Velocity Magnitude'
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=initial_data['p'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='RdBu',
        name='Pressure'
    ), row=1, col=2)

    fig.add_trace(go.Heatmap(
        z=initial_data['vorticity'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='RdBu',
        name='Vorticity'
    ), row=1, col=3)

    # Row 2
    skip = 5
    X_sub = X[::skip, ::skip]
    Y_sub = Y[::skip, ::skip]
    U_sub = initial_data['u'][::skip, ::skip]
    V_sub = initial_data['v'][::skip, ::skip]

    fig.add_trace(go.Scatter(
        x=X_sub.flatten(),
        y=Y_sub.flatten(),
        mode='markers',
        marker=dict(
            size=8,
            color=np.sqrt(U_sub**2 + V_sub**2).flatten(),
            colorscale='Viridis'
        ),
        name='Vector Field'
    ), row=2, col=1)

    fig.add_trace(go.Scatter3d(
        x=X.flatten(),
        y=Y.flatten(),
        z=initial_data['velocity_mag'].flatten(),
        mode='markers',
        marker=dict(
            size=2,
            color=initial_data['velocity_mag'].flatten(),
            colorscale='Viridis'
        ),
        name='3D Flow'
    ), row=2, col=2)

    fig.add_trace(go.Heatmap(
        z=initial_data['T'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='Plasma',
        name='Temperature'
    ), row=2, col=3)

    # Row 3
    fig.add_trace(go.Heatmap(
        z=initial_data['velocity_mag'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='Viridis',
        opacity=0.7,
        name='Combined'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=times,
        y=[np.mean(data['velocity_mag']) for data in all_data],
        mode='lines',
        name='Avg Velocity'
    ), row=3, col=2)

    fig.add_trace(go.Surface(
        z=initial_data['velocity_mag'],
        x=initial_data['x'],
        y=initial_data['y'],
        colorscale='Viridis',
        name='3D Surface'
    ), row=3, col=3)

    # Add animation controls
    fig.frames = frames

    # Add play button
    fig.update_layout(
        title={
            'text': 'Interactive CFD Analysis Dashboard',
            'x': 0.5,
            'font': {'size': 20}
        },
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0.02
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Iteration:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[frame.name], {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }],
                    'label': str(times[i]),
                    'method': 'animate'
                } for i, frame in enumerate(frames)
            ]
        }],
        height=1000,
        showlegend=False
    )

    return fig

def create_flow_analysis_dashboard(vtk_files):
    """Create specialized flow analysis dashboard"""

    # Read data
    all_data = []
    times = []

    for filename in sorted(vtk_files):
        try:
            x, y, data_fields = read_vtk_file(filename)
            if 'u' in data_fields and 'v' in data_fields and 'p' in data_fields:
                velocity_mag = np.sqrt(data_fields['u']**2 + data_fields['v']**2)

                all_data.append({
                    'x': x,
                    'y': y,
                    'u': data_fields['u'],
                    'v': data_fields['v'],
                    'p': data_fields['p'],
                    'velocity_mag': velocity_mag
                })

                iteration = int(os.path.basename(filename).split('_')[-1].split('.')[0])
                times.append(iteration)

        except Exception as e:
            continue

    if not all_data:
        return None

    # Calculate statistics over time
    max_velocities = [np.max(data['velocity_mag']) for data in all_data]
    avg_velocities = [np.mean(data['velocity_mag']) for data in all_data]
    min_pressures = [np.min(data['p']) for data in all_data]
    max_pressures = [np.max(data['p']) for data in all_data]

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add velocity statistics
    fig.add_trace(go.Scatter(
        x=times,
        y=max_velocities,
        mode='lines+markers',
        name='Max Velocity',
        line=dict(color='red', width=3),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=avg_velocities,
        mode='lines+markers',
        name='Avg Velocity',
        line=dict(color='blue', width=3),
        yaxis='y1'
    ))

    # Add pressure statistics on secondary y-axis
    fig.add_trace(go.Scatter(
        x=times,
        y=max_pressures,
        mode='lines+markers',
        name='Max Pressure',
        line=dict(color='green', width=2, dash='dash'),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=times,
        y=min_pressures,
        mode='lines+markers',
        name='Min Pressure',
        line=dict(color='orange', width=2, dash='dash'),
        yaxis='y2'
    ))

    # Update layout
    fig.update_layout(
        title='CFD Flow Analysis - Time Evolution',
        xaxis=dict(title='Iteration'),
        yaxis=dict(title='Velocity', side='left'),
        yaxis2=dict(title='Pressure', side='right', overlaying='y'),
        legend=dict(x=0.02, y=0.98),
        height=600,
        hovermode='x unified'
    )

    return fig

def main():
    """Main function to create interactive visualizations"""

    # Create output directory if it doesn't exist
    os.makedirs("visualization/visualization_output", exist_ok=True)

    # Find all VTK files
    vtk_pattern = "output/output_optimized_*.vtk"
    vtk_files = glob.glob(vtk_pattern)

    if not vtk_files:
        print(f"No VTK files found matching pattern: {vtk_pattern}")
        return

    print(f"Found {len(vtk_files)} VTK files")

    # Create interactive dashboard
    print("Creating interactive dashboard...")
    dashboard_fig = create_interactive_dashboard(vtk_files)

    if dashboard_fig:
        plot(dashboard_fig, filename='visualization/visualization_output/cfd_interactive_dashboard.html', auto_open=True)
        print("Interactive dashboard saved as 'visualization/visualization_output/cfd_interactive_dashboard.html'")

    # Create flow analysis dashboard
    print("Creating flow analysis dashboard...")
    analysis_fig = create_flow_analysis_dashboard(vtk_files)

    if analysis_fig:
        plot(analysis_fig, filename='visualization/visualization_output/cfd_flow_analysis.html', auto_open=True)
        print("Flow analysis dashboard saved as 'visualization/visualization_output/cfd_flow_analysis.html'")

    print("Interactive visualizations complete!")
    print("Open the HTML files in your web browser to interact with the visualizations.")

if __name__ == "__main__":
    main()