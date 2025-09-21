# CFD Visualization Tools

Advanced Python scripts for visualizing CFD simulation results with multiple visualization types.

## Setup

Install required packages:
```bash
pip install -r requirements.txt
```

## Available Visualizations

### 1. Basic Animations (`visualize_cfd.py`)
- Velocity magnitude evolution
- Pressure field changes
- Flow streamlines
- Saves as GIF files

### 2. Enhanced Analysis (`enhanced_visualize.py`)
- **Comprehensive 6-panel dashboard**: Velocity, pressure, vorticity, vectors, streamlines, combined analysis
- **Vorticity analysis**: Detailed vorticity field and streamline visualization
- **Statistical analysis**: Time evolution plots of flow statistics
- High-quality animations with custom colormaps

### 3. Interactive Dashboards (`interactive_dashboard.py`)
- **Web-based interactive dashboard**: Multi-panel view with animation controls
- **Flow analysis dashboard**: Time series analysis with interactive plots
- Opens in web browser with play/pause controls and sliders

## Usage

### Quick visualization:
```bash
python visualize_cfd.py
```

### Enhanced analysis:
```bash
python enhanced_visualize.py
```

### Interactive dashboards:
```bash
python interactive_dashboard.py
```

### Complete workflow:
```bash
python run_visualization.py
```

## Output Files

**GIF Animations:**
- `cfd_animation_velocity_magnitude.gif` - Basic velocity animation
- `cfd_animation_p.gif` - Pressure field animation
- `cfd_streamlines.gif` - Streamline animation
- `cfd_comprehensive_analysis.gif` - 6-panel analysis (3.5MB)
- `cfd_vorticity_analysis.gif` - Vorticity-focused analysis (2.2MB)

**Interactive HTML:**
- `cfd_interactive_dashboard.html` - Full interactive dashboard (13MB)
- `cfd_flow_analysis.html` - Time series analysis (4.8MB)

**Static Images:**
- `cfd_statistical_analysis.png` - Statistical summary plots

## Features

✅ **Multi-panel dashboards** with synchronized animations
✅ **Vorticity analysis** with curl calculations
✅ **Interactive web dashboards** with plotly
✅ **Statistical time series** analysis
✅ **Custom colormaps** for better visualization
✅ **Vector field** and **streamline** plots
✅ **Pressure contour** overlays
✅ **3D surface** representations

Open the HTML files in any web browser for interactive exploration!