#!/usr/bin/env python3
"""
Streamlit IMU Data Plotter - Interactive web-based visualization
Deploy with: streamlit run streamlit_imu_plotter.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Sensa IMU Data Plotter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Color palette
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]

def detect_channel_groups(headers):
    """Build channel groups dynamically based on present columns"""
    groups = {}
    
    # Find present IMU indices
    present_indices = set()
    for i in range(1, 7):
        for sig in ['a', 'g']:
            for axis in ['x', 'y', 'z']:
                ch = f"{sig}{axis}{i}"
                if ch in headers:
                    present_indices.add(i)
                    break
    
    # Per-IMU groups
    for i in sorted(present_indices):
        imu_list = []
        for sig in ['a', 'g']:
            for axis in ['x', 'y', 'z']:
                ch = f"{sig}{axis}{i}"
                if ch in headers:
                    imu_list.append(ch)
        if imu_list:
            groups[f"IMU {i}"] = imu_list
    
    # Super-groups
    accel_all = [f"a{axis}{i}" for i in sorted(present_indices) 
                 for axis in ['x', 'y', 'z'] if f"a{axis}{i}" in headers]
    gyro_all = [f"g{axis}{i}" for i in sorted(present_indices) 
                for axis in ['x', 'y', 'z'] if f"g{axis}{i}" in headers]
    
    if accel_all:
        groups["All Accelerometers"] = accel_all
    if gyro_all:
        groups["All Gyroscopes"] = gyro_all
    
    # Stimulation parameters
    stim_params = ["stim_state", "stim_on_ms", "stim_off_ms", "stim_timing_ms"]
    stim_present = [ch for ch in stim_params if ch in headers]
    if stim_present:
        groups["Stimulation"] = stim_present
    
    # Onset detection parameters
    onset_params = ["onset_detected", "onset_time_ms", "onset_latency_ms", 
                    "predicted_onset_flag", "predicted_onset_time_ms"]
    onset_present = [ch for ch in onset_params if ch in headers]
    if onset_present:
        groups["Onset Detection"] = onset_present
    
    # Event markers
    event_channels = [ch for ch in ['cue', 'cue_norm'] if ch in headers]
    if event_channels:
        groups["Events"] = event_channels
    
    return groups

def plot_data(data, selected_channels, show_events):
    """Create the matplotlib plot"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get time data
    if 't_ms' in data.columns:
        time_data = pd.to_numeric(data['t_ms'], errors='coerce')
    else:
        time_data = np.arange(len(data), dtype=float)
    
    # Event marker definitions
    event_marker_channels = {
        'cue': ('red', '--', 'Cue'),
        'cue_norm': ('red', '--', 'Cue'),
        'onset_detected': ('green', '-', 'Onset Detected'),
        'predicted_onset_flag': ('blue', '-.', 'Predicted Onset')
    }
    
    # Plot regular channels
    lines_plotted = 0
    color_idx = 0
    
    for channel in selected_channels:
        if channel == 't_ms' or channel.lower() in event_marker_channels:
            continue
        
        try:
            y_data = pd.to_numeric(data[channel], errors='coerce')
            if not y_data.isna().all():
                color = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
                ax.plot(time_data, y_data, label=channel, color=color, linewidth=1.5, alpha=0.8)
                lines_plotted += 1
                color_idx += 1
        except Exception:
            continue
    
    # Plot event markers if requested
    if show_events:
        for marker_ch, (color, linestyle, label) in event_marker_channels.items():
            if marker_ch in selected_channels:
                try:
                    marker_data = pd.to_numeric(data[marker_ch], errors='coerce')
                    if not marker_data.isna().all():
                        if marker_ch == 'onset_detected':
                            # Rising edge detection for onset_detected
                            rising_edges = (marker_data > 0) & (marker_data.shift(1, fill_value=0) == 0)
                            marker_times = time_data[rising_edges]
                        else:
                            # All instances for other markers
                            marker_times = time_data[marker_data > 0]
                        
                        for idx, mt in enumerate(marker_times):
                            ax.axvline(x=mt, color=color, linestyle=linestyle,
                                      alpha=0.7, linewidth=2, 
                                      label=label if idx == 0 else "")
                except Exception:
                    continue
    
    # Formatting
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("IMU Data Visualization", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if lines_plotted > 0 or show_events:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    return fig

def get_download_link(fig, filename="plot.png"):
    """Generate a download link for the plot"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Plot (PNG)</a>'

# Main app
def main():
    st.title("📊 Sensa IMU Data Plotter")
    st.markdown("Interactive visualization of multi-IMU sensor data with onset detection")
    
    # File upload
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload IMU data file (CSV/TXT)", 
        type=['csv', 'txt'],
        help="Upload a comma-separated file with IMU data"
    )
    
    # Demo data option
    use_demo = st.sidebar.checkbox("Use demo data", value=True if not uploaded_file else False)
    
    data = None
    
    # Load demo data if selected
    if use_demo and not uploaded_file:
        try:
            # Try to load the demo file from the same directory
            import os
            demo_file = "demo_data.csv"
            if os.path.exists(demo_file):
                data = pd.read_csv(demo_file)
                st.sidebar.success("✓ Demo data loaded")
            else:
                st.sidebar.warning("Demo data file not found. Please upload your own data.")
        except Exception as e:
            st.sidebar.error(f"Error loading demo data: {str(e)}")
    
    # Load uploaded file
    elif uploaded_file is not None:
        try:
            # Try different separators
            separators = [',', '\t', ';', '|']
            for sep in separators:
                try:
                    data = pd.read_csv(uploaded_file, sep=sep)
                    data.columns = data.columns.str.strip()
                    if len(data.columns) > 1:
                        break
                except Exception:
                    continue
            
            if data is not None and len(data.columns) > 1:
                st.sidebar.success(f"✓ Loaded {len(data)} rows × {len(data.columns)} columns")
            else:
                st.sidebar.error("Could not parse file. Please check the format.")
                return
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            return
    
    if data is None:
        st.info("👈 Please upload a data file or enable demo data in the sidebar")
        
        # Show expected format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Required columns:**
            - `t_ms`: Timestamp in milliseconds
            
            **IMU channels (any subset of 1-6 IMUs):**
            - Accelerometer: `ax1, ay1, az1, ax2, ay2, az2, ...`
            - Gyroscope: `gx1, gy1, gz1, gx2, gy2, gz2, ...`
            
            **Optional columns:**
            - Events: `cue`, `onset_detected`, `predicted_onset_flag`
            - Stimulation: `stim_state`, `stim_on_ms`, `stim_off_ms`, `stim_timing_ms`
            - Onset params: `onset_time_ms`, `onset_latency_ms`, `predicted_onset_time_ms`
            """)
        return
    
    # Get headers (excluding t_ms)
    headers = [col for col in data.columns if col != 't_ms']
    
    # Build channel groups
    channel_groups = detect_channel_groups(headers)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Plot Controls")
    
    # Group-based selection
    st.sidebar.subheader("Channel Groups")
    selected_channels = []
    
    for group_name, group_channels in channel_groups.items():
        if st.sidebar.checkbox(group_name, value=False, key=f"group_{group_name}"):
            selected_channels.extend(group_channels)
    
    # Individual channel selection
    with st.sidebar.expander("Individual Channels"):
        for channel in headers:
            if channel not in selected_channels:
                if st.checkbox(channel, value=False, key=f"ch_{channel}"):
                    selected_channels.append(channel)
    
    # Remove duplicates while preserving order
    selected_channels = list(dict.fromkeys(selected_channels))
    
    # Event markers toggle
    show_events = st.sidebar.checkbox("Show Event Markers", value=True)
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Info")
    st.sidebar.metric("Total Rows", f"{len(data):,}")
    st.sidebar.metric("Channels Selected", len(selected_channels))
    
    if 't_ms' in data.columns:
        t_range = data['t_ms'].max() - data['t_ms'].min()
        st.sidebar.metric("Duration", f"{t_range/1000:.2f} sec")
    
    # Main plot area
    if len(selected_channels) == 0:
        st.warning("⚠️ Please select at least one channel from the sidebar")
    else:
        # Create plot
        with st.spinner("Generating plot..."):
            fig = plot_data(data, selected_channels, show_events)
        
        # Display plot
        st.pyplot(fig)
        
        # Download button
        st.markdown(get_download_link(fig, "sensa_plot.png"), unsafe_allow_html=True)
        
        # Channel statistics
        with st.expander("📈 Channel Statistics"):
            stats_data = []
            for channel in selected_channels:
                if channel not in ['cue', 'onset_detected', 'predicted_onset_flag']:
                    try:
                        y_data = pd.to_numeric(data[channel], errors='coerce')
                        if not y_data.isna().all():
                            stats_data.append({
                                'Channel': channel,
                                'Mean': f"{y_data.mean():.2f}",
                                'Std': f"{y_data.std():.2f}",
                                'Min': f"{y_data.min():.2f}",
                                'Max': f"{y_data.max():.2f}",
                            })
                    except Exception:
                        continue
            
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        # Event counts
        event_cols = ['cue', 'onset_detected', 'predicted_onset_flag']
        event_data = []
        for col in event_cols:
            if col in data.columns:
                count = (data[col] > 0).sum()
                if col == 'onset_detected':
                    # Count rising edges
                    rising_edges = (data[col] > 0) & (data[col].shift(1, fill_value=0) == 0)
                    edge_count = rising_edges.sum()
                    event_data.append({'Event Type': col, 'Count': f"{edge_count} (edges)"})
                else:
                    event_data.append({'Event Type': col, 'Count': count})
        
        if event_data:
            with st.expander("🎯 Event Counts"):
                st.dataframe(pd.DataFrame(event_data), use_container_width=True)

if __name__ == "__main__":
    main()
