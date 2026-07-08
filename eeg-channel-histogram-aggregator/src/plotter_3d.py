import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append('/home/data/ninalaemmermann/forschung/eeg-channel-histogram-aggregator')
from config.settings import OUTPUT_DIR
import os


def plot_3d_scatter(sample_matrix_data, seizure_type, n_channels=15):
    """
    Erstellt einen interaktiven 3D-Scatter-Plot mit den Top-3 Kanälen als Achsen
    und weiteren Kanälen als Farbkodierung.
    
    Parameters:
    -----------
    sample_matrix_data : dict
        Dictionary mit 'seizure_matrix', 'non_seizure_matrix', 'channel_names'
    seizure_type : str
        Typ des Anfalls (CPSZ, ABSZ, etc.)
    n_channels : int
        Anzahl der Kanäle
    """
    print(f"\n{'='*60}")
    print(f"ERSTELLE 3D-SCATTER-PLOT FÜR {seizure_type}")
    print(f"{'='*60}")
    
    seizure_matrix = sample_matrix_data['seizure_matrix']
    non_seizure_matrix = sample_matrix_data['non_seizure_matrix']
    channel_names = sample_matrix_data['channel_names']
    
    print(f"\nSeizure-Samples: {seizure_matrix.shape[0]:,}")
    print(f"Non-Seizure-Samples: {non_seizure_matrix.shape[0]:,}")
    print(f"Kanäle: {len(channel_names)}")
    
    # Top-3 Kanäle für X, Y, Z Achsen (Indizes 0, 1, 2)
    x_channel_idx = 0
    y_channel_idx = 1
    z_channel_idx = 2
    
    # Kanal 4 (Index 3) für Farbkodierung - Ch_5 mit hoher Separabilität
    color_channel_idx = 3 if len(channel_names) > 3 else 0
    
    print(f"\nAchsen-Zuordnung:")
    print(f"  X-Achse: {channel_names[x_channel_idx]}")
    print(f"  Y-Achse: {channel_names[y_channel_idx]}")
    print(f"  Z-Achse: {channel_names[z_channel_idx]}")
    print(f"  Farbe:   {channel_names[color_channel_idx]}")
    
    # Create figure
    fig = go.Figure()
    
    # Seizure-Samples (Rot)
    print("\nErstelle Seizure-Scatter...")
    fig.add_trace(go.Scatter3d(
        x=seizure_matrix[:, x_channel_idx],
        y=seizure_matrix[:, y_channel_idx],
        z=seizure_matrix[:, z_channel_idx],
        mode='markers',
        name='Seizure',
        marker=dict(
            size=2,
            color=seizure_matrix[:, color_channel_idx],  # Farbkodierung
            colorscale='Reds',
            showscale=True,
            opacity=0.6,
            colorbar=dict(
                title=f"{channel_names[color_channel_idx]}<br>Amplitude (µV)",
                x=1.15,
                len=0.5,
                y=0.75
            ),
            line=dict(width=0)
        ),
        text=[f"Sample {i}" for i in range(seizure_matrix.shape[0])],
        hovertemplate=
            f'<b>Seizure Sample</b><br>' +
            f'{channel_names[x_channel_idx]}: %{{x:.2f}} µV<br>' +
            f'{channel_names[y_channel_idx]}: %{{y:.2f}} µV<br>' +
            f'{channel_names[z_channel_idx]}: %{{z:.2f}} µV<br>' +
            f'{channel_names[color_channel_idx]}: %{{marker.color:.2f}} µV<br>' +
            '<extra></extra>'
    ))
    
    # Non-Seizure-Samples (Blau)
    print("Erstelle Non-Seizure-Scatter...")
    fig.add_trace(go.Scatter3d(
        x=non_seizure_matrix[:, x_channel_idx],
        y=non_seizure_matrix[:, y_channel_idx],
        z=non_seizure_matrix[:, z_channel_idx],
        mode='markers',
        name='Non-Seizure',
        marker=dict(
            size=2,
            color=non_seizure_matrix[:, color_channel_idx],  # Farbkodierung
            colorscale='Blues',
            showscale=True,
            opacity=0.6,
            colorbar=dict(
                title=f"{channel_names[color_channel_idx]}<br>Amplitude (µV)",
                x=1.15,
                len=0.5,
                y=0.25
            ),
            line=dict(width=0)
        ),
        text=[f"Sample {i}" for i in range(non_seizure_matrix.shape[0])],
        hovertemplate=
            f'<b>Non-Seizure Sample</b><br>' +
            f'{channel_names[x_channel_idx]}: %{{x:.2f}} µV<br>' +
            f'{channel_names[y_channel_idx]}: %{{y:.2f}} µV<br>' +
            f'{channel_names[z_channel_idx]}: %{{z:.2f}} µV<br>' +
            f'{channel_names[color_channel_idx]}: %{{marker.color:.2f}} µV<br>' +
            '<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f'3D EEG Amplitude Distribution - {seizure_type} (Top {n_channels} Channels)<br>' +
                 f'<sub>X: {channel_names[x_channel_idx]}, Y: {channel_names[y_channel_idx]}, ' +
                 f'Z: {channel_names[z_channel_idx]}, Color: {channel_names[color_channel_idx]}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                title=f'{channel_names[x_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            yaxis=dict(
                title=f'{channel_names[y_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            zaxis=dict(
                title=f'{channel_names[z_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1400,
        height=900,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Speichern
    output_path = os.path.join(OUTPUT_DIR, f'3d_scatter_{seizure_type}_top{n_channels}_channels.html')
    print(f"\nSpeichere interaktive 3D-Visualisierung...")
    fig.write_html(output_path)
    print(f"✓ Gespeichert: {output_path}")
    print(f"\n{'='*60}")
    print(f"FERTIG! Öffne die HTML-Datei im Browser für interaktive 3D-Ansicht")
    print(f"{'='*60}")
    
    return fig


def plot_3d_scatter_with_animation(sample_matrix_data, seizure_type, n_channels=15):
    """
    Erstellt einen animierten 3D-Scatter-Plot, der durch verschiedene
    Kanal-Kombinationen rotiert.
    
    Parameters:
    -----------
    sample_matrix_data : dict
        Dictionary mit 'seizure_matrix', 'non_seizure_matrix', 'channel_names'
    seizure_type : str
        Typ des Anfalls (CPSZ, ABSZ, etc.)
    n_channels : int
        Anzahl der Kanäle
    """
    print(f"\n{'='*60}")
    print(f"ERSTELLE ANIMIERTEN 3D-SCATTER-PLOT FÜR {seizure_type}")
    print(f"{'='*60}")
    
    seizure_matrix = sample_matrix_data['seizure_matrix']
    non_seizure_matrix = sample_matrix_data['non_seizure_matrix']
    channel_names = sample_matrix_data['channel_names']
    
    # Verschiedene Kamerawinkel für Animation
    camera_angles = []
    n_frames = 36  # 360° in 10° Schritten
    
    for i in range(n_frames):
        angle = i * 10  # Grad
        rad = np.radians(angle)
        camera_angles.append(dict(
            eye=dict(
                x=1.5 * np.cos(rad),
                y=1.5 * np.sin(rad),
                z=1.3
            )
        ))
    
    # Verwende Top-3 Kanäle
    x_channel_idx = 0
    y_channel_idx = 1
    z_channel_idx = 2
    color_channel_idx = 3 if len(channel_names) > 3 else 0
    
    print(f"\nAchsen-Zuordnung:")
    print(f"  X-Achse: {channel_names[x_channel_idx]}")
    print(f"  Y-Achse: {channel_names[y_channel_idx]}")
    print(f"  Z-Achse: {channel_names[z_channel_idx]}")
    print(f"  Farbe:   {channel_names[color_channel_idx]}")
    print(f"\nAnimations-Frames: {n_frames}")
    
    # Create figure with animation frames
    frames = []
    for i, camera in enumerate(camera_angles):
        frame = go.Frame(
            name=f'frame_{i}',
            layout=dict(scene_camera=camera)
        )
        frames.append(frame)
    
    # Create base figure (same as before)
    fig = go.Figure(
        data=[
            # Seizure trace
            go.Scatter3d(
                x=seizure_matrix[:, x_channel_idx],
                y=seizure_matrix[:, y_channel_idx],
                z=seizure_matrix[:, z_channel_idx],
                mode='markers',
                name='Seizure',
                marker=dict(
                    size=2,
                    color=seizure_matrix[:, color_channel_idx],
                    colorscale='Reds',
                    showscale=True,
                    opacity=0.6,
                    colorbar=dict(
                        title=f"{channel_names[color_channel_idx]}<br>Amplitude (µV)",
                        x=1.15,
                        len=0.5,
                        y=0.75
                    ),
                    line=dict(width=0)
                ),
                hovertemplate=
                    f'<b>Seizure</b><br>' +
                    f'{channel_names[x_channel_idx]}: %{{x:.2f}} µV<br>' +
                    f'{channel_names[y_channel_idx]}: %{{y:.2f}} µV<br>' +
                    f'{channel_names[z_channel_idx]}: %{{z:.2f}} µV<br>' +
                    '<extra></extra>'
            ),
            # Non-Seizure trace
            go.Scatter3d(
                x=non_seizure_matrix[:, x_channel_idx],
                y=non_seizure_matrix[:, y_channel_idx],
                z=non_seizure_matrix[:, z_channel_idx],
                mode='markers',
                name='Non-Seizure',
                marker=dict(
                    size=2,
                    color=non_seizure_matrix[:, color_channel_idx],
                    colorscale='Blues',
                    showscale=True,
                    opacity=0.6,
                    colorbar=dict(
                        title=f"{channel_names[color_channel_idx]}<br>Amplitude (µV)",
                        x=1.15,
                        len=0.5,
                        y=0.25
                    ),
                    line=dict(width=0)
                ),
                hovertemplate=
                    f'<b>Non-Seizure</b><br>' +
                    f'{channel_names[x_channel_idx]}: %{{x:.2f}} µV<br>' +
                    f'{channel_names[y_channel_idx]}: %{{y:.2f}} µV<br>' +
                    f'{channel_names[z_channel_idx]}: %{{z:.2f}} µV<br>' +
                    '<extra></extra>'
            )
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title=dict(
            text=f'3D EEG Amplitude Distribution (Animated) - {seizure_type}<br>' +
                 f'<sub>X: {channel_names[x_channel_idx]}, Y: {channel_names[y_channel_idx]}, ' +
                 f'Z: {channel_names[z_channel_idx]}, Color: {channel_names[color_channel_idx]}</sub>',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                title=f'{channel_names[x_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            yaxis=dict(
                title=f'{channel_names[y_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            zaxis=dict(
                title=f'{channel_names[z_channel_idx]} Amplitude (µV)',
                backgroundcolor="rgb(230, 230, 230)",
                gridcolor="white",
                showbackground=True,
            ),
            camera=camera_angles[0]
        ),
        width=1400,
        height=900,
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True,
                                  "mode": "immediate",
                                  "transition": {"duration": 50}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}])
            ],
            x=0.1,
            y=1.15
        )],
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Speichern
    output_path = os.path.join(OUTPUT_DIR, f'3d_scatter_animated_{seizure_type}_top{n_channels}_channels.html')
    print(f"\nSpeichere animierte 3D-Visualisierung...")
    fig.write_html(output_path)
    print(f"✓ Gespeichert: {output_path}")
    print(f"\n{'='*60}")
    print(f"FERTIG! Öffne die HTML-Datei und klicke 'Play' für Animation")
    print(f"{'='*60}")
    
    return fig
