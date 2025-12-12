"""
Visualization utilities for SfM reconstructions using Plotly.
"""

from __future__ import annotations

import plotly.graph_objs as go
import numpy as np

from sfm_app.sfm_inc.data_structures import SceneGraph


def plot_sfm_reconstruct(scene: SceneGraph) -> go.Figure:
    """
    Create a 3D Plotly visualization of the SfM reconstruction.

    Args:
        scene: SceneGraph containing cameras, 3D points, and observations.

    Returns:
        Plotly Figure object with 3D scatter plots of points and camera centers.
    """
    # Extract 3D points
    if len(scene.points3d) > 0:
        points_xyz = np.array([pt.xyz for pt in scene.points3d])
        points_colors = np.array([pt.color for pt in scene.points3d])
    else:
        points_xyz = np.array([]).reshape(0, 3)
        points_colors = np.array([]).reshape(0, 3)

    # Compute camera centers: C = -R^T @ t
    camera_centers = []
    for cam in scene.cameras:
        C = -cam.R.T @ cam.t
        camera_centers.append(C.flatten())

    camera_centers = np.array(camera_centers) if camera_centers else np.array([]).reshape(0, 3)

    # Create figure
    fig = go.Figure()

    # Add 3D points
    if len(points_xyz) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=points_xyz[:, 0],
                y=points_xyz[:, 1],
                z=points_xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=2,
                    color=points_colors,
                    colorscale=None,  # Use direct RGB colors
                    opacity=0.8,
                ),
                name="3D Points",
                text=[f"Point {i}" for i in range(len(points_xyz))],
            )
        )

    # Add camera centers
    if len(camera_centers) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=camera_centers[:, 0],
                y=camera_centers[:, 1],
                z=camera_centers[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color="red",
                    symbol="diamond",
                ),
                name="Camera Centers",
                text=[f"Camera {i}" for i in range(len(camera_centers))],
            )
        )

    # Set layout
    fig.update_layout(
        title="SfM 3D Reconstruction",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        width=800,
        height=600,
    )

    return fig


__all__ = ["plot_sfm_reconstruct"]

