"""
Incremental Structure-from-Motion pipeline.
"""

from __future__ import annotations

from typing import List

import numpy as np

from sfm_app.features.keypoints import detect_keypoints
from sfm_app.features.matching import filter_matches_ratio_test, match_keypoints
from sfm_app.geometry.essential import compute_essential_matrix, extract_RT_essential_matrix
from sfm_app.geometry.fundamental import constrain_F, fundamental_matrix_ransac
from sfm_app.geometry.pnp import estimate_camera_pose_pnp
from sfm_app.geometry.triangulation import triangulate_matched_key_pts_to_3D_pts
from sfm_app.sfm.data_structures import Camera, Observation, Point3D, SceneGraph


def build_base_scene_from_two_frames(
    frames: List[np.ndarray],
    frame_idx1: int,
    frame_idx2: int,
    K: np.ndarray,
) -> SceneGraph:
    """
    Build initial scene from two keyframes using two-view reconstruction.

    Args:
        frames: List of frame images.
        frame_idx1: Index of first keyframe.
        frame_idx2: Index of second keyframe.
        K: Intrinsic camera matrix (3x3).

    Returns:
        SceneGraph containing two cameras, triangulated 3D points, and observations.
    """
    frame1 = frames[frame_idx1]
    frame2 = frames[frame_idx2]

    # Detect keypoints and descriptors
    kp1, desc1 = detect_keypoints(frame1, use_sift=True)
    kp2, desc2 = detect_keypoints(frame2, use_sift=True)
    print(
        f"[sfm] Base frames {frame_idx1} & {frame_idx2}: "
        f"{len(kp1)} kp in frame1, {len(kp2)} kp in frame2"
    )

    if len(kp1) == 0 or len(kp2) == 0 or len(desc1) == 0 or len(desc2) == 0:
        # Return empty scene if no features detected
        print("[sfm] No features detected in one of the base frames; aborting SfM base.")
        return SceneGraph()

    # Match descriptors
    knn_matches = match_keypoints(desc1, desc2, use_flann=True)
    pts1, pts2, good_matches = filter_matches_ratio_test(kp1, kp2, knn_matches, ratio=0.75)
    print(
        f"[sfm] Base matching: {len(knn_matches)} knn matches, "
        f"{len(good_matches)} after ratio test"
    )
    # Debug: inspect matched 2D points before any geometry.
    print(
        "[DEBUG] After ratio test:",
        "pts1 shape", pts1.shape,
        "pts2 shape", pts2.shape,
    )
    if len(pts1) > 0:
        print("[DEBUG] pts1 first 5:\n", pts1[:5])
        print("[DEBUG] pts2 first 5:\n", pts2[:5])
        print(
            "[DEBUG] pts1 unique (rounded 1px):",
            np.unique(pts1.round(1), axis=0).shape[0],
        )
        print(
            "[DEBUG] pts2 unique (rounded 1px):",
            np.unique(pts2.round(1), axis=0).shape[0],
        )

    if len(pts1) < 8:
        # Need at least 8 points for fundamental matrix
        print(f"[sfm] Only {len(pts1)} good matches; need >=8 for F. Aborting base scene.")
        return SceneGraph()

    # Estimate fundamental matrix
    F, inlier_mask = fundamental_matrix_ransac(pts1, pts2)
    F = constrain_F(F)

    # Filter matches by inliers
    pts1_inliers = pts1[inlier_mask]
    pts2_inliers = pts2[inlier_mask]
    good_matches_inliers = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
    print(f"[sfm] F RANSAC inliers: {len(pts1_inliers)}")
    # Debug: inspect F-RANSAC inliers.
    print(
        "[DEBUG] After F inliers:",
        "pts1_inliers shape", pts1_inliers.shape,
        "pts2_inliers shape", pts2_inliers.shape,
    )
    if len(pts1_inliers) > 0:
        print("[DEBUG] pts1_inliers first 5:\n", pts1_inliers[:5])
        print("[DEBUG] pts2_inliers first 5:\n", pts2_inliers[:5])
        print(
            "[DEBUG] pts1_inliers unique (rounded 1px):",
            np.unique(pts1_inliers.round(1), axis=0).shape[0],
        )
        print(
            "[DEBUG] pts2_inliers unique (rounded 1px):",
            np.unique(pts2_inliers.round(1), axis=0).shape[0],
        )

    if len(pts1_inliers) < 8:
        print("[sfm] Too few F inliers (<8); aborting base scene.")
        return SceneGraph()

    # Compute essential matrix
    E = compute_essential_matrix(K, F)

    # Extract pose
    R, t, pose_mask = extract_RT_essential_matrix(E, K, pts1_inliers, pts2_inliers)
    num_pose_inliers = int(np.sum(pose_mask))
    print(f"[sfm] Pose recoverPose inliers: {num_pose_inliers}")

    # Filter by pose inliers (points in front of both cameras)
    pts1_pose_inliers = pts1_inliers[pose_mask]
    pts2_pose_inliers = pts2_inliers[pose_mask]

    if len(pts1_pose_inliers) < 8:
        print("[sfm] Too few pose inliers (<8); aborting base scene.")
        return SceneGraph()

    # Set up poses: camera 0 is identity, camera 1 uses recovered R, t
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    R2 = R
    t2 = t

    # Triangulate 3D points
    print(
        "[DEBUG] Triangulation inputs:",
        "pts1_pose_inliers shape", pts1_pose_inliers.shape,
        "pts2_pose_inliers shape", pts2_pose_inliers.shape,
    )
    if len(pts1_pose_inliers) > 0:
        print("[DEBUG] pts1_pose_inliers first 5:\n", pts1_pose_inliers[:5])
        print("[DEBUG] pts2_pose_inliers first 5:\n", pts2_pose_inliers[:5])
        print(
            "[DEBUG] pts1 unique (rounded 1px):",
            np.unique(pts1_pose_inliers.round(1), axis=0).shape[0],
        )
        print(
            "[DEBUG] pts2 unique (rounded 1px):",
            np.unique(pts2_pose_inliers.round(1), axis=0).shape[0],
        )
    points_3d, reproj_errors = triangulate_matched_key_pts_to_3D_pts(
        K, R1, t1, R2, t2, pts1_pose_inliers, pts2_pose_inliers
    )
    print(f"[sfm] Triangulated {len(points_3d)} points from pose inliers")

    # Filter out points with high reprojection error or behind cameras.
    # Points should be in front of both cameras (positive Z in camera coordinates)
    z1 = (R1 @ points_3d.T + t1).T[:, 2]
    z2 = (R2 @ points_3d.T + t2).T[:, 2]

    mask_z = (z1 > 0) & (z2 > 0)
    # Use a tighter reprojection error threshold; scenes/videos can be noisy
    # but 50px is too lax and admits far-away outliers.
    mask_err = reproj_errors < 5.0

    # Norm-based sanity filter: discard points whose radius is far beyond
    # the typical scale of this batch.
    norms = np.linalg.norm(points_3d, axis=1)
    median_norm = np.median(norms) if norms.size > 0 else 0.0
    if median_norm > 0:
        max_allowed_norm = 5.0 * median_norm
        mask_norm = norms < max_allowed_norm
    else:
        mask_norm = np.ones_like(norms, dtype=bool)

    valid_mask = mask_z & mask_err & mask_norm

    num_valid = int(np.sum(valid_mask))
    print(
        "[sfm] Triangulated depth / error stats:\n"
        f"       z1 range: [{z1.min():.3f}, {z1.max():.3f}], "
        f"z2 range: [{z2.min():.3f}, {z2.max():.3f}]\n"
        f"       reproj error: min={reproj_errors.min():.3f}, "
        f"mean={reproj_errors.mean():.3f}, max={reproj_errors.max():.3f}\n"
        f"       points passing z>0: {int(mask_z.sum())}, "
        f"passing error<5: {int(mask_err.sum())}, "
        f"passing all filters: {num_valid}"
    )

    if num_valid == 0:
        print(
            "[sfm] No points pass z>0 & error<50px; "
            "falling back to using only cheirality (z>0)"
        )
        valid_mask = mask_z
        num_valid = int(np.sum(valid_mask))

    if num_valid == 0:
        print("[sfm] No valid 3D points even after cheirality-only filtering; aborting base scene.")
        return SceneGraph()

    points_3d_valid = points_3d[valid_mask]
    pts1_valid = pts1_pose_inliers[valid_mask]
    pts2_valid = pts2_pose_inliers[valid_mask]

    # Create SceneGraph
    scene = SceneGraph()

    # Create cameras
    # Camera 0 (first frame)
    cam0 = Camera(
        id=0,
        K=K,
        R=R1,
        t=t1,
        image_idx=frame_idx1,
        keypoints=pts1_valid,
        descriptors=desc1,  # Store full descriptor array
    )
    scene.cameras.append(cam0)

    # Camera 1 (second frame)
    cam1 = Camera(
        id=1,
        K=K,
        R=R2,
        t=t2,
        image_idx=frame_idx2,
        keypoints=pts2_valid,
        descriptors=desc2,  # Store full descriptor array
    )
    scene.cameras.append(cam1)

    # Create 3D points and observations
    # Sample colors from the images
    for i, pt_3d in enumerate(points_3d_valid):
        # Get color from first image (simple bilinear sampling would be better)
        u1, v1 = int(pts1_valid[i][0]), int(pts1_valid[i][1])
        if 0 <= v1 < frame1.shape[0] and 0 <= u1 < frame1.shape[1]:
            color = frame1[v1, u1].astype(np.uint8)
        else:
            color = np.array([128, 128, 128], dtype=np.uint8)

        point3d = Point3D(
            id=i,
            xyz=pt_3d,
            color=color,
            observations=[],
        )
        scene.points3d.append(point3d)

        # Create observations
        obs0_idx = len(scene.observations)
        obs1_idx = len(scene.observations) + 1

        obs0 = Observation(camera_id=0, point_id=i, uv=pts1_valid[i])
        obs1 = Observation(camera_id=1, point_id=i, uv=pts2_valid[i])

        scene.observations.append(obs0)
        scene.observations.append(obs1)

        point3d.observations.extend([obs0_idx, obs1_idx])

    return scene


def add_camera_incremental(
    scene: SceneGraph,
    K: np.ndarray,
    new_image: np.ndarray,
    image_idx: int,
) -> SceneGraph:
    """
    Add a new camera to the scene using PnP and triangulate new 3D points.

    Args:
        scene: Existing SceneGraph.
        K: Intrinsic camera matrix (3x3).
        new_image: New image frame to add.
        image_idx: Index of the new image in the original frame list.

    Returns:
        Updated SceneGraph with new camera, observations, and possibly new 3D points.
    """
    if len(scene.cameras) == 0:
        raise ValueError("Cannot add camera to empty scene")

    # Detect keypoints and descriptors for new image
    kp_new, desc_new = detect_keypoints(new_image, use_sift=True)

    if len(kp_new) == 0 or len(desc_new) == 0:
        return scene  # No features detected

    # Match with existing cameras to find correspondences
    best_camera_match = None
    best_matches = None
    best_kp_new = None
    best_desc_new = desc_new
    max_matches = 0

    # Try matching with all existing cameras
    for cam in scene.cameras:
        if len(cam.descriptors) == 0:
            continue

        knn_matches = match_keypoints(desc_new, cam.descriptors, use_flann=True)
        
        # Extract keypoints from matches - we need to create dummy keypoints from cam.keypoints
        # since cam.keypoints is stored as (N, 2) array, not cv2.KeyPoint objects
        # For ratio test, we just need to get the 2D points, so we'll extract them differently
        if len(knn_matches) == 0:
            continue
            
        # Apply ratio test manually since we don't have cv2.KeyPoint objects for cam
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair[0], match_pair[1]
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_camera_match = cam
            best_matches = good_matches
            best_kp_new = kp_new

        # Get keypoints from camera (we need to map descriptors back to keypoints)
        # For now, we'll use a simpler approach: match and then find 2D-3D correspondences
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_camera_match = cam
            best_matches = good_matches
            best_kp_new = kp_new

    if best_camera_match is None or max_matches < 4:
        return scene  # Not enough matches

    # Build 2D-3D correspondences for PnP.
    #
    # `best_matches` are descriptor matches between the new image and
    # `best_camera_match`. Each match links:
    #   - new image keypoint index:  match.queryIdx
    #   - existing camera keypoint index: match.trainIdx
    #
    # The existing camera already has observations that tie some of its
    # keypoints to 3D points in the scene. We recover 2D-3D pairs by:
    #   1) For the matched keypoint on the existing camera, find the
    #      closest observation (same camera) in pixel space.
    #   2) If that observation is within a small pixel threshold, we
    #      treat it as the corresponding 3D point.
    points_3d_list = []
    points_2d_list = []
    matched_point_ids = {}  # Ensure we only use each 3D point once.

    # Collect all observations for the camera we are matching against.
    obs_uvs = []
    obs_point_ids = []
    for obs in scene.observations:
        if obs.camera_id == best_camera_match.id:
            obs_uvs.append(obs.uv)
            obs_point_ids.append(obs.point_id)

    if not obs_uvs:
        return scene  # No existing observations to link to 3D points

    obs_uvs = np.array(obs_uvs, dtype=np.float32)  # (M, 2)
    obs_point_ids = np.array(obs_point_ids, dtype=int)  # (M,)

    # Maximum allowed pixel distance between a matched keypoint and an
    # observation on the existing camera for them to be considered the
    # same feature.
    max_uv_dist = 2.0

    for match in best_matches:
        if match.trainIdx >= len(best_camera_match.keypoints):
            continue

        # 2D point in the existing camera and in the new image.
        pt_cam = best_camera_match.keypoints[match.trainIdx]  # (2,)
        pt_new = best_kp_new[match.queryIdx].pt

        pt_cam = np.array(pt_cam, dtype=np.float32)

        # Find closest observation on this camera in pixel space.
        diffs = obs_uvs - pt_cam[None, :]
        dists_sq = np.sum(diffs * diffs, axis=1)
        j = int(np.argmin(dists_sq))

        if dists_sq[j] > max_uv_dist * max_uv_dist:
            # No sufficiently close observation; skip this match.
            continue

        point_id = int(obs_point_ids[j])
        if point_id in matched_point_ids:
            # Already used this 3D point.
            continue

        matched_point_ids[point_id] = len(points_2d_list)
        points_3d_list.append(scene.points3d[point_id].xyz)
        points_2d_list.append(pt_new)

    points_3d_array = np.array(points_3d_list) if points_3d_list else np.array([]).reshape(0, 3)
    points_2d_array = np.array(points_2d_list) if points_2d_list else np.array([]).reshape(0, 2)

    if len(points_3d_array) < 4:
        return scene  # Not enough 2D-3D correspondences for PnP

    # Estimate new camera pose using PnP
    R_new, t_new, inlier_mask = estimate_camera_pose_pnp(K, points_3d_array, points_2d_array)

    num_pnp_inliers = int(np.sum(inlier_mask))
    if num_pnp_inliers < 30:
        print(
            f"[sfm] Rejecting new camera at image_idx={image_idx}: "
            f"only {num_pnp_inliers} PnP inliers"
        )
        return scene  # PnP pose not reliable enough

    # Create new camera
    new_camera_id = len(scene.cameras)
    new_camera = Camera(
        id=new_camera_id,
        K=K,
        R=R_new,
        t=t_new,
        image_idx=image_idx,
        keypoints=np.array([kp.pt for kp in best_kp_new], dtype=np.float32),
        descriptors=desc_new,
    )
    scene.cameras.append(new_camera)

    # Add observations for matched EXISTING 3D points (PnP inliers only).
    inlier_point_ids = [
        list(matched_point_ids.keys())[i]
        for i in range(len(matched_point_ids))
        if inlier_mask[i]
    ]
    inlier_points_2d = points_2d_array[inlier_mask]

    for i, point_id in enumerate(inlier_point_ids):
        obs_idx = len(scene.observations)
        obs = Observation(camera_id=new_camera_id, point_id=point_id, uv=inlier_points_2d[i])
        scene.observations.append(obs)
        scene.points3d[point_id].observations.append(obs_idx)

    # ------------------------------------------------------------------
    # Triangulate NEW 3D points from unmatched image features.
    #
    # Strategy:
    #   - Reuse descriptor matches between the new camera and
    #     `best_camera_match`.
    #   - For each 2D–2D match, check if either side is already
    #     associated (within a small pixel threshold) with an existing
    #     3D point observation. If so, skip it (that feature already has
    #     a 3D point).
    #   - For truly "unclaimed" matches, triangulate between the two
    #     camera poses to create new 3D points and add observations in
    #     both cameras.
    # ------------------------------------------------------------------

    # Gather existing observations for both cameras.
    cam_obs_uvs = []
    cam_obs_point_ids = []
    new_obs_uvs = []
    new_obs_point_ids = []

    for idx, obs in enumerate(scene.observations):
        if obs.camera_id == best_camera_match.id:
            cam_obs_uvs.append(obs.uv)
            cam_obs_point_ids.append(obs.point_id)
        elif obs.camera_id == new_camera_id:
            new_obs_uvs.append(obs.uv)
            new_obs_point_ids.append(obs.point_id)

    cam_obs_uvs = np.array(cam_obs_uvs, dtype=np.float32) if cam_obs_uvs else np.zeros((0, 2), dtype=np.float32)
    new_obs_uvs = np.array(new_obs_uvs, dtype=np.float32) if new_obs_uvs else np.zeros((0, 2), dtype=np.float32)

    # Pixel-distance threshold for deciding whether a keypoint already
    # has a 3D point observation attached.
    existing_obs_dist = 2.0
    existing_obs_dist_sq = existing_obs_dist * existing_obs_dist

    new_match_pts_cam = []
    new_match_pts_new = []

    for match in best_matches:
        if match.trainIdx >= len(best_camera_match.keypoints):
            continue

        pt_cam = np.array(best_camera_match.keypoints[match.trainIdx], dtype=np.float32)
        pt_new = np.array(best_kp_new[match.queryIdx].pt, dtype=np.float32)

        # Skip if this feature on the existing camera is already tied to
        # a 3D point (observation very close in pixel space).
        if cam_obs_uvs.shape[0] > 0:
            diffs_cam = cam_obs_uvs - pt_cam[None, :]
            if np.min(np.sum(diffs_cam * diffs_cam, axis=1)) <= existing_obs_dist_sq:
                continue

        # Skip if this feature on the new camera is already tied to a
        # 3D point (should be rare, but be safe).
        if new_obs_uvs.shape[0] > 0:
            diffs_new = new_obs_uvs - pt_new[None, :]
            if np.min(np.sum(diffs_new * diffs_new, axis=1)) <= existing_obs_dist_sq:
                continue

        new_match_pts_cam.append(pt_cam)
        new_match_pts_new.append(pt_new)

    if len(new_match_pts_cam) >= 8:
        pts_cam_np = np.array(new_match_pts_cam, dtype=np.float32)
        pts_new_np = np.array(new_match_pts_new, dtype=np.float32)

        # Triangulate between best_camera_match and the new camera.
        points_3d_new, reproj_errors_new = triangulate_matched_key_pts_to_3D_pts(
            K,
            best_camera_match.R,
            best_camera_match.t,
            new_camera.R,
            new_camera.t,
            pts_cam_np,
            pts_new_np,
        )

        # Filter out points with high reprojection error, behind cameras,
        # or with extreme distance relative to the typical scale of this
        # incremental batch.
        z_cam = (best_camera_match.R @ points_3d_new.T + best_camera_match.t).T[:, 2]
        z_new = (new_camera.R @ points_3d_new.T + new_camera.t).T[:, 2]

        mask_z = (z_cam > 0) & (z_new > 0)
        # Tight reprojection error threshold (in pixels).
        mask_err = reproj_errors_new < 5.0

        # Norm-based sanity check: reject points far beyond the median radius.
        norms = np.linalg.norm(points_3d_new, axis=1)
        median_norm = np.median(norms) if norms.size > 0 else 0.0
        if median_norm > 0:
            max_allowed_norm = 5.0 * median_norm
            mask_norm = norms < max_allowed_norm
        else:
            mask_norm = np.ones_like(norms, dtype=bool)

        valid_mask = mask_z & mask_err & mask_norm

        num_valid_new = int(np.sum(valid_mask))
        print(
            f"[sfm] Incremental triangulation between cam {best_camera_match.id} and {new_camera_id}: "
            f"{len(points_3d_new)} candidates, {num_valid_new} valid; "
            f"norms min={norms.min():.2f}, median={median_norm:.2f}, max={norms.max():.2f}"
        )

        if num_valid_new > 0:
            pts_cam_valid = pts_cam_np[valid_mask]
            pts_new_valid = pts_new_np[valid_mask]
            points_3d_valid = points_3d_new[valid_mask]

            for i, pt_3d in enumerate(points_3d_valid):
                new_point_id = len(scene.points3d)

                # Sample color from the NEW image at the new-camera pixel.
                u, v = int(pts_new_valid[i][0]), int(pts_new_valid[i][1])
                if 0 <= v < new_image.shape[0] and 0 <= u < new_image.shape[1]:
                    color = new_image[v, u].astype(np.uint8)
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)

                point3d = Point3D(
                    id=new_point_id,
                    xyz=pt_3d,
                    color=color,
                    observations=[],
                )
                scene.points3d.append(point3d)

                # Observations in both cameras.
                obs_idx_cam = len(scene.observations)
                obs_cam = Observation(
                    camera_id=best_camera_match.id,
                    point_id=new_point_id,
                    uv=pts_cam_valid[i],
                )
                scene.observations.append(obs_cam)

                obs_idx_new = len(scene.observations)
                obs_new = Observation(
                    camera_id=new_camera_id,
                    point_id=new_point_id,
                    uv=pts_new_valid[i],
                )
                scene.observations.append(obs_new)

                point3d.observations.extend([obs_idx_cam, obs_idx_new])

    else:
        print(
            f"[sfm] Not enough candidate matches for incremental triangulation "
            f"between cam {best_camera_match.id} and {new_camera_id}: "
            f"{len(new_match_pts_cam)}"
        )

    return scene


def run_sfm_from_frames(
    frames: List[np.ndarray],
    K: np.ndarray,
    keyframe_indices: List[int],
) -> SceneGraph:
    """
    Run incremental SfM on a sequence of keyframes.

    Args:
        frames: List of all frame images.
        K: Intrinsic camera matrix (3x3).
        keyframe_indices: List of frame indices to use as keyframes.

    Returns:
        SceneGraph containing all cameras, 3D points, and observations.
    """
    if len(keyframe_indices) < 2:
        return SceneGraph()

    # Choose base keyframes from the provided keyframe_indices rather than
    # using hard-coded timestamps. By default we use:
    #   - the first keyframe, and
    #   - the 5th keyframe if it exists, otherwise the last keyframe.
    i0 = keyframe_indices[0]

    # ------------------------------------------------------------
    # Helper to count 3D points in a SceneGraph robustly
    # ------------------------------------------------------------
    def count_points(scene: SceneGraph) -> int:
        if hasattr(scene, "points3d") and scene.points3d is not None:
            return len(scene.points3d)
        if hasattr(scene, "points") and scene.points is not None:
            return len(scene.points)
        return 0

    # ------------------------------------------------------------
    # Candidate second keyframe: fixed offset from the first.
    #
    # We choose the base pair to be:
    #   - the earliest keyframe (i0), and
    #   - the keyframe that is 5 positions later in the keyframe list,
    #     or the last keyframe if there are fewer than 6 total.
    #
    # This approximates "first frame and first + 5 frames" under the
    # keyframe subsampling used by the CLI.
    # ------------------------------------------------------------
    if len(keyframe_indices) > 5:
        candidate_indices = [keyframe_indices[5]]
    else:
        i1 = keyframe_indices[-1]

    print(f"[sfm] Using base keyframes {i0} and {i1} for initial reconstruction")

    # Build base scene from two frames
    scene = build_base_scene_from_two_frames(frames, i0, i1, K)

    if len(scene.cameras) == 0:
        return scene

    # Add remaining keyframes incrementally (all others).
    remaining_indices = [idx for idx in keyframe_indices if idx not in (i0, i1)]

    for idx in remaining_indices:
        scene = add_camera_incremental(scene, K, frames[idx], idx)

    return scene


__all__ = [
    "build_base_scene_from_two_frames",
    "add_camera_incremental",
    "run_sfm_from_frames",
]

