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

    # Map each valid triangulated point back to the original keypoint
    # indices in kp1 / kp2 via the surviving matches.
    # good_matches_inliers: matches after F-RANSAC.
    good_matches_inliers = [good_matches[i] for i in range(len(good_matches)) if inlier_mask[i]]
    pose_matches = [good_matches_inliers[i] for i in range(len(good_matches_inliers)) if pose_mask[i]]
    # Now apply the same valid_mask used on points_3d.
    feature_idx_cam0 = np.array(
        [m.queryIdx for i, m in enumerate(pose_matches) if valid_mask[i]],
        dtype=int,
    )
    feature_idx_cam1 = np.array(
        [m.trainIdx for i, m in enumerate(pose_matches) if valid_mask[i]],
        dtype=int,
    )

    # Create SceneGraph
    scene = SceneGraph()

    # Full keypoint arrays for both base cameras.
    kp1_all = np.array([kp.pt for kp in kp1], dtype=np.float32)
    kp2_all = np.array([kp.pt for kp in kp2], dtype=np.float32)
    point_ids_cam0 = -np.ones(len(kp1_all), dtype=int)
    point_ids_cam1 = -np.ones(len(kp2_all), dtype=int)

    # Camera 0 (first frame)
    cam0 = Camera(
        id=0,
        K=K,
        R=R1,
        t=t1,
        image_idx=frame_idx1,
        keypoints=kp1_all,
        descriptors=desc1,
        point_ids=point_ids_cam0,
    )
    scene.cameras.append(cam0)

    # Camera 1 (second frame)
    cam1 = Camera(
        id=1,
        K=K,
        R=R2,
        t=t2,
        image_idx=frame_idx2,
        keypoints=kp2_all,
        descriptors=desc2,
        point_ids=point_ids_cam1,
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

        point_id = len(scene.points3d)
        point3d = Point3D(
            id=point_id,
            xyz=pt_3d,
            color=color,
            observations=[],
        )
        scene.points3d.append(point3d)

        # Record which keypoints (by index) observe this 3D point.
        k0 = feature_idx_cam0[i]
        k1 = feature_idx_cam1[i]
        scene.cameras[0].point_ids[k0] = point_id
        scene.cameras[1].point_ids[k1] = point_id

        # Create observations
        obs0_idx = len(scene.observations)
        obs1_idx = len(scene.observations) + 1

        obs0 = Observation(camera_id=0, point_id=point_id, uv=pts1_valid[i])
        obs1 = Observation(camera_id=1, point_id=point_id, uv=pts2_valid[i])

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

    # Build 2D-3D correspondences for PnP directly from feature-level
    # mappings. `best_camera_match.point_ids[idx]` gives the 3D point id
    # (or -1 if none) for the matched descriptor at `idx`.
    points_3d_list = []
    points_2d_list = []
    matched_point_ids: list[int] = []
    matched_query_indices: list[int] = []
    used_point_ids = set()

    for match in best_matches:
        pid = int(best_camera_match.point_ids[match.trainIdx]) if len(best_camera_match.point_ids) > 0 else -1
        if pid < 0:
            continue  # this feature on the existing camera has no 3D point yet
        if pid in used_point_ids:
            continue  # avoid duplicate 2D-3D pairs for the same 3D point

        used_point_ids.add(pid)
        points_3d_list.append(scene.points3d[pid].xyz)
        points_2d_list.append(best_kp_new[match.queryIdx].pt)
        matched_point_ids.append(pid)
        matched_query_indices.append(match.queryIdx)

    points_3d_array = np.array(points_3d_list) if points_3d_list else np.array([]).reshape(0, 3)
    points_2d_array = np.array(points_2d_list) if points_2d_list else np.array([]).reshape(0, 2)

    if len(points_3d_array) < 4:
        return scene  # Not enough 2D-3D correspondences for PnP

    # Estimate new camera pose using PnP
    R_new, t_new, inlier_mask = estimate_camera_pose_pnp(K, points_3d_array, points_2d_array)

    num_pnp_inliers = int(np.sum(inlier_mask))
    if num_pnp_inliers < 10:
        print(
            f"[sfm] Rejecting new camera at image_idx={image_idx}: "
            f"only {num_pnp_inliers} PnP inliers"
        )
        return scene  # PnP pose not reliable enough

    # Create new camera
    new_camera_id = len(scene.cameras)
    new_kp_array = np.array([kp.pt for kp in best_kp_new], dtype=np.float32)
    new_point_ids = -np.ones(len(new_kp_array), dtype=int)
    new_camera = Camera(
        id=new_camera_id,
        K=K,
        R=R_new,
        t=t_new,
        image_idx=image_idx,
        keypoints=new_kp_array,
        descriptors=desc_new,
        point_ids=new_point_ids,
    )
    scene.cameras.append(new_camera)

    # Add observations for matched EXISTING 3D points (PnP inliers only),
    # and mark the corresponding new-image keypoints as observing those
    # 3D points.
    inlier_indices = np.where(inlier_mask)[0]
    for idx in inlier_indices:
        pid = matched_point_ids[idx]
        uv = np.array(points_2d_array[idx], dtype=np.float32)

        obs_idx = len(scene.observations)
        obs = Observation(camera_id=new_camera_id, point_id=pid, uv=uv)
        scene.observations.append(obs)
        scene.points3d[pid].observations.append(obs_idx)

        k_new = matched_query_indices[idx]
        new_camera.point_ids[k_new] = pid

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

    # Build candidate matches for NEW 3D points: require that neither
    # side of the match currently has an associated 3D point.
    new_match_pts_cam = []
    new_match_pts_new = []
    new_kp_idx_cam: list[int] = []
    new_kp_idx_new: list[int] = []

    for match in best_matches:
        pid_cam = int(best_camera_match.point_ids[match.trainIdx]) if len(best_camera_match.point_ids) > 0 else -1
        pid_new = int(new_camera.point_ids[match.queryIdx]) if len(new_camera.point_ids) > 0 else -1
        if pid_cam >= 0 or pid_new >= 0:
            continue  # this feature is already tied to a 3D point in at least one view

        pt_cam = np.array(best_camera_match.keypoints[match.trainIdx], dtype=np.float32)
        pt_new = np.array(best_kp_new[match.queryIdx].pt, dtype=np.float32)

        new_match_pts_cam.append(pt_cam)
        new_match_pts_new.append(pt_new)
        new_kp_idx_cam.append(match.trainIdx)
        new_kp_idx_new.append(match.queryIdx)

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
            kp_idx_cam_valid = np.array(new_kp_idx_cam, dtype=int)[valid_mask]
            kp_idx_new_valid = np.array(new_kp_idx_new, dtype=int)[valid_mask]

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

                # Mark which keypoints now observe this 3D point.
                k_cam = kp_idx_cam_valid[i]
                k_new = kp_idx_new_valid[i]
                best_camera_match.point_ids[k_cam] = new_point_id
                new_camera.point_ids[k_new] = new_point_id

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
    # Always use the earliest keyframe as the first base frame (e.g., 0)
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
    # Candidate second keyframes: ALL remaining keyframes
    # ------------------------------------------------------------
    candidate_indices = keyframe_indices[1:]  # this preserves order: 1,2,3,...,N

    print(
        f"[sfm] Evaluating {len(candidate_indices)} candidate base pairs "
        f"starting from frame {i0}..."
    )

    best_scene: Optional[SceneGraph] = None
    best_pair: Optional[Tuple[int, int]] = None
    best_score: int = 0

    # Minimum number of 3D points we consider "usable" for a base scene
    # (You can tweak or even set this to 0 if you want *some* base no matter what.)
    min_points_for_base = 20

    # ------------------------------------------------------------
    # Try each candidate as the second base keyframe: (i0, i1)
    # ------------------------------------------------------------
    for i1 in candidate_indices:
        print(f"[sfm] Trying base keyframes {i0} and {i1} for initial reconstruction...")
        candidate_scene = build_base_scene_from_two_frames(frames, i0, i1, K)

        # If base-building failed completely, skip
        if not hasattr(candidate_scene, "cameras") or len(candidate_scene.cameras) < 2:
            print(f"[sfm]   -> Rejected: < 2 cameras in base scene.")
            continue

        num_pts = count_points(candidate_scene)
        print(
            f"[sfm]   -> Candidate base scene: "
            f"{len(candidate_scene.cameras)} cameras, {num_pts} 3D points"
        )

        # Skip very weak bases
        if num_pts < min_points_for_base:
            continue

        # Keep the best one by number of 3D points
        if num_pts > best_score:
            best_score = num_pts
            best_pair = (i0, i1)
            best_scene = candidate_scene

    # ------------------------------------------------------------
    # Fallback if no strong candidate was found
    # ------------------------------------------------------------
    if best_scene is None:
        # Fall back to the first two keyframes (i0, keyframe_indices[1])
        fallback_i1 = keyframe_indices[1]
        print(
            "[sfm] WARNING: Could not find a strong base pair; "
            f"falling back to first two keyframes ({i0}, {fallback_i1})."
        )
        fallback_scene = build_base_scene_from_two_frames(frames, i0, fallback_i1, K)

        if not hasattr(fallback_scene, "cameras") or len(fallback_scene.cameras) < 2:
            print("[sfm] ERROR: Base scene failed even with fallback pair.")
            return fallback_scene

        best_scene = fallback_scene
        best_pair = (i0, fallback_i1)
        best_score = count_points(best_scene)

    i0, i1 = best_pair
    print(
        f"[sfm] Using base keyframes {i0} and {i1} for initial reconstruction "
        f"(score = {best_score} 3D points)"
    )

    scene = best_scene

    # ------------------------------------------------------------
    # Incrementally add remaining keyframes
    # ------------------------------------------------------------
    remaining_indices = [idx for idx in keyframe_indices if idx not in (i0, i1)]

    for idx in remaining_indices:
        scene = add_camera_incremental(scene, K, frames[idx], idx)

    return scene

__all__ = [
    "build_base_scene_from_two_frames",
    "add_camera_incremental",
    "run_sfm_from_frames",
]

