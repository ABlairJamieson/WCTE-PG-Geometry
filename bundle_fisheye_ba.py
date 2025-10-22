"""
bundle_fisheye_ba.py

Joint bundle-adjustment for fisheye cameras using OpenCV fisheye projection
and scipy.optimize.least_squares.

Requirements:
    pip install numpy scipy opencv-contrib-python

Usage:
 - Populate `input_image_names`, `label_to_xyz`, `K_per_cam`, `D_per_cam` in the __main__ section
 - Run the script. It will print progress and save a results JSON.

Notes:
 - By default intrinsics/distortion are held fixed (use `optimize_intrinsics=True` to enable a
   simple shared-intrinsics optimization — not recommended on the first pass).
"""

import json
import numpy as np
import cv2
from scipy.optimize import least_squares, OptimizeResult
import os
from collections import defaultdict
import math
import sys
import time
import copy
from PredictImage import project_leds, get_camera_orientation
from functools import partial
import matplotlib.pyplot as plt

# -------------------------
# Helpers for parsing JSON
# -------------------------
def get_2d_from_match_entry(entry):
    """Robustly extract a 2D pixel coordinate from typical JSON forms."""
    if entry is None:
        return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return float(entry[0]), float(entry[1])
    if isinstance(entry, dict):
        for kx, ky in (('x','y'), ('u','v'), ('col','row'), ('px','py'), ('c','r')):
            if kx in entry and ky in entry:
                return float(entry[kx]), float(entry[ky])
        if 'pt' in entry and isinstance(entry['pt'], (list, tuple)) and len(entry['pt']) >= 2:
            return float(entry['pt'][0]), float(entry['pt'][1])
    # fallback - try to convert numeric-like strings
    try:
        xv = float(entry)
        return (xv, 0.0)
    except Exception:
        return None

# -------------------------
# Build observation list
# -------------------------
def build_observations(input_image_names, label_to_index):
    """
    Returns:
      observations: list of tuples (cam_idx, point_idx, u, v)
      cam_to_obs: dict cam_idx -> list of obs indices
      point_to_obs: dict point_idx -> list of obs indices
    """
    observations = []
    cam_to_obs = defaultdict(list)
    point_to_obs = defaultdict(list)

    for cam_idx, info in input_image_names.items():
        fname = info.get('filename')
        if fname is None:
            continue
        if not os.path.exists(fname):
            raise FileNotFoundError(f"JSON file for camera {cam_idx} not found: {fname}")
        with open(fname, 'r') as f:
            data = json.load(f)
        matches = data.get('matches', {})
        for label, entry in matches.items():
            if label not in label_to_index:
                continue
            xy = get_2d_from_match_entry(entry)
            if xy is None:
                continue
            pt_idx = label_to_index[label]
            obs_idx = len(observations)
            observations.append((cam_idx, pt_idx, float(xy[0]), float(xy[1])))
            cam_to_obs[cam_idx].append(obs_idx)
            point_to_obs[pt_idx].append(obs_idx)
    return observations, cam_to_obs, point_to_obs

# -------------------------
# Pack / unpack parameters
# -------------------------
def pack_params(rvecs, tvecs, points):
    keys = sorted(rvecs.keys())
    cam_params = np.concatenate([
        np.hstack([rvecs[k].ravel(), tvecs[k].ravel()])
        for k in keys
    ])
    points_flat = points.ravel()
    return np.concatenate([cam_params, points_flat])

def unpack_params(x, cam_keys, n_points):
    n_cams = len(cam_keys)
    cam_params = x[:n_cams*6].reshape(n_cams, 6)
    points = x[n_cams*6:].reshape(n_points, 3)

    rvecs = {k: cam_params[i, :3] for i, k in enumerate(cam_keys)}
    tvecs = {k: cam_params[i, 3:] for i, k in enumerate(cam_keys)}
    return rvecs, tvecs, points


def project_points_fisheye_ba(points3d, rvec, cam_pos, K, D,
                              fov_limit_deg=180, ylims=(-1.1, 1.9)):
    """
    Fisheye projection for bundle adjustment using the same math
    as PredictImage.project_leds, but self-contained and simplified.
    """
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    N = points3d.shape[0]
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    cam_pos = np.asarray(cam_pos, dtype=np.float64).ravel()

    # --- Filter in world coordinates (same as project_leds) ---
    keep_mask = (points3d[:, 1] > ylims[0]) & (points3d[:, 1] < ylims[1])
    points3d = points3d[keep_mask]
    N_filtered = points3d.shape[0]

    # --- Transform to camera frame ---
    R_cam, _ = cv2.Rodrigues(rvec)
    pts_cam = (R_cam.T @ (points3d - cam_pos).T).T

    # --- Forward-facing points only ---
    visible_mask = pts_cam[:, 2] > 0

    projected = np.full((N_filtered, 2), np.nan, dtype=np.float64)
    if not np.any(visible_mask):
        # No valid projections
        full_mask = np.zeros_like(keep_mask)
        return projected, full_mask

    # --- Project visible points using fisheye model ---
    pts_to_project = pts_cam[visible_mask].reshape(-1, 1, 3)
    img_pts, _ = cv2.fisheye.projectPoints(pts_to_project,
                                           np.zeros(3), np.zeros(3),
                                           K, D)
    img_pts = img_pts.reshape(-1, 2)

    # --- Optional FOV limit ---
    if fov_limit_deg is not None:
        theta = np.arccos(
            pts_cam[visible_mask, 2] / np.linalg.norm(pts_cam[visible_mask], axis=1)
        )
        fov_mask = np.degrees(theta) < fov_limit_deg
    else:
        fov_mask = np.ones_like(visible_mask[visible_mask], dtype=bool)

    # --- Merge masks (fixed indexing) ---
    full_mask = np.zeros_like(keep_mask, dtype=bool)

    # Get indices of all points that passed keep_mask
    visible_idx = np.where(keep_mask)[0][visible_mask]
    
    # Apply FOV filtering to those indices
    visible_idx = visible_idx[fov_mask]

    # Update mask and projected array in one consistent indexing step
    full_mask[visible_idx] = True
    projected[visible_idx] = img_pts[fov_mask]

    return projected, full_mask


def make_snapshot_callback(build_ba_results, cam_keys, filtered_labels,
                           label_to_xyz_design, snapshot_prefix, snapshot_every,
                           res_fun):
    call_counter = {'count': 0}

    def snapshot_callback(x, *args, **kwargs):
        call_counter['count'] += 1
        count = call_counter['count']

        if count % snapshot_every == 0:
            # Compute RMSE using the residual function
            residuals = res_fun(x)
            rmse_pixels = np.sqrt(np.mean(residuals ** 2))

            print(f"\nIteration {count}: RMSE = {rmse_pixels:.3f} px")

            # Optionally still store a snapshot of current parameter state
            snapshot = build_ba_results(
                x, cam_keys, filtered_labels, label_to_xyz_design
            )
            snapshot["rmse_pixels"] = rmse_pixels
            save_ba_results_json(snapshot, f"{snapshot_prefix}{count}.json")

    return snapshot_callback



def residuals_flat(
    x,
    obs_cam_keys,
    obs_point_indices,
    obs_uv,
    K_per_cam,
    D_per_cam,
    cam_rvecs,
    cam_tvecs,
    points3d,
    points3d_free_mask,
    filtered_labels,
    label_to_xyz_design,
    fov_limit_deg=180.0,
    penalty_for_lost_point=1e3
):
    """
    Residuals function for bundle adjustment, top-level for pickling.
    Includes optional snapshots every snapshot_every calls.
    """
    cam_keys_sorted = sorted(cam_rvecs.keys())

    # --- Unpack parameters ---
    cam_rvecs_opt, cam_tvecs_opt, points_flat = unpack_params(
        x, cam_keys_sorted, points3d.shape[0]
    )

    points3d_opt = points3d.copy()
    points3d_opt[points3d_free_mask] = points_flat

    residual_list = []

    #penalty_count = {k:0 for k in cam_keys_sorted}
    #total_count = {k:0 for k in cam_keys_sorted}

    for obs_idx, (cam_key, pt_idx) in enumerate(zip(obs_cam_keys, obs_point_indices)):
        rvec = cam_rvecs_opt[cam_key]
        tvec = cam_tvecs_opt[cam_key]
        K = K_per_cam[cam_key]
        D = D_per_cam.get(cam_key, np.zeros((4,1),dtype=np.float64))
        pt3d = points3d_opt[pt_idx:pt_idx+1]

        proj_pt, valid_mask = project_points_fisheye_ba(
            pt3d, rvec, tvec, K, D, fov_limit_deg=fov_limit_deg
        )

        #total_count[cam_key] += 1
        if (not np.any(valid_mask)) or np.any(np.isnan(proj_pt[0])):
            residual_list.extend([penalty_for_lost_point, penalty_for_lost_point])
            #penalty_count[cam_key] += 1
            continue

        residual_list.extend(obs_uv[obs_idx] - proj_pt[0])

   
    return np.array(residual_list, dtype=np.float64)


def save_ba_results_json(ba_results, filename):
    """
    Save BA results dict to JSON, safely converting NumPy arrays,
    scipy OptimizeResult, and nested objects into JSON-serializable types.
    """
    def make_json_safe(obj):
        """Recursively convert any object into JSON-safe types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        elif isinstance(obj, OptimizeResult):
            return {k: make_json_safe(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {str(k): make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            # Fallback for objects with attributes
            return {k: make_json_safe(v) for k, v in obj.__dict__.items()}
        else:
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)

    # Convert results to JSON-safe format
    ba_results_jsonsafe = make_json_safe(ba_results)

    # Save to disk
    with open(filename, "w") as f:
        json.dump(ba_results_jsonsafe, f, indent=2)
    print(f"Results saved to {filename}")

def run_bundle_adjustment_filtered(
    input_image_names,
    label_to_xyz_design,
    filtered_observations,
    initial_camera_poses,
    initial_points3d=None,
    K_per_cam=None,
    D_per_cam=None,
    method='trf',
    verbose=True
):
    cam_keys = sorted(initial_camera_poses.keys())
    Nc = len(cam_keys)

    #print('initial_camera_poses=',initial_camera_poses)
    # --- Points ---
    filtered_labels = list(sorted(label_to_xyz_design.keys()))
    Np = len(filtered_labels)
    pts3d_design_compact = np.array([label_to_xyz_design[l] for l in filtered_labels], dtype=float)

    # --- Observations ---
    obs_cam_keys = [obs[0] for obs in filtered_observations]
    obs_point_indices = [obs[1] for obs in filtered_observations]
    obs_uv = np.array([[obs[2], obs[3]] for obs in filtered_observations], dtype=float)

    # --- Initialize camera dicts ---
    rvecs = {k: np.asarray(initial_camera_poses[k][0], dtype=float).ravel() for k in cam_keys}
    tvecs = {k: np.asarray(initial_camera_poses[k][1], dtype=float).ravel() for k in cam_keys}

    cam_keys = sorted(rvecs.keys())
    #print('tvecs=',tvecs)
    #print('rvecs=',rvecs)
    
    # --- Initial 3D points ---
    if initial_points3d is None:
        points3d = pts3d_design_compact.copy()
    elif isinstance(initial_points3d, dict):
        points3d = np.array([initial_points3d.get(l, label_to_xyz_design[l]) for l in filtered_labels], dtype=float)
    else:
        points3d = np.asarray(initial_points3d, dtype=float)

    points3d_free_mask = np.ones(len(points3d), dtype=bool)

    # --- Residual function ---
    residual_func = partial(
        residuals_flat,
        obs_cam_keys=obs_cam_keys,
        obs_point_indices=obs_point_indices,
        obs_uv=obs_uv,
        K_per_cam=K_per_cam,
        D_per_cam=D_per_cam,
        cam_rvecs=rvecs,
        cam_tvecs=tvecs,
        points3d=points3d,
        points3d_free_mask=points3d_free_mask,
        filtered_labels=filtered_labels,
        label_to_xyz_design=label_to_xyz_design
    )

    # --- Snapshot callback with RMSE printing ---
    snapshot_callback = make_snapshot_callback(
        build_ba_results=build_ba_results,
        cam_keys=cam_keys,
        filtered_labels=filtered_labels,
        label_to_xyz_design=label_to_xyz_design,
        snapshot_prefix="bundle_fisheye_snapshot_",
        snapshot_every=10,
        res_fun=residual_func
    )
    
    # --- Pack initial parameters ---
    x0_arr = pack_params(rvecs, tvecs, points3d)

    # --- Bounds ---
    lb = -np.inf * np.ones_like(x0_arr)
    ub = np.inf * np.ones_like(x0_arr)
    max_rot = np.deg2rad(3.0)
    max_trans = 0.03
    max_point_move = 0.1

    # Camera rotations + translations bounds
    offset = 0
    for k in cam_keys:
        lb[offset:offset+3] = x0_arr[offset:offset+3] - max_rot
        ub[offset:offset+3] = x0_arr[offset:offset+3] + max_rot
        offset += 3
    for k in cam_keys:
        lb[offset:offset+3] = x0_arr[offset:offset+3] - max_trans
        ub[offset:offset+3] = x0_arr[offset:offset+3] + max_trans
        offset += 3

    # Point bounds
    for i in range(Np):
        lb[offset:offset+3] = x0_arr[offset:offset+3] - max_point_move
        ub[offset:offset+3] = x0_arr[offset:offset+3] + max_point_move
        offset += 3

    # --- Least squares ---
    ls_kwargs = dict(
        fun=residual_func,
        x0=x0_arr,
        jac='2-point',
        verbose=2 if verbose else 0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=50000,
        bounds=(lb, ub),
        workers=18,  # <--- use 18 CPU threads, change to 1 or remove to run serially
        callback=snapshot_callback 
    )

    if method == 'lm':
        ls_kwargs.pop('bounds')
        res = least_squares(**ls_kwargs, method='lm')
    else:
        ls_kwargs.update(dict(method='trf', loss='huber', f_scale=1.0))
        #print('ls_kwargs=',ls_kwargs)
        res = least_squares(**ls_kwargs)
    
    # --- Return final results ---
    ba_results = build_ba_results(res.x, cam_keys, filtered_labels, label_to_xyz_design, res_fun=res)
    save_ba_results_json(ba_results, "bundle_adjustment_results.json")
    return ba_results


def build_ba_results(x, cam_keys, filtered_labels, label_to_xyz_design, res_fun=None):
    """
    Build full bundle adjustment results dictionary including fitted points,
    camera poses, deltas, summary statistics, and uncertainties.

    Parameters
    ----------
    x : ndarray
        Current parameter vector (camera poses + 3D points)
    cam_keys : list
        List of camera keys
    filtered_labels : list
        List of point labels
    label_to_xyz_design : dict
        Mapping from labels to design 3D coordinates
    res_fun : OptimizeResult or None
        Result object from least_squares, required for covariance/uncertainties

    Returns
    -------
    dict
        Dictionary containing 'res', 'cameras', 'points', 'summary', 'rmse_pixels', etc.
    """
    # --- Unpack parameters ---
    rvecs_opt, tvecs_opt, points3d_opt = unpack_params(x, cam_keys, len(filtered_labels))

    # --- Compute uncertainties if res_fun is given ---
    if res_fun is not None:
        cam_covs, cam_sigmas, point_covs, point_sigmas = compute_ba_uncertainties(
            res_fun, cam_keys, len(filtered_labels)
        )
    else:
        cam_covs = cam_sigmas = point_covs = point_sigmas = [np.zeros((3, 3))]*len(filtered_labels)

    # --- Pack camera results ---
    cam_poses = {
        k: {
            'rvec': rvecs_opt[k].tolist(),
            'tvec': tvecs_opt[k].tolist(),
            'cov': cam_covs[k].tolist() if res_fun is not None else None,
            'sigma': cam_sigmas[k].tolist() if res_fun is not None else None
        }
        for k in cam_keys
    }

    # --- Pack point results ---
    points_out = {}
    deltas = []
    for i, label in enumerate(filtered_labels):
        delta = points3d_opt[i] - np.array(label_to_xyz_design[label])
        delta_norm = float(np.linalg.norm(delta))
        deltas.append(delta_norm)

        points_out[label] = {
            'fitted_xyz': points3d_opt[i].tolist(),
            'design_xyz': label_to_xyz_design[label],
            'delta_xyz': delta.tolist(),
            'delta_norm_m': delta_norm,
            'cov_xyz': point_covs[i].tolist() if res_fun is not None else None,
            'sigma_xyz': point_sigmas[i].tolist() if res_fun is not None else None
        }

    # --- Summary statistics ---
    summary = {
        'mean_delta_m': float(np.mean(deltas)),
        'median_delta_m': float(np.median(deltas)),
        'max_delta_m': float(np.max(deltas)),
        'rmse_pixels': float(np.sqrt(np.mean(res_fun.fun**2))) if res_fun is not None else None,
        'n_points': len(filtered_labels),
        'n_cameras': len(cam_keys)
    }

    # --- Return packed results ---
    return {
        'res': res_fun,
        'cameras': cam_poses,
        'points': points_out,
        'rmse_pixels': summary['rmse_pixels'],
        'summary': summary,
        'filtered_labels': filtered_labels,
        'n_observations': len(filtered_labels)  # optionally pass real obs count if needed
    }


def compute_ba_uncertainties(res, cam_keys, n_points):
    """
    Compute per-camera and per-point uncertainties from least_squares result.

    Returns:
        cam_covs: dict, each cam key -> 6x6 covariance (rot + trans)
        cam_sigmas: dict, each cam key -> 6-vector standard deviations
        point_covs: (n_points,3,3) covariance matrices
        point_sigmas: (n_points,3) standard deviations
    """
    J = res.jac
    n_res, n_params = J.shape
    sigma2 = np.sum(res.fun**2) / (n_res - n_params)

    try:
        cov_params = np.linalg.inv(J.T @ J) * sigma2
    except np.linalg.LinAlgError:
        cov_params = np.linalg.pinv(J.T @ J) * sigma2

    n_cams = len(cam_keys)
    cam_covs = {}
    cam_sigmas = {}
    for i, k in enumerate(cam_keys):
        idx = 6*i
        cam_covs[k] = cov_params[idx:idx+6, idx:idx+6]
        cam_sigmas[k]   = np.sqrt(np.maximum(np.diagonal(cam_covs[k]), 0, where=~np.isnan(np.diagonal(cam_covs[k]))))

    point_covs = np.zeros((n_points, 3, 3))
    point_sigmas = np.zeros((n_points, 3))
    for i in range(n_points):
        idx = n_cams*6 + 3*i
        point_covs[i] = cov_params[idx:idx+3, idx:idx+3]
        point_sigmas[i] = np.sqrt(np.maximum(np.diagonal(point_covs[i]), 0, where=~np.isnan(np.diagonal(point_covs[i]))))

    return cam_covs, cam_sigmas, point_covs, point_sigmas



def build_jacobian_sparsity(observations, Nc, Np):
    """
    Returns a boolean matrix of shape (n_residuals, n_parameters)
    where True indicates which parameters each residual depends on.
    """
    n_res = len(observations)*2 + Nc*6
    n_params = Nc*6 + Np*3
    J_sparsity = np.zeros((n_res, n_params), dtype=bool)

    # --- reprojection residuals ---
    for i, (cam_idx, pt_idx, _, _) in enumerate(observations):
        cam_i = cam_index_map[cam_idx]
        # camera parameters: 6 per camera (tvec + rvec)
        J_sparsity[2*i:2*i+2, cam_i*6 : cam_i*6+6] = True
        # 3D point parameters: 3 per point
        J_sparsity[2*i:2*i+2, 6*Nc + pt_idx*3 : 6*Nc + pt_idx*3 + 3] = True

    # --- camera motion penalty residuals ---
    base_idx = len(observations)*2
    for cam_i in range(Nc):
        # 6 residuals depend only on this camera's 6 params
        J_sparsity[base_idx + cam_i*6 : base_idx + cam_i*6 + 6, cam_i*6 : cam_i*6+6] = True

    return J_sparsity



def safe_draw_circle(img, point, color=(0,255,0), radius=5, thickness=2):
    """Draw a circle if point is valid (non-NaN)."""
    if point is None:
        return
    x, y = point
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return
    cv2.circle(img, (int(round(x)), int(round(y))), radius, color, thickness)

def safe_put_text(img, text, point, color=(0,255,0), font_scale=0.5, thickness=1):
    """Draw text if point is valid."""
    x, y = point
    if x is None or y is None or np.isnan(x) or np.isnan(y):
        return
    cv2.putText(img, str(text), (int(round(x))+4, int(round(y))-4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def plot_module_lines_ordered(modules_coords, modules_labels, color, radius=3.5/2):
    """
    Plot module lines in order of feature number, closing the polygon,
    and avoid drawing lines across the cylinder unroll boundary.
    Uses matplotlib plt.plot (can be drawn on a blank image for debug or saved separately).
    """
    for module, coords in modules_coords.items():
        coords = np.array(coords)
        labels_sorted = np.array(modules_labels[module])
        sort_idx = np.argsort(labels_sorted)
        pts_sorted = coords[sort_idx]

        # Close the polygon
        pts_closed = np.vstack([pts_sorted, pts_sorted[0]])

        # Compute X differences between consecutive points
        diffs = np.abs(np.diff(pts_closed[:, 0]))

        split_idx = np.where(diffs > np.pi * radius)[0]

        segment_start = 0
        for idx in np.append(split_idx, len(pts_closed)-1):
            segment = pts_closed[segment_start:idx+1]
            if len(segment) > 1:
                plt.plot(segment[:,0], segment[:,1], color=color, lw=1.2, alpha=0.5)
            segment_start = idx+1

def plot_fitted_overlay(
    cam_idx,
    image_filename,
    fitted_rvec,
    fitted_cam_pos,
    K,
    D,
    points3d,
    labels,
    observed_uv=None,
    output_filename=None,
    font_scale=0.6,
    thickness=1,
    residual_threshold=10.0,
    fov_limit_deg=180,
    ylims=(-1.1, 1.9)
):
    """
    Overlay fitted points, design points, lines, observed points, and module lines.
    Modules are inferred from label prefixes (e.g., 'pch1_').
    """
    # Load base image
    img = cv2.imread(image_filename)
    if img is None:
        raise FileNotFoundError(f"Could not load image {image_filename}")

    points3d = np.asarray(points3d, dtype=float)
    labels = np.asarray(labels)

    # --- Project points with fisheye
    proj_pts, valid_mask = project_points_fisheye_ba(points3d, fitted_rvec, fitted_cam_pos, K, D,
                                                     fov_limit_deg=fov_limit_deg, ylims=ylims)
    fitted_labels = [lbl for lbl, v in zip(labels, valid_mask) if v]
    proj_pts_valid = proj_pts[valid_mask]

    # --- Draw design -> fitted lines & points
    for i, label in enumerate(labels):
        design_pt = points3d[i]
        design_uv, _ = project_points_fisheye_ba([design_pt], np.zeros(3), np.zeros(3), K, D)
        design_uv = design_uv[0]

        if label not in fitted_labels:
            continue
        fitted_idx = fitted_labels.index(label)
        fitted_uv = proj_pts_valid[fitted_idx]

        # Draw points
        safe_draw_circle(img, design_uv, color=(255,0,0), radius=5, thickness=2)   # blue design
        safe_draw_circle(img, fitted_uv, color=(0,255,0), radius=5, thickness=2)    # green fitted
        safe_put_text(img, label, fitted_uv, color=(0,255,0), font_scale=font_scale, thickness=thickness)

        # Draw line connecting matching points
        if not np.any(np.isnan(design_uv)) and not np.any(np.isnan(fitted_uv)):
            cv2.line(img,
                     (int(round(design_uv[0])), int(round(design_uv[1]))),
                     (int(round(fitted_uv[0])), int(round(fitted_uv[1]))),
                     (200,200,200), 1)

    # --- Draw modules from design points (group by prefix before '_') ---
    modules_coords = {}
    modules_labels = {}
    for lbl, pt in zip(labels, points3d):
        module_name = lbl.split('_')[0]
        if module_name not in modules_coords:
            modules_coords[module_name] = []
            modules_labels[module_name] = []
        modules_coords[module_name].append(pt)
        modules_labels[module_name].append(lbl)

    plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plot_module_lines_ordered(modules_coords, modules_labels, color='cyan')

    plt.axis('off')
    plt.tight_layout()
    if output_filename:
        plt.savefig(output_filename, dpi=100)
        plt.close()
        print(f"Saved overlay: {output_filename}")

    return img
