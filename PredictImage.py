#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from scipy.spatial import cKDTree
from scipy.optimize import minimize, least_squares
import os



def load_geometry(json_path):
    with open(json_path, 'r') as f:
        geo = json.load(f)
    return geo

def select_leds(geo):
    """
    Selects LEDs 6 through 11 from each mPMT using label-based filtering.

    Assumes labels are in format 'MMM-L', e.g., '001-3'
    """
    led_xyz = np.stack([geo['led_x'], geo['led_y'], geo['led_z']], axis=1)
    labels = np.array(geo['labels'])  # e.g. ['000-0', '000-1', ..., '105-8']

    # Filter for LED numbers 3 through 8 (inclusive)
    selected_leds = []
    selected_labels = []
    pmt_numbers = set()

    for i, label in enumerate(labels):
        try:
            pmt_str, led_str = label.split("-")
            pmt = int(pmt_str)
            led = int(led_str)
            if 6 <= led <= 11:
                selected_leds.append(led_xyz[i])
                selected_labels.append(label)
                pmt_numbers.add(pmt)
        except Exception as e:
            print(f"Skipping label {label}: {e}")
            continue

    leds = np.array(selected_leds)
    labels = np.array(selected_labels)
    num_pmts = len(pmt_numbers)

    return leds, labels, num_pmts


def filter_leds_by_position(leds, labels, axis=1, limits=(-1.0, 1.8)): # default takes all
    """
    Filters LEDs along a specified axis (default Y) between two limits.
    eg. limits=(-0.9, 1.8) removes bottom endcap for axis =1 (y-axis)
    """
    keep_mask = (leds[:, axis] >= limits[0]) & (leds[:, axis] <= limits[1])
    return leds[keep_mask], [label for i, label in enumerate(labels) if keep_mask[i]]

def filter_leds_by_pmt_number(leds, labels, excluded_pmts):
    """
    Removes LEDs belonging to specified mPMT module numbers.

    Args:
        leds: (N, 3) array of LED positions
        labels: list of N labels in format 'MMM-LL'
        excluded_pmts: list or set of mPMT numbers (as integers) to exclude

    Returns:
        filtered_leds: (M, 3) array of remaining LED positions
        filtered_labels: list of M corresponding labels
    """
    keep_mask = []
    for label in labels:
        try:
            pmt_number = int(label.split('-')[0])
            keep_mask.append(pmt_number not in excluded_pmts)
        except ValueError:
            # If label parsing fails, exclude it
            keep_mask.append(False)

    keep_mask = np.array(keep_mask)
    return leds[keep_mask], [label for i, label in enumerate(labels) if keep_mask[i]]



def get_camera_orientation(
    geo,
    index,
    delta_pitch_deg=0.0,
    delta_yaw_deg=0.0,
    delta_roll_deg=0.0,
    delta_r=0.0
):
    # Base camera position and basis vectors
    cam_pos = np.array([geo['cam_x'][index], geo['cam_y'][index], geo['cam_z'][index]])
    dx = np.array([geo['cam_dxx'][index], geo['cam_dxy'][index], geo['cam_dxz'][index]])
    dz = np.array([geo['cam_dzx'][index], geo['cam_dzy'][index], geo['cam_dzz'][index]])
    #print('camera index',index,'cam_pos=',cam_pos,'dx=',dx,'dz=',dz)
    
    # Normalize to unit vectors
    dx /= np.linalg.norm(dx)
    dz /= np.linalg.norm(dz)
    dy = np.cross(dz, dx)
    dy /= np.linalg.norm(dy)

    #print('dx=',dx,'dy=',dy,'dz=',dz)
    # Check if camera is mounted at the top (e.g. y > 0), and flip the up and forward directions
    if cam_pos[1] > 0:
        #print(f"[Info] Camera index {index} is upside-down. Adding 180 degree roll.")
        delta_roll_deg += 180
    #    dz = -dz   # flip it to look downward (into detector)
    #    dy = np.cross(dz, dx)
    #    dy /= np.linalg.norm(dy)
    
    # Original camera rotation matrix: columns are local x, y, z
    R = np.vstack([dx, dy, dz]).T

    # Rotation corrections in degrees → radians
    pitch = np.deg2rad(delta_pitch_deg)
    yaw   = np.deg2rad(delta_yaw_deg)
    roll  = np.deg2rad(delta_roll_deg)

    # Pitch: rotate around x-axis (right vector)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

    # Yaw: rotate around y-axis (up vector)
    R_yaw = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Roll: rotate around z-axis (optical axis)
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]
    ])

    # Combine rotations: roll → pitch → yaw (can adjust order if needed)
    R_correction = R_yaw @ R_pitch @ R_roll

    # Final camera rotation matrix
    R_total = R @ R_correction

    # Shift camera position along new optical axis
    dz_corrected = R_total[:, 2]  # local z-axis
    cam_pos_shifted = cam_pos + delta_r * dz_corrected

    return cam_pos_shifted, R_total
    


def project_leds(leds, labels, cam_pos, R_cam, K, D, fov_limit_deg=62.5, ylims=(-0.9,1.8), excluded_pmts=[] ):
    """
    Projects 3D LED points into the image using fisheye distortion and optional FOV cutoff.
    
    Args:
        ...
        fov_limit_deg: optional FOV cutoff (e.g., 62.5 means 125 degrees f.o.v. of forward)

    Returns:
        image_points, labels_valid
    """
    # Mask by physical location (cut off one of endcaps nearest camera)
    leds, labels = filter_leds_by_position( leds, labels, 1, ylims ) 
    leds, labels = filter_leds_by_pmt_number(leds, labels, excluded_pmts)
    
    # Transform to camera frame
    xyz_cam_frame = (R_cam.T @ (leds - cam_pos).T).T

    # Default: forward-facing mask (z > 0)
    valid_mask = xyz_cam_frame[:, 2] > 0
    
    if fov_limit_deg is not None:
        # Angle to optical axis (z = forward in cam frame)
        norms = np.linalg.norm(xyz_cam_frame, axis=1)
        cos_theta = xyz_cam_frame[:, 2] / norms  # cos(angle to z-axis)
        angle_deg = np.degrees(np.arccos(cos_theta))
        angle_mask = angle_deg <= fov_limit_deg
        valid_mask = valid_mask & angle_mask  # combine with z > 0

    xyz_valid = xyz_cam_frame[valid_mask]
    labels = np.array(labels)
    labels_valid = labels[valid_mask]

    if len(xyz_valid) == 0:
        return np.empty((0, 2)), []

    # Project points
    object_points = xyz_valid.reshape(-1, 1, 3)
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    image_points, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
    image_points = image_points.reshape(-1, 2)

    assert image_points.shape[0] == labels_valid.shape[0], "Mismatch after projection"

    return image_points, labels_valid.tolist()

def circle_points(y_val, radius=1.5, num_points=100):
    """
    Generate 3D points on a horizontal circle (x-z plane) at a fixed y-value.

    Returns: (N, 1, 3) array of float64 points suitable for cv2.fisheye.projectPoints
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    y = np.full_like(x, y_val)
    
    points_3d = np.stack((x, y, z), axis=1)  # Shape: (N, 3)
    return points_3d.reshape(-1, 1, 3).astype(np.float64)  # Shape: (N, 1, 3)



def project_points_fisheye(points_3d_world, cam_pos, R_cam, K, D, fov_limit_deg=89.9):
    """
    Projects arbitrary 3D world points into 2D image points using a fisheye camera model.

    Args:
        points_3d_world: (N, 3) array of 3D points in world coordinates
        cam_pos: (3,) array, position of the camera in world coordinates
        R_cam: (3,3) rotation matrix from world to camera orientation
        K: (3,3) camera intrinsic matrix
        D: (4,) distortion coefficients for fisheye
        fov_limit_deg: optional FOV limit in degrees (cone around +Z in camera frame)

    Returns:
        image_points: (M, 2) projected 2D image points (after filtering)
        visible_mask: (N,) boolean array, True for points that were projected
    """
    # Ensure input is np.array
    points_3d_world = np.asarray(points_3d_world)
    points_3d_world = points_3d_world.reshape(-1, 3)  # ensure correct shape

    
    # Transform to camera frame: X_cam = R_cam.T @ (X_world - cam_pos)
    xyz_cam_frame = (R_cam.T @ (points_3d_world - cam_pos).T).T  # shape: (N, 3)

    # Keep only points in front of camera (z > 0)
    valid_mask = xyz_cam_frame[:, 2] > 0

    # Optionally apply FOV cone cut
    if fov_limit_deg is not None:
        norms = np.linalg.norm(xyz_cam_frame, axis=1)
        cos_theta = xyz_cam_frame[:, 2] / norms
        angle_deg = np.degrees(np.arccos(cos_theta))
        angle_mask = angle_deg <= fov_limit_deg
        valid_mask &= angle_mask

    xyz_visible = xyz_cam_frame[valid_mask]

    if xyz_visible.shape[0] == 0:
        return np.empty((0, 2)), valid_mask

    # Project visible points using fisheye model
    object_points = xyz_visible.reshape(-1, 1, 3)
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    image_points, _ = cv2.fisheye.projectPoints(object_points, rvec, tvec, K, D)
    image_points = image_points.reshape(-1, 2)

    return image_points, valid_mask




def draw_cylinder_ellipses(K, D, cam_pos, R_cam, img, 
                           cylinder_radius=1.0, 
                           y_top=1.0, 
                           y_bottom=-0.4, 
                           color=(255, 105, 180), 
                           thickness=2):
    """
    Draw top and bottom ellipses of a vertical cylinder in the image.
    
    Parameters:
    - K: camera intrinsics (3x3)
    - R_cam, cam_pos: rotation and translation vectors (camera pose)
    - img: image to draw on
    - cylinder_radius: in meters
    - y_top, y_bottom: y-values of the top and bottom of the cylinder
    - color: BGR tuple, default is pink
    - thickness: line thickness
    """

    # Generate and project top and bottom circles
    for y_val in [y_top, y_bottom]:
        points_3d = circle_points(y_val,cylinder_radius)
        points_2d, _ = project_points_fisheye(points_3d, cam_pos, R_cam, K, D) 
        poly = points_2d.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness)


def draw_projection(image_points, labels, K, img=None):
    img_height = 6336 #int(K[1, 2] * 2)
    img_width = 9504 #int(K[0, 2] * 2)
    if img is None:
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Determine max PMT number from labels
    pmt_ids = []
    for label in labels:
        try:
            pmt_id = int(label.split("-")[0])
            pmt_ids.append(pmt_id)
        except:
            pass

    max_pmt_id = max(pmt_ids) if pmt_ids else 0
    #print('num_pmts=max_mpmt_id=',max_pmt_id)
    grouped_points = [[] for _ in range(max_pmt_id + 1)]

    for idx, ((u, v), label) in enumerate(zip(image_points, labels)):
        u = int(round(u))
        v = int(round(v))
        if 0 <= u < img_width and 0 <= v < img_height:
            # Draw dot and label
            cv2.circle(img, (u, v), 10, (0, 255, 0), 2)
            cv2.putText(img, label, (u + 12, v), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Parse mPMT number from label and store point
            try:
                m_id = int(label.split("-")[0])
                #if 0 <= m_id < num_pmts:
                grouped_points[m_id].append((u, v))
            except:
                pass  # In case of unexpected label format

    # Draw polylines only for mPMTs with ≥ 2 valid points
    for pts in grouped_points:
        if len(pts) >= 2:
            pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_array], isClosed=True, color=(0, 180, 0), thickness=2)

    return img




def simulate_fisheye_view(json_path, camera_index, K, D, delta_pitch_deg=15.0, delta_yaw_deg=5.0, delta_roll_deg=0.0, delta_r=0.0, return_points_only=False, excluded_pmts=[]):
    #print('simulate_fisheye_view camera_index=',camera_index)
    geo = load_geometry(json_path)
    leds, labels, num_pmts = select_leds(geo)
    cam_pos, R_cam = get_camera_orientation(geo, camera_index, delta_pitch_deg, delta_yaw_deg, delta_roll_deg, delta_r)
    #print("cam_pos =", cam_pos)
    #print("R_cam =", R_cam)
    ylims = (-0.9,1.8)
    if camera_index > 3:
        ylims= (-1.1,1.6)
    img_pts, labels_valid = project_leds(leds, labels, cam_pos, R_cam, K, D, 75.0, ylims, excluded_pmts ) # project_leds(leds, labels, cam_pos, R_cam, K, D):
    #labels = np.array(labels)[valid_mask]  # this is now already filtered in project_leds

    if return_points_only:
        return img_pts, labels_valid
        
    img = draw_projection(img_pts, labels_valid, K)

    # draw end caps
    draw_cylinder_ellipses(K, D, cam_pos, R_cam, img, cylinder_radius=1.65, y_top=1.85, y_bottom=-1.0 )
  

    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Camera {camera_index+1} LED View")
    plt.show()

    filename = f'camera{camera_index+1}_f{int(K[0][0])}_p{int(delta_pitch_deg)}_y{int(delta_yaw_deg)}_r{int(delta_roll_deg)}_dr{int(delta_r*100)}.png'
    cv2.imwrite(filename, img)
    print(f"Saved image to {filename}")


def match_leds_to_blobs(sim_points, blob_points, labels, threshold=100):
    tree_blobs = cKDTree(blob_points)
    tree_leds = cKDTree(sim_points)

    # Nearest blob for each LED
    led_dists, led_to_blob = tree_blobs.query(sim_points, k=1)
    # Nearest LED for each blob
    blob_dists, blob_to_led = tree_leds.query(blob_points, k=1)

    matches = []
    for i_led, (d, i_blob) in enumerate(zip(led_dists, led_to_blob)):
        if d > threshold:
            continue
        # Check if this is mutual best match
        if blob_to_led[i_blob] == i_led:
            x, y = blob_points[i_blob]
            label = labels[i_led]
            matches.append((label, x, y))

    return matches


# In[8]:
from shapely.geometry import Polygon

def polygon_penalty(mPMT_groups, sim_groups, area_weight=1.0):
    penalty = 0.0
    polygons = {}

    for mPMT_id, sim_pts in sim_groups.items():
        if len(sim_pts) < 3:
            continue

        # Expected polygon from simulated LED positions
        sim_sorted = sorted(sim_pts, key=lambda p: int(p[2].split('-')[1]))
        sim_coords = [(x, y) for x, y, _ in sim_sorted]
        sim_poly = Polygon(sim_coords)
        expected_area = sim_poly.area

        # Matched polygon (if any points matched)
        if mPMT_id in mPMT_groups and len(mPMT_groups[mPMT_id]) >= 3:
            pts_sorted = sorted(mPMT_groups[mPMT_id], key=lambda p: int(p[2].split('-')[1]))
            coords = [(x, y) for x, y, _ in pts_sorted]
            blob_poly = Polygon(coords)

            # Self-intersection check
            if not blob_poly.is_valid:
                penalty += 5000.0

            polygons[mPMT_id] = blob_poly

            # Area difference penalty
            area_diff = abs(blob_poly.area - expected_area)
            penalty += area_weight * (area_diff / (expected_area + 1e-6))
        else:
            # No polygon possible (too few matches) → big penalty
            penalty += area_weight * expected_area

    # Penalize overlaps between polygons
    ids = list(polygons.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if polygons[ids[i]].intersects(polygons[ids[j]]):
                inter_area = polygons[ids[i]].intersection(polygons[ids[j]]).area
                penalty += 1000.0 * inter_area

    return penalty



def chisquare(simulated_points, blob_points, labels=None, match_threshold=300.0, sigma_pix = 10.0, control_points={}, polygon_weight=100.0 ):
     # Build label -> point dictionary safely
    label_to_sim = {label: pt for pt, label in zip(simulated_points, labels)}

    # match LEDs to blobs
    matches = match_leds_to_blobs(simulated_points, blob_points, labels, threshold=match_threshold)

    if len(matches) == 0:
        return 1e12  # big penalty if no matches

    # χ² = sum of squared distances
    
    # Build map from label to simulated position
    label_to_sim = {label: (sx, sy) for (sx, sy), label in zip(simulated_points, labels)}

    # Compute chi² sum over matched points
    chi2 = 0.0
    for (label, x, y) in matches:
        #print('chi2 calc: x,y,label=',x,y,label)
        if label not in label_to_sim:
            continue # skip unmatched??? should I add a penalty?
        sx, sy = label_to_sim[label]
        dx = sx - x
        dy = sy - y
        chi2 += (dx**2 + dy**2) / sigma_pix**2

    # Penalize missed matches (LEDs with no match)
    n_matched = len(matches)
    missed_penalty = 10.0 * (len(simulated_points) - n_matched)
    chi2 += missed_penalty

    # Optional: penalize unmatched blobs
    # matched_blob_indices = indices[matched]
    # unmatched_blobs = set(range(len(blob_points))) - set(matched_blob_indices)
    # chi2 += 5.0 * len(unmatched_blobs)

    # --- Polygon penalties ---
    # Build per-mPMT groups of matched blobs
    mPMT_groups = {}
    for (label, x, y) in matches:
        #print('x,y,label=',x,y,label)
        mPMT_id = label.split('-')[0]  # e.g. "003"
        mPMT_groups.setdefault(mPMT_id, []).append((x, y, label))

    sim_groups = {}
    for (sx, sy), label in zip(simulated_points, labels):
        mPMT_id = str(label).split('-')[0]
        sim_groups.setdefault(mPMT_id, []).append((sx, sy, str(label)))
    
    chi2 += polygon_weight * polygon_penalty(mPMT_groups, sim_groups)

    control_weight = 1.0e8  # very strong penalty
    for label, (cx, cy) in control_points.items():
        # Try to find a matched blob with this label
        matched = [ (x, y) for l, x, y in matches if l == label ]
        if matched:
            x, y = matched[0]
            dx = cx - x
            dy = cy - y
            chi2 += control_weight * (dx**2 + dy**2) / sigma_pix**2
        else:
            # If no match, apply severe penalty
            chi2 += control_weight**2
            
    return chi2


def chisquare_residuals_fixed(sim_points, sim_labels,
                              blob_points, all_labels,
                              match_threshold=300.0, sigma_pix=10.0,
                              control_points={}):
    """
    Fixed-length residuals for least_squares LM.
    
    Args:
        sim_points: simulated points (Nx2)
        sim_labels: labels corresponding to sim_points
        blob_points: detected blobs (Mx2)
        all_labels: fixed list of all possible LED labels for this camera
        match_threshold: max distance for matching
        sigma_pix: scaling for residuals
        control_points: dict of control points {label: (x, y)}
        
    Returns:
        residuals: 1D array of length 2*len(all_labels) + 2*len(control_points)
    """
    N_leds = len(all_labels)
    residuals = np.zeros(2*N_leds + 2*len(control_points), dtype=np.float64)

    # --- Match LEDs to blobs ---
    matches = match_leds_to_blobs(sim_points, blob_points, sim_labels,
                                  threshold=match_threshold)
    matched_dict = {label: (x, y) for label, x, y in matches}
    sim_dict     = {label: tuple(pt) for pt, label in zip(sim_points, sim_labels)}

    # --- Residuals for all LEDs ---
    for i, label in enumerate(all_labels):
        if label in sim_dict and label in matched_dict:
            sx, sy = sim_dict[label]
            x, y   = matched_dict[label]
            dx = (sx - x) / sigma_pix
            dy = (sy - y) / sigma_pix
        else:
            # missing sim point or unmatched blob => large penalty
            dx = dy = 100.0
        residuals[2*i]   = dx
        residuals[2*i+1] = dy

    # --- Residuals for control points ---
    for j, (label, (cx, cy)) in enumerate(control_points.items()):
        idx = 2*N_leds + 2*j
        dx = dy = 0.0

        # Penalize distance to matched blob if exists
        if label in matched_dict:
            x, y = matched_dict[label]
            dx += (cx - x) / sigma_pix
            dy += (cy - y) / sigma_pix
        else:
            dx += 100.0
            dy += 100.0

        # Penalize distance to simulated point if exists
        if label in sim_dict:
            sx, sy = sim_dict[label]
            dx += (cx - sx) / sigma_pix
            dy += (cy - sy) / sigma_pix
        else:
            dx += 10.0 / sigma_pix
            dy += 10.0 / sigma_pix

        residuals[idx]   = dx
        residuals[idx+1] = dy

    return residuals





def match_blobs(blobs, json_path, camera_index, K, D, initial_guess=None,
                excluded_pmts={}, bounds=None, control_points={},
                use_lm=False):

    blob_points = np.array([b.pt for b in blobs], dtype=np.float32)

    if initial_guess is None:
        initial_guess = [0.0]*8  # pitch, yaw, roll, r, k1..k4

    match_threshold = 1000

    geo = load_geometry(json_path)
    
    def objective(params):
        sim_points, labels_valid = simulate_fisheye_view(
            json_path=json_path,
            camera_index=camera_index,
            K=K,
            D=np.array(params[4:], dtype=np.float64),
            delta_pitch_deg=params[0],
            delta_yaw_deg=params[1],
            delta_roll_deg=params[2],
            delta_r=params[3],
            return_points_only=True,
            excluded_pmts=excluded_pmts
        )
        return chisquare(sim_points, blob_points, labels_valid,
                         match_threshold, 20.0, control_points)

    if use_lm:
        def residuals(params):
            sim_points, labels_valid = simulate_fisheye_view(
                json_path=json_path,
                camera_index=camera_index,
                K=K,
                D=np.array(params[4:], dtype=np.float64),
                delta_pitch_deg=params[0],
                delta_yaw_deg=params[1],
                delta_roll_deg=params[2],
                delta_r=params[3],
                return_points_only=True,
                excluded_pmts=excluded_pmts
            )
            leds_all, labels_all, num_pmts_all = select_leds(geo)

            return chisquare_residuals_fixed(sim_points, labels_valid, blob_points, labels_all,
                                       match_threshold, 20.0, control_points)

        result = least_squares(residuals, initial_guess, method='lm', max_nfev=20000)
        best_params = result.x
        chi2 = np.sum(result.fun**2)
        success = result.success
    else:
        result = minimize(objective, initial_guess, method='Powell', bounds=bounds)
        best_params = result.x
        chi2 = result.fun
        success = result.success

    # Final simulation with best-fit
    D_fit = np.array(best_params[4:])
    sim_points, labels_valid = simulate_fisheye_view(
        json_path=json_path,
        camera_index=camera_index,
        K=K,
        D=D_fit,
        delta_pitch_deg=best_params[0],
        delta_yaw_deg=best_params[1],
        delta_roll_deg=best_params[2],
        delta_r=best_params[3],
        return_points_only=True,
        excluded_pmts=excluded_pmts
    )

    # Match again with best-fit
    matches = match_leds_to_blobs(sim_points, blob_points, labels_valid,
                                  threshold=match_threshold)

    return matches, {
        'delta_pitch_deg': best_params[0],
        'delta_yaw_deg': best_params[1],
        'delta_roll_deg': best_params[2],
        'delta_r': best_params[3],
        'D_fit': D_fit,
        'chi2': chi2,
        'success': success
    }



# In[11]:


def run_blob_detector(camera_index, image_file_name, minarea=40, maxarea=1000, minthres=40, maxthres=255 ):
    # Load grayscale image
    img = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_file_name}")

    # Flip?
    if camera_index>=4:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    processed_image = cv2.bilateralFilter(np.array(img), 5, 75, 75)
    
    # Set up detector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Filter by brightness
    params.filterByColor = True
    params.blobColor = 255  # detect bright spots on dark background

    # Area filtering
    params.filterByArea = True
    params.minArea = minarea
    params.maxArea = maxarea

    params.minThreshold = minthres
    params.maxThreshold = maxthres

    # Circularity (optional)
    params.filterByCircularity = False

    # Inertia & convexity (optional)
    params.filterByInertia = False
    params.filterByConvexity = False

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect( processed_image )

    return keypoints  # list of cv2.KeyPoint (or cv2.Blob.Blob)



def filter_blobs_by_distance(blobs, min_distance=1):
    filtered = []
    points = []

    for b in blobs:
        x, y = b.pt
        if all(np.hypot(x - px, y - py) >= min_distance for px, py in points):
            filtered.append(b)
            points.append((x, y))

    return filtered



def draw_image_with_blobs_save(camera_index, image_file_name, blobs, output_file_name):
    # Load the original image (color)
    img = cv2.imread(image_file_name)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_file_name}")

    # Flip?
    if camera_index>=4:
        img = cv2.rotate(img, cv2.ROTATE_180)
        
    # Draw blobs (yellow hollow circles)
    for b in blobs:
        x, y = int(b.pt[0]), int(b.pt[1])
        radius = int(b.size / 2)
        cv2.circle(img, (x, y), radius, (0, 255, 255), 2)  # BGR yellow

    # Save annotated image at full resolution
    cv2.imwrite(output_file_name, img)
    print(f"Annotated image saved to: {output_file_name}")


def visualize_all_leds_and_matches(
    image_filename, blobs, matches, fitpars, geo_file, camera_index, K, D,
    output_filename="blob_match_overlay.png",
    font_scale=1.5,
    thickness=2,
    excluded_pmts=[],
    control_points={},
    overlay_image_name=None,
    alpha=0.5 ):
    """
    Visualize all simulated LEDs and matched blobs on the image.
    Saves the image to `output_filename` and prints the filename.

    Args:
        image_filename: path to input image file
        blobs: list of cv2.Blob.Blob objects (detected blobs)
        matches: list of (x, y, label) matched points from match_blobs
        fitpars: [delta_pitch_deg, delta_yaw_deg, delta_roll_deg, delta_r] optimized camera params
        geo_file: path to geometry json file
        camera_index: integer index of camera
        K, D: camera intrinsic matrix and distortion coefficients
        output_filename: output image filename to save
        font_scale, thickness: text rendering params
    """
    
    # Load image
    img = cv2.imread(image_filename)
    if img is None:
        raise FileNotFoundError(f"Could not load image {image_filename}")

    # Flip?
    if camera_index>=4:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    # Extract blob points as (int x, int y)
    blob_points = [(int(b.pt[0]), int(b.pt[1]), b.size) for b in blobs]

    # Run simulation to get all LED projected points and labels
    print(fitpars)
    sim_points, labels_valid = simulate_fisheye_view(
        geo_file, camera_index, K, D,
        delta_pitch_deg=fitpars.get('delta_pitch_deg', 0),
        delta_yaw_deg=fitpars.get('delta_yaw_deg', 0),
        delta_roll_deg=fitpars.get('delta_roll_deg', 0),
        delta_r=fitpars.get('delta_r', 0),
        return_points_only=True,
        excluded_pmts=excluded_pmts
    )

    # Draw all simulated LEDs with labels and polygons on image in last arg
    sim_points_int = np.round(sim_points).astype(int)

    img = draw_projection(sim_points_int, labels_valid, K, img)
    
    
    # draw end caps
    geo = load_geometry(geo_file)
    cam_pos, R_cam = get_camera_orientation(geo, camera_index, fitpars.get('delta_pitch_deg', 0), fitpars.get('delta_yaw_deg', 0), fitpars.get('delta_roll_deg', 0), fitpars.get('delta_r', 0))

    draw_cylinder_ellipses(K, D, cam_pos, R_cam, img, cylinder_radius=1.65, y_top=1.85, y_bottom=-1.0 )
    
    
    # Draw all detected blobs in red
    for (bx, by, sz) in blob_points:
        cv2.circle(img, (bx, by), int(sz), (0, 0, 255), 2 )

    # Draw matched blobs in cyan with labels
    for (label, mx, my) in matches:
        mx_i, my_i = int(mx), int(my)
        cv2.circle(img, (mx_i, my_i), 12, (0, 255, 255), 2)  # yellow circle outline
        cv2.putText(img, label, (mx_i + 5, my_i + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)


    # --- Draw control points ---
    for label, (cx, cy) in control_points.items():
        # Draw a circle at the control point
        cv2.circle(
            img,                     # the image array
            (int(cx), int(cy)),        # position
            radius=20,                  # circle radius
            color=(255, 0, 255),       # magenta color (BGR)
            thickness=3               # 
        )
        # Put the label text near the control point
        cv2.putText(
            img,
            str(label),
            (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 255),
            thickness
        )

    # --- Optional overlay ---
    if overlay_image_name is not None:
        overlay_img = cv2.imread(overlay_image_name)
        if overlay_img is not None:
            # Flip for upside-down cameras
            if camera_index >= 4:
                overlay_img = cv2.rotate(overlay_img, cv2.ROTATE_180)
            # Resize overlay to match base image if needed
            #if overlay_img.shape[:2] != img.shape[:2]:
            #    overlay_img = cv2.resize(overlay_img, (img.shape[1], img.shape[0]))
            # Apply transparency
            img = cv2.addWeighted(img, 1.0 - alpha, overlay_img, alpha, 0)
        else:
            print(f"Warning: Could not load overlay image {overlay_img_name}")
    # Save and print filename
    cv2.imwrite(output_filename, img)
    print(f"Overlay image saved to: {output_filename}")

    return img


def build_matches_info_dict(matches, blobs, geo_file, camera_index, K, D, fitpars, excluded_pmts=[]):
    """
    Build a dict of matched blob info using simulation and fitted parameters.

    Args:
        matches (list of (x_blob, y_blob, label)): Matches from match_blobs().
        blobs (list of cv2.KeyPoint): Full list of detected blobs.
        geo_file (str): Path to the geometry JSON file.
        camera_index (int): Index of the camera used.
        K (ndarray): Camera intrinsic matrix.
        D (ndarray): Fisheye distortion coefficients.
        fitpars (dict): Fitted parameters including delta_pitch_deg, delta_yaw_deg, etc.

    Returns:
        dict: { label: (x_blob, y_blob, size, dist_to_simulation), ... }
    """
    from math import sqrt
    print('geo_file=', geo_file)
    

    
    # Get full simulated LED positions under current fit parameters
    sim_points, labels = simulate_fisheye_view(
        geo_file,
        camera_index,
        K,
        D,
        delta_pitch_deg=fitpars['delta_pitch_deg'],
        delta_yaw_deg=fitpars['delta_yaw_deg'],
        delta_roll_deg=fitpars['delta_roll_deg'],
        delta_r=fitpars['delta_r'],
        return_points_only=True,
        excluded_pmts=excluded_pmts
    )

    # Build a dict of simulated label → (x_sim, y_sim)
    label_to_sim = {label: (sx, sy) for (sx, sy), label in zip(sim_points, labels)}

    # Build a label → blob mapping for lookup
    label_to_blob = {}
    for label, x_blob, y_blob in matches:
        if label not in label_to_sim:
            continue  # skip excluded PMTs entirely
        # Find the blob size for this (x, y) from the list of KeyPoints
        closest_blob = min(blobs, key=lambda b: (b.pt[0] - x_blob)**2 + (b.pt[1] - y_blob)**2)
        size = closest_blob.size

        
        if label in label_to_sim:
            x_sim, y_sim = label_to_sim[label]
            dist = sqrt((x_blob - x_sim)**2 + (y_blob - y_sim)**2)
            label_to_blob[label] = (x_blob, y_blob, size, dist)
        else:
            label_to_blob[label] = (x_blob, y_blob, size, -1)

    print("Sample matches labels:", matches[:10])
    print("Sample label_to_sim keys:", list(label_to_sim.keys())[:10])
    return label_to_blob



def save_matching_results_to_json(
    output_filename,
    image_filename,
    fitpars,
    matches, 
    blobs, 
    geo_file, 
    camera_index, 
    K, 
    D, 
    excluded_pmts=[]):
    """
    Save the matching results and fit parameters to a JSON file.

    Args:
        output_filename (str): Path to the output JSON file.
        image_filename (str): Path to the input image being matched.
        fitpars (dict): Dictionary of best-fit parameters.
        matches_info (list of tuples): Each tuple is (label, (x, y), size, dist).
    """

    print('excluded_pmts=',excluded_pmts)
    matches_info = build_matches_info_dict(matches, blobs, geo_file, camera_index, K, D, fitpars, excluded_pmts )

    # Structure the output
    results = {
        "image_file": os.path.basename(image_filename),
        "fit_parameters": {
            key: float(value) for key, value in fitpars.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        "success": bool(fitpars.get("success", False)),  # separate field

        "matches": {
            label: {
                "x": float(x),
                "y": float(y),
                "size": float(size),
                "distance": float(dist)
            }
            for label, (x, y, size, dist) in matches_info.items()
        }
    }

    # Write to JSON
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {output_filename}")



