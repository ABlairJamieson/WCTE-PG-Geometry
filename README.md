
## Geometry to file code 

Use Dean's Geometry package (https://github.com/WCTE/Geometry).  

Geometry_blair.py is a helper/wrapper that runs Dean's geometry package to extract design locations of LEDs, Domes and cameras in WCTE.  

WCTE_Geometry_to_file.py uses Geometry_blair.py to get the geometry, and puts it into a json file.

This github also includes a sample output json file.

Potential future updates:
- output non-design positions
- save both design and non-design positions in the json file

## Predictor Labeller code

Code to label an image using the design LED locations as a guide.  Predicts what an image from the ideal design geometry should look like, then matches LEDs to the closest blob.

PredictImage.py : module of functions to help with labelling the image

Predict_Image_PCH1.ipynb : uses PredictImage.py to do image labelling for an image from PCH1

### JSON Output Structure

This JSON file stores the results of blob–LED matching and camera parameter fitting for a given image.  

#### Example
```json
{
    "image_file": "example.png",
    "fit_parameters": {
        "delta_pitch_deg": 0.15,
        "delta_yaw_deg": -0.05,
        "delta_roll_deg": 0.02,
        "delta_r": 0.01
    },
    "success": true,
    "matches": {
        "043-10": {
            "x": 7278.0,
            "y": 3697.0,
            "size": 12.5,
            "distance": 15.3
        },
        "044-05": {
            "x": 6120.0,
            "y": 2900.0,
            "size": 11.8,
            "distance": 12.7
        }
    }
}
```

#### Top-level fields

- **`image_file`** *(string)*:  
  The basename of the input image file being processed.

- **`fit_parameters`** *(object of floats)*:  
  Dictionary of fitted camera adjustment parameters returned by the optimizer.  
  Keys include:
  - `delta_pitch_deg` – adjustment in pitch (degrees)  
  - `delta_yaw_deg` – adjustment in yaw (degrees)  
  - `delta_roll_deg` – adjustment in roll (degrees)  
  - `delta_r` – radial translation offset  
  - *(any other numeric entries in `fitpars` are also included here)*  

- **`camera_intrinsics`** *(array, shape 3×3)*:  
  The intrinsic calibration matrix `K` of the camera, stored as a nested list.  
  This encodes focal lengths and principal point offsets:
[[fx, 0, cx],
[ 0, fy, cy],
[ 0, 0, 1]]

- **`distortion_coefficients`** *(array)*:  
The fisheye distortion coefficients vector `D`, stored as a list of floats.  
For the OpenCV fisheye model `[k1, k2, k3, k4]`, representing the four radial distortion parameters.

- **`success`** *(boolean)*:  
Indicates whether the optimization converged successfully.

- **`matches`** *(object)*:  
Dictionary mapping each LED label to its matched blob information, with keys being of the form '044-5' with the first numbers in the string being mPMT number and after the dash being LED number:
- `x` *(float)* – x-coordinate of the matched blob in image pixels  
- `y` *(float)* – y-coordinate of the matched blob in image pixels  
- `size` *(float)* – blob size (as detected)  
- `distance` *(float)* – residual distance between blob and simulated projection

Example section of a labelled image output image is below.
<img width="1325" height="864" alt="image" src="https://github.com/user-attachments/assets/1c7ae4df-9cd3-4864-8a3b-b96184e69807" />



# WCTE_pg_fitter

This notebook performs **3D photogrammetry analysis** for WCTE detector modules using fisheye cameras. It includes both a Jupyter notebook for visualization and a library for bundle adjustment.

---

## Notebook: `WCTE_pg_fitter.ipynb`

This notebook handles the **end-to-end photogrammetry workflow**.

### 1. Setup & Imports
- Libraries: `numpy`, `matplotlib`, `collections.defaultdict`, `cv2`, `scipy`.
- Cylinder parameters (radius, offsets) are defined to map features onto the unrolled cylinder and top/bottom caps.

### 2. Data Loading
- **Design points**: 3D reference positions of features on detector modules.  
- **Measured/fitted points**: Differences (`diff`) between initial guesses and optimized positions after bundle adjustment.  
- **Labels**: Feature/module identifiers, e.g., `M1-12`.

### 3. Masking & Categorization
- Identify **wall**, **top**, and **bottom** features based on Y-coordinate.  
- Create `fitted_mask` for features that moved during the fit.  
- Compute **out-of-plane displacements** for color-coding deviations:
  - Wall: radial shift from cylinder center.  
  - Top/Bottom: vertical shift along Y-axis.  

### 4. Coordinate Transformation / Unrolling
- **Wall points** mapped using cylindrical coordinates (`phi`) to an unrolled 2D plane.  
- **Top/Bottom caps** offset appropriately in X/Z plane.  
- **Fitted points** exaggerated in-plane by 10× to emphasize deviations.

### 5. Module Grouping
- Features grouped by module using `defaultdict(list)`.  
- Each module’s design points, fitted points, and labels are stored for plotting.

### 6. Plotting
- **Arrows**: Show in-plane deviations from design.  
- **Color-coded scatter points**: Out-of-plane deviations using `coolwarm` colormap (±50 mm).  
- **Labels**: Only for features with deviation > 50 mm.  
- **Non-fitted points**: Light grey.  
- **Module polygons**: Lines connecting features in order, avoiding cylinder cut jumps.  
- **Module centroids**: Module names displayed at centroid positions.  
- Optional: **Camera positions & pointing directions** plotted on the unrolled plane.

### 7. Export
- Figures saved as `.png` and `.pdf`.  
- Axis, grid, and aspect ratio adjustments for clarity.

---

## Library: `bundle_fisheye_ba.py`

This module provides **bundle adjustment routines** specialized for fisheye cameras and WCTE photogrammetry.

### 1. Core Functions
- `simulate_fisheye_view(...)`: Projects 3D points into camera coordinates using fisheye optics.  
- `rvec_to_direction(rvec)`: Converts rotation vectors to 3D pointing directions.  
- `project_points(...)`: Standard projection for comparison with measured image points.  
- `compute_residuals(...)`: Calculates reprojection errors for all features and cameras.

### 2. Bundle Adjustment
- Uses **scipy.optimize.least_squares** to refine camera extrinsics (`rvec`/`tvec`) and optionally 3D point positions.  
- Handles **constraints** such as fixed module geometries, cylinder radius, and cap offsets.  
- Returns a dictionary `ba_results` containing:
  - `cameras`: Fitted camera extrinsics (`tvec`, `rvec`), number of observations.  
  - `points`: Adjusted 3D feature positions.  
  - `residuals`: Per-feature reprojection errors.

### 3. Utilities
- Coordinate transformations between **camera frame**, **gantry**, and **unrolled cylinder plane**.  
- Matching routines to associate **detected LED blobs** in images with **simulated LED positions**.  
- Functions to **evaluate fit quality**, e.g., RMS error computation.

---

## Summary

- `WCTE_pg_fitter.ipynb` → **Data ingestion, visualization, and plotting** of fitted photogrammetry results.  
- `bundle_fisheye_ba.py` → **Algorithmic library** for bundle adjustment, camera simulation, and feature matching.  
- Together, they allow:
  1. Camera extrinsic refinement.  
  2. Feature point reconstruction.  
  3. Intuitive visual feedback on deviations and module alignment.
 
Here is an example of the output so far.  Some additional work needs to be done on the inputs to improve the matching for some of the LEDs.

<img width="1500" height="1200" alt="wcte_3d_fit_result_07102025" src="https://github.com/user-attachments/assets/e584d7b9-2fd8-4e55-a8ad-291befec10a1" />


> **Note / Warning**
>
> - This code has been tested with **SciPy ≥ 1.16.0**. Older versions may raise warnings or fail due to changes in `scipy.optimize.least_squares` handling of bounds, Jacobians, or parallel workers.
> - **SciPy 1.16.0** was released on **June 22, 2025**, and is compatible with **Python 3.11** and **3.12**. For more details, refer to the [SciPy 1.16.0 release notes](https://docs.scipy.org/doc/scipy/release/1.16.0-notes.html).
> - By default, bundle adjustment uses multiple workers (`nworkers`) for parallel residual evaluation. To **run serially**, set `nworkers=1` or comment out the `nworkers` argument in the call to `least_squares` inside `bundle_fisheye_ba.py`.
>   ```python
>   # Original (parallel):
>   result = least_squares(residuals_func, x0, bounds=bounds, workers=nworkers)
>
>   # Serial version:
>   result = least_squares(residuals_func, x0, bounds=bounds)  # single-threaded
>   ```
> - This can help debug issues or run on systems without multiprocessing support.
