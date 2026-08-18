# AP6 — 3D Candle Volume Reconstruction

Reconstructs the 3D shape and volume of a pillar candle from a single 2D photo, using edge detection, spline fitting, and numerical integration.

## What it does

- **Edge detection:** finds the candle's silhouette in a 2D image with Canny edge detection (OpenCV).
- **Contour extraction:** extracts the radius profile of the candle along its height from the detected edges.
- **Spline fitting:** fits a smoothing spline to the radius profile to get a continuous `r(y)` function.
- **Volume of revolution:** integrates `V = π ∫ r(y)² dy` with Simpson's rule (`scipy.integrate.simpson`) to compute the candle's volume, assuming rotational symmetry.
- **3D mesh generation:** builds a 3D surface mesh by revolving the fitted radius profile around the central axis.
- Falls back to a synthetically generated candle image if no real photo is provided, so the pipeline can run end-to-end without external input.
- Compares the computed volume against the theoretical volume of a frustum (truncated cone) as a sanity check.

## Run it

```bash
pip install numpy scipy matplotlib opencv-python pillow
python main.py
```

## Output

PNG figures saved to this folder:
- `candle_processing_steps.png` — edge detection and contour extraction pipeline
- `candle_3d_reconstruction.png` — the reconstructed 3D mesh
- `candle_volume_integration.png` — visualization of the disk-method volume integration

Plus console output with the computed volume (in cubic pixels) and, when available, a comparison against the theoretical frustum volume.
