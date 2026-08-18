# CP1 — Video Object Tracking & Motion Derivatives

Tracks moving objects in video using frame-difference edge detection, then computes velocity, acceleration, jerk, and jounce (1st-4th derivatives of position) via finite differences.

## What it does

- Reads a video frame by frame, converts to grayscale, and computes Sobel gradients to find edges.
- Detects motion by comparing edge maps between consecutive frames.
- Tracks the centroid of the moving region(s) over time:
  - **Single object:** one centroid = mean position of all moving pixels.
  - **Multiple objects:** KMeans clustering groups moving pixels into `n_objects` clusters, each tracked independently.
- From the position/time series, computes velocity, acceleration, jerk, and jounce using finite differences.
- Overlays tracking markers on the video and writes an annotated output video; plots the derivative curves over time.

## Files

| File | Description |
|---|---|
| `cp1_single_object.py` | Single-object tracking using OpenCV/NumPy (library-based). |
| `multiple_objects.py` | Multi-object tracking with KMeans clustering (library-based). |
| `single_scratch.py` | Single-object tracking with Gaussian blur, Sobel, thresholding, and 1-cluster KMeans implemented from scratch (no OpenCV/sklearn filtering). |

## Run it

```bash
pip install opencv-python numpy matplotlib scikit-learn
python cp1_single_object.py            # single object, uses 2fish.mp4 by default
python multiple_objects.py             # multiple objects
python single_scratch.py               # from-scratch pipeline
```

## Data

`2fish.mp4` and `videoplayback.mp4` — sample input videos included in this folder.

## Output

An annotated output video plus matplotlib plots of position, velocity, acceleration, jerk, and jounce over time.
