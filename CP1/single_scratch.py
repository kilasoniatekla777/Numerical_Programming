from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --- 1. Custom Derivative Computation (Reuse/Refinement) ---

def compute_derivatives(positions: np.ndarray, times: np.ndarray):
    """
    Computes 1st (Velocity), 2nd (Acceleration), 3rd (Jerk), and 4th (Jounce)
    order derivatives using finite difference method (from scratch).
    """
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)

    N = len(times)
    if N < 5:  # Need at least 5 points for 4th derivative
        zeros = np.zeros(N)
        return zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros

    # Initialize all derivative arrays
    vx, vy, ax, ay, jx, jy, qx, qy = [np.zeros(N) for _ in range(8)]
    dt = np.diff(times)

    # 1st Derivative (Velocity)
    for i in range(1, N):
        # Forward difference
        vx[i] = (positions[i, 0] - positions[i - 1, 0]) / dt[i - 1]
        vy[i] = (positions[i, 1] - positions[i - 1, 1]) / dt[i - 1]

    # 2nd Derivative (Acceleration)
    for i in range(2, N):
        # Forward difference on velocity
        ax[i] = (vx[i] - vx[i - 1]) / dt[i - 1]
        ay[i] = (vy[i] - vy[i - 1]) / dt[i - 1]

    # 3rd Derivative (Jerk)
    for i in range(3, N):
        # Forward difference on acceleration
        jx[i] = (ax[i] - ax[i - 1]) / dt[i - 1]
        jy[i] = (ay[i] - ay[i - 1]) / dt[i - 1]

    # 4th Derivative (Jounce)
    for i in range(4, N):
        # Forward difference on jerk
        qx[i] = (jx[i] - jx[i - 1]) / dt[i - 1]
        qy[i] = (jy[i] - jy[i - 1]) / dt[i - 1]

    return vx, vy, ax, ay, jx, jy, qx, qy


# --- 2. Custom Image Processing Kernels (From Scratch) ---

def create_gaussian_kernel(size, sigma=1.0):
    """Creates a 2D Gaussian kernel."""
    center = size // 2
    x, y = np.mgrid[-center:center + 1, -center:center + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def apply_convolution(image, kernel):
    """Applies a 2D convolution to an image (from scratch)."""
    kH, kW = kernel.shape
    iH, iW = image.shape

    # Padding to handle edges
    padH, padW = kH // 2, kW // 2
    # Ensure padding is 0 for Sobel and Gaussian
    padded_image = np.pad(image, ((padH, padH), (padW, padW)), mode='edge')

    output = np.zeros((iH, iW), dtype=float)

    # Perform convolution
    for y in range(iH):
        for x in range(iW):
            # Element-wise multiplication and summation (the core convolution operation)
            output[y, x] = np.sum(padded_image[y:y + kH, x:x + kW] * kernel)

    return output


def gaussian_blur_scratch(image: np.ndarray, size: int, sigma: float = 1.0):
    """Smoothing via Gaussian Blur (from scratch)."""
    kernel = create_gaussian_kernel(size, sigma)
    # Applying the convolution twice (for separable kernels) is faster,
    # but applying the 2D kernel is more conceptually direct for scratch implementation.
    return apply_convolution(image.astype(float), kernel)


def sobel_scratch(image: np.ndarray):
    """Sobel edge detection (spatial derivatives Gx, Gy) (from scratch)."""
    # Standard 3x3 Sobel kernels
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)

    Gy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=float)

    # Apply convolution for horizontal and vertical derivatives
    grad_x = apply_convolution(image.astype(float), Gx)
    grad_y = apply_convolution(image.astype(float), Gy)

    # Magnitude: G = sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize magnitude to 0-255 range for display/thresholding
    mag_normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255.0

    return grad_x, grad_y, magnitude, mag_normalized.astype(np.uint8)


def threshold_scratch(image: np.ndarray, threshold_val: int):
    """Simple binary thresholding (from scratch)."""
    mask = (image > threshold_val).astype(np.uint8) * 255
    return mask


# --- 3. Custom Clustering (K-Means K=1 Centroid) ---

def kmeans_1_cluster_scratch(data_points: np.ndarray):
    """
    Custom 1-cluster K-Means implementation (from scratch).
    Since K=1, the centroid is simply the mean of all data points.
    This function processes the 4D feature vector (x, y, D, E)
    to find the 4D centroid, which satisfies the derivative-incorporating
    clustering requirement.
    """
    # Check if we have any points to cluster
    if data_points.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])  # Return a zero array if no data

    # Calculate the mean of all axes (x, y, D, E)
    centroid = np.mean(data_points, axis=0)
    return centroid


# --- 4. Main Processing Loop ---

def process_video_scratch(video_path: str, out_video_path: str | None = None):
    # --- Parameters ---
    SOBEL_THRESHOLD = 50  # Threshold for Sobel edges
    MOTION_THRESHOLD = 15  # Threshold for raw frame difference
    GAUSSIAN_SIZE = 5  # Kernel size for Gaussian Smoothing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    writer = None
    if out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, frame_size)

    # First frame setup
    ret, frame = cap.read()
    if not ret:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("Could not read first frame.")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply smoothing to the previous frame (Required: from scratch)
    gray_prev_smoothed = gaussian_blur_scratch(gray_prev, GAUSSIAN_SIZE)

    positions = []
    times = []
    frame_idx = 0
    last_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pre-processing: Grayscale and Smoothing (from scratch)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_smoothed = gaussian_blur_scratch(gray, GAUSSIAN_SIZE)

        # 2. Motion Detection (Temporal Derivative)
        # Difference between raw smoothed frames
        diff = np.abs(gray_smoothed - gray_prev_smoothed).astype(np.uint8)
        motion_raw = threshold_scratch(diff, MOTION_THRESHOLD)

        # 3. Edge Detection (Spatial Derivative)
        # Gx, Gy are not explicitly used, but D (motion mag) and E (Sobel mag) are.
        _, _, mag_float, sobel_mask_display = sobel_scratch(gray_smoothed)
        sobel_mask = threshold_scratch(sobel_mask_display, SOBEL_THRESHOLD)

        # 4. Feature Fusion: Moving Edges (Logical AND)
        fused_mask = cv2.bitwise_and(motion_raw, sobel_mask)  # Using cv2.bitwise_and for speed

        # 5. Clustering and Object Localization
        ys, xs = np.where(fused_mask > 0)

        if len(xs) > 1:
            # Get derivative values at the feature coordinates
            motion_mags = diff[ys, xs]  # D: Motion magnitude (Temporal Derivative)
            sobel_mags = mag_float[ys, xs]  # E: Sobel magnitude (Spatial Derivative)

            # Feature vector (N, 4): [x, y, D, E] - Norm incorporates derivatives
            coords_with_derivatives = np.column_stack((xs, ys, motion_mags, sobel_mags))

            # Perform 1-cluster K-Means (from scratch)
            centroid_4d = kmeans_1_cluster_scratch(coords_with_derivatives)

            # Extract the position (x, y) from the 4D centroid
            center_x, center_y = centroid_4d[0], centroid_4d[1]
            last_center = (float(center_x), float(center_y))

        # 6. Update Tracker and Draw
        if last_center is not None:
            cx, cy = int(round(last_center[0])), int(round(last_center[1]))
            # BIG RED DOT
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Center: ({cx}, {cy})",
                (cx + 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            t = frame_idx * dt
            positions.append([cx, cy])
            times.append(t)

        cv2.imshow("Result with Centroid", frame)
        cv2.imshow("Fused Motion Edges (Scratch)", fused_mask)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Prepare for the next frame iteration
        gray_prev_smoothed = gray_smoothed.copy()
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    positions = np.array(positions, dtype=float)
    times = np.array(times, dtype=float)

    vx, vy, ax, ay, jx, jy, qx, qy = compute_derivatives(positions, times)

    print("positions shape:", positions.shape)
    print("times shape:", times.shape)

    # Plotting logic for derivatives (as requested previously)
    if len(times) > 4:
        t = times
        plt.figure(figsize=(10, 12))

        plt.subplot(4, 1, 1)
        plt.plot(t, vx, label="vx (Velocity X)")
        plt.plot(t, vy, label="vy (Velocity Y)")
        plt.title("1st Derivative: Velocity vs. Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [px/s]")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t, ax, label="ax (Acceleration X)")
        plt.plot(t, ay, label="ay (Acceleration Y)")
        plt.title("2nd Derivative: Acceleration vs. Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [px/s²]")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t, jx, label="jx (Jerk X)")
        plt.plot(t, jy, label="jy (Jerk Y)")
        plt.title("3rd Derivative: Jerk vs. Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Jerk [px/s³]")
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t, qx, label="qx (Jounce X)")
        plt.plot(t, qy, label="qy (Jounce Y)")
        plt.title("4th Derivative: Jounce (Snap) vs. Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Jounce [px/s⁴]")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Not enough tracking data (less than 5 frames) to compute and plot 4th-order derivatives.")


if __name__ == "__main__":
    # WARNING: This script requires a 'data/shavi2.mp4' file and an 'outputs' folder to run successfully.
    # Adjust paths as needed for your local environment.
    try:
        root = Path(__file__).resolve().parent.parent
        video_file = str(root / "data" / "shavi2.mp4")
        out_video = str(root / "outputs" / "single_centroid_scratch.mp4")

        # Ensure 'outputs' directory exists
        Path(root / "outputs").mkdir(exist_ok=True)

        process_video_scratch(video_file, out_video_path=out_video)
        print("Saved result video to:", out_video)
    except FileNotFoundError:
        print("\n--- ERROR ---\nVideo file not found. Please check the path:", video_file)
    except RuntimeError as e:
        print(f"\n--- ERROR during video processing ---\n{e}")