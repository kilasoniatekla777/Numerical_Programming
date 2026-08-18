from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path


# ---------------------------------------------------------
# 1. Derivatives (same as your version)
# ---------------------------------------------------------
def compute_derivatives(positions: np.ndarray, times: np.ndarray):
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)

    N = len(times)
    if N < 2:
        zeros = np.zeros(N)
        return zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros

    dt = np.diff(times)

    vx = np.zeros(N)
    vy = np.zeros(N)
    ax = np.zeros(N)
    ay = np.zeros(N)
    jx = np.zeros(N)
    jy = np.zeros(N)
    qx = np.zeros(N)
    qy = np.zeros(N)

    # velocity
    for i in range(1, N):
        vx[i] = (positions[i, 0] - positions[i - 1, 0]) / dt[i - 1]
        vy[i] = (positions[i, 1] - positions[i - 1, 1]) / dt[i - 1]

    # acceleration
    for i in range(1, N):
        ax[i] = (vx[i] - vx[i - 1]) / dt[i - 1]
        ay[i] = (vy[i] - vy[i - 1]) / dt[i - 1]

    # jerk
    for i in range(1, N):
        jx[i] = (ax[i] - ax[i - 1]) / dt[i - 1]
        jy[i] = (ay[i] - ay[i - 1]) / dt[i - 1]

    # jounce
    for i in range(1, N):
        qx[i] = (jx[i] - jx[i - 1]) / dt[i - 1]
        qy[i] = (jy[i] - jy[i - 1]) / dt[i - 1]

    return vx, vy, ax, ay, jx, jy, qx, qy


# ---------------------------------------------------------
# 2. Video processing for 1 or MANY objects
# ---------------------------------------------------------
def process_video(
    video_path: str,
    out_video_path: str | None = None,
    n_objects: int = 1,
):
    """
    Track 1 or multiple moving objects using edge-based motion mask + KMeans.
    n_objects = 1  -> single centroid (mean of all moving pixels)
    n_objects >= 2 -> KMeans clustering of motion pixels into n_objects clusters
    """
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

    # ---- FIRST FRAME ----
    ret, frame = cap.read()
    if not ret:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("Could not read first frame.")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, (7, 7), 0)

    # trajectories: one list per object
    trajectories = [[] for _ in range(n_objects)]
    times: list[float] = []

    frame_idx = 0

    # some distinct colors (BGR) for drawing multiple objects
    colors = [
        (0, 0, 255),     # red
        (0, 255, 0),     # green
        (255, 0, 0),     # blue
        (0, 255, 255),   # yellow
        (255, 0, 255),   # magenta
        (255, 255, 0),   # cyan
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- grayscale + smoothing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # --- Sobel edges ---
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, edge_mask = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)

        # --- frame differencing for motion ---
        diff_gray = cv2.absdiff(gray, gray_prev)
        _, motion_raw = cv2.threshold(diff_gray, 12, 255, cv2.THRESH_BINARY)

        # moving edges = motion ∧ edges
        motion_mask = cv2.bitwise_and(motion_raw, edge_mask)

        # morphology: open → remove specks, close → fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # coordinates of moving pixels
        ys, xs = np.where(motion_mask > 0)

        if len(xs) >= n_objects:
            coords = np.column_stack((xs, ys))  # shape (M, 2), columns = [x, y]

            # find centers:
            if n_objects == 1:
                centers = coords.mean(axis=0, keepdims=True)  # (1,2)
            else:
                # KMeans clustering into n_objects moving blobs
                kmeans = KMeans(
                    n_clusters=n_objects,
                    n_init=5,
                    random_state=0,
                )
                kmeans.fit(coords)
                centers = kmeans.cluster_centers_  # (n_objects, 2)

            t = frame_idx * dt
            times.append(t)

            # draw & store each object's center
            motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

            for k in range(n_objects):
                cx, cy = centers[k]
                cx_i, cy_i = int(round(cx)), int(round(cy))

                color = colors[k % len(colors)]

                # filled circle + small outline
                cv2.circle(frame, (cx_i, cy_i), 5, color, -1)
                cv2.circle(frame, (cx_i, cy_i), 7, (255, 255, 255), 1)

                cv2.circle(motion_bgr, (cx_i, cy_i), 5, color, -1)
                cv2.circle(motion_bgr, (cx_i, cy_i), 7, (255, 255, 255), 1)

                trajectories[k].append([cx, cy])

        else:
            # not enough points – just show mask, don’t append to trajectories
            motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Result with centroids", frame)
        cv2.imshow("Motion mask", motion_bgr)

        if writer is not None:
            writer.write(frame)

        # ESC to quit early
        if cv2.waitKey(1) & 0xFF == 27:
            break

        gray_prev = gray.copy()
        frame_idx += 1

    # ---- CLEANUP ----
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # ---- CONVERT TRAJECTORIES TO ARRAYS & COMPUTE DERIVATIVES ----
    times_arr = np.array(times, dtype=float)
    print("Total tracked frames (with detections):", len(times_arr))

    # store results for each object in a dict
    all_results = []

    for k in range(n_objects):
        if len(trajectories[k]) == 0:
            print(f"Object {k}: no detections.")
            continue

        positions_k = np.array(trajectories[k], dtype=float)
        print(f"Object {k}: positions shape:", positions_k.shape)

        vx, vy, ax, ay, jx, jy, qx, qy = compute_derivatives(positions_k, times_arr)

        print(f"Object {k}: vx shape:", vx.shape, "vy shape:", vy.shape)
        print(f"Object {k}: ax shape:", ax.shape, "ay shape:", ay.shape)
        print(f"Object {k}: jx shape:", jx.shape, "jy shape:", jy.shape)
        print(f"Object {k}: qx shape:", qx.shape, "qy shape:", qy.shape)

        all_results.append(
            {
                "object_id": k,
                "times": times_arr,
                "positions": positions_k,
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "jx": jx,
                "jy": jy,
                "qx": qx,
                "qy": qy,
            }
        )

    # OPTIONAL: plot only first object’s derivatives (to keep it simple)
    if len(all_results) > 0:
        res0 = all_results[0]
        t = res0["times"]

        if len(t) > 1:
            plt.figure(figsize=(10, 12))

            plt.subplot(4, 1, 1)
            plt.plot(t, res0["vx"], label="vx")
            plt.plot(t, res0["vy"], label="vy")
            plt.title("Velocity vs time (object 0)")
            plt.xlabel("time [s]")
            plt.ylabel("velocity [px/s]")
            plt.legend()

            plt.subplot(4, 1, 2)
            plt.plot(t, res0["ax"], label="ax")
            plt.plot(t, res0["ay"], label="ay")
            plt.title("Acceleration vs time (object 0)")
            plt.xlabel("time [s]")
            plt.ylabel("acceleration [px/s^2]")
            plt.legend()

            plt.subplot(4, 1, 3)
            plt.plot(t, res0["jx"], label="jx")
            plt.plot(t, res0["jy"], label="jy")
            plt.title("Jerk vs time (object 0)")
            plt.xlabel("time [s]")
            plt.ylabel("jerk [px/s^3]")
            plt.legend()

            plt.subplot(4, 1, 4)
            plt.plot(t, res0["qx"], label="qx")
            plt.plot(t, res0["qy"], label="qy")
            plt.title("Jounce vs time (object 0)")
            plt.xlabel("time [s]")
            plt.ylabel("jounce [px/s^4]")
            plt.legend()

            plt.tight_layout()
            plt.show()

    return all_results


# ---------------------------------------------------------
# 3. Main
# ---------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    video_file = str(root / "data" / "fishess.mp4")   # adjust if needed
    out_video = str(root / "outputs" / "multi_centroid.mp4")

    # change n_objects to however many fish / moving blobs you expect
    results = process_video(video_file, out_video_path=out_video, n_objects=2)
    print("Saved result video to:", out_video)