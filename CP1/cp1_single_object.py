from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

def compute_derivatives(positions: np.ndarray, times: np.ndarray):
    positions = np.asarray(positions, dtype=float)
    times = np.asarray(times, dtype=float)

    N = len(times)
    if N < 2:
        # Return 8 zero arrays: vx, vy, ax, ay, jx, jy, qx, qy
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

    for i in range(1, N):
        vx[i] = (positions[i, 0] - positions[i - 1, 0]) / dt[i - 1]
        vy[i] = (positions[i, 1] - positions[i - 1, 1]) / dt[i - 1]

    for i in range(1, N):
        ax[i] = (vx[i] - vx[i - 1]) / dt[i - 1]
        ay[i] = (vy[i] - vy[i - 1]) / dt[i - 1]

    for i in range(1, N):
        jx[i] = (ax[i] - ax[i - 1]) / dt[i - 1]
        jy[i] = (ay[i] - ay[i - 1]) / dt[i - 1]

    for i in range(1, N):
        qx[i] = (jx[i] - jx[i - 1]) / dt[i - 1]
        qy[i] = (jy[i] - jy[i - 1]) / dt[i - 1]

    return vx, vy, ax, ay, jx, jy, qx, qy

def process_video(video_path: str, out_video_path: str | None = None):
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

    # first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        if writer is not None:
            writer.release()
        raise RuntimeError("Could not read first frame.")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gx = cv2.Sobel(gray_prev, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_prev, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges_prev = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)

    positions = []
    times = []
    frame_idx = 0

    last_center = None  # remember last centroid

    while True:
        ret, frame = cap.read()
        positions = []
        times = []
        frame_idx = 0
        last_center = None

        while True:
            ret, frame = cap.read()
            positions = []
            times = []
            frame_idx = 0
            last_center = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # --- grayscale + stronger smoothing for cleaner edges ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # --- Sobel edges ---
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = cv2.magnitude(gx, gy)
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # slightly higher threshold → thinner, cleaner edges
                _, edge_mask = cv2.threshold(mag, 100, 255, cv2.THRESH_BINARY)

                # --- frame differencing for motion ---
                diff_gray = cv2.absdiff(gray, gray_prev)
                _, motion_raw = cv2.threshold(diff_gray, 12, 255, cv2.THRESH_BINARY)

                # moving edges = motion ∧ edges
                motion_mask = cv2.bitwise_and(motion_raw, edge_mask)

                # morphology: open → remove specks, close → fill tiny gaps
                kernel = np.ones((3, 3), np.uint8)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

                # centroid of all white pixels
                ys, xs = np.where(motion_mask > 0)
                if len(xs) > 0:
                    cx = xs.mean()
                    cy = ys.mean()
                    last_center = (float(cx), float(cy))

                # show mask in color so red dot is visible
                motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

                if last_center is not None:
                    cx, cy = int(round(last_center[0])), int(round(last_center[1]))

                    # SMALLER red dot (radius 5) with thin green outline
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # filled red
                    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), 1)  # thin green ring

                    cv2.circle(motion_bgr, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.circle(motion_bgr, (cx, cy), 7, (0, 255, 0), 1)

                    t = frame_idx * dt
                    positions.append([cx, cy])
                    times.append(t)

                cv2.imshow("Result with centroid", frame)
                cv2.imshow("Motion mask", motion_bgr)

                if writer is not None:
                    writer.write(frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

                gray_prev = gray.copy()
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
    print("vx, vy:", vx.shape, vy.shape)
    print("ax, ay:", ax.shape, ay.shape)
    print("jx, jy:", jx.shape, jy.shape)
    print("qx, qy:", qx.shape, qy.shape)

    # plots (same idea as before)
    if len(times) > 1:
        t = times
        plt.figure(figsize=(10, 12))

        plt.subplot(4, 1, 1)
        plt.plot(t, vx, label="vx")
        plt.plot(t, vy, label="vy")
        plt.title("Velocity vs time")
        plt.xlabel("time [s]")
        plt.ylabel("velocity [px/s]")
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t, ax, label="ax")
        plt.plot(t, ay, label="ay")
        plt.title("Acceleration vs time")
        plt.xlabel("time [s]")
        plt.ylabel("acceleration [px/s^2]")
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(t, jx, label="jx")
        plt.plot(t, jy, label="jy")
        plt.title("Jerk vs time")
        plt.xlabel("time [s]")
        plt.ylabel("jerk [px/s^3]")
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t, qx, label="qx")
        plt.plot(t, qy, label="qy")
        plt.title("Jounce vs time")
        plt.xlabel("time [s]")
        plt.ylabel("jounce [px/s^4]")
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    video_file = str(root  / "videoplayback.mp4")            # change if name different
    out_video = str(root / "single_centroid.mp4")
    process_video(video_file, out_video_path=out_video)
    print("Saved result video to:", out_video)