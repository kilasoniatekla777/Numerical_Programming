import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont


def main():

    # IVP-VT Physics Parameters
    N = 1000
    kv = 15.0
    kd = 5.0
    vmax = 60.0
    m = 1.0
    dt = 0.05


    # Setup Targets: Happy New Year
    def sample_points(text, N, size=(700, 200)):
        img = Image.new("L", size, 255)
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.load_default()
        except:
            f = None
        d.text((20, 60), text, fill=0, font=f)
        img_np = np.array(img)
        ys, xs = np.where(img_np < 120)
        idx = np.random.choice(len(xs), N, replace=True)
        pts = np.vstack([xs[idx], ys[idx]]).T
        pts = pts - np.mean(pts, axis=0)
        return pts.astype(float)

    text_targets = sample_points("Happy New Year!", N)

    # Initial state
    xi = text_targets.copy()
    vi = np.zeros_like(xi)


    # Ball Trajectory & SHAPE DEFINITION

    start_pos = np.array([-350, 150])    #We are making a ball of drones.
    end_pos = np.array([350, -150])
    ball_radius = 60.0

    # --- FIX: Define the ball structure ONCE here, not in the loop ---
    # We generate relative offsets from the center (0,0)
    ball_angles = np.random.uniform(0, 2 * np.pi, N)

    # improved math: sqrt(random) makes the distribution uniform across the circle area
    # otherwise it clumps too much in the center
    ball_radii = ball_radius * np.sqrt(np.random.uniform(0, 1, N))

    # These offsets stay constant for each specific drone
    ball_offsets = np.column_stack([ball_radii * np.cos(ball_angles), ball_radii * np.sin(ball_angles)])


    # Animation Setup

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#000814')
    fig.patch.set_facecolor('#000814')
    scat = ax.scatter(xi[:, 0], -xi[:, 1], s=15, c="cyan", edgecolors='white', linewidths=0.2)

    ax.set_xlim(-450, 450)
    ax.set_ylim(-250, 250)
    ax.axis("off")

    total_frames = 600
    transition_frame = 40

    def update(frame):
        nonlocal xi, vi

        # 1. Determine Target Field
        if frame < transition_frame:
            target_positions = text_targets
        else:
            t = (frame - transition_frame) / (total_frames - transition_frame)

            # Calculate current center of the ball
            center_ball = start_pos + (end_pos - start_pos) * t

            # --- FIX: Apply the PRE-CALCULATED offsets to the moving center ---
            # This ensures drone[0] always goes to the same spot relative to the center
            target_positions = center_ball + ball_offsets

        # 2. IVP-VT Dynamics
        V_field = (target_positions - xi)   #Which way should this drone go
        norm_V = np.linalg.norm(V_field, axis=1, keepdims=True)  # distance from each drone to its target.
        norm_V[norm_V == 0] = 1e-6

        V_sat = V_field * np.minimum(1, vmax / norm_V)   #Makes sure drones don’t go faster than vmax.
        vi_dot = (1.0 / m) * (kv * (V_sat - vi) - kd * vi)  #how much the drone’s velocity changes this frame.

        vi += vi_dot * dt  #new velocity.

        norm_vi = np.linalg.norm(vi, axis=1, keepdims=True)
        norm_vi[norm_vi == 0] = 1e-6
        xi_dot = vi * np.minimum(1, vmax / norm_vi)   # how much the position should change this frame.

        xi += xi_dot * dt  #Move the drones by their velocity → new position.

        scat.set_offsets(np.c_[xi[:, 0], -xi[:, 1]])
        return scat,

    ani = FuncAnimation(fig, update, frames=total_frames, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()