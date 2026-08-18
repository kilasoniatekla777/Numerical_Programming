import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def main():
    # -----------------------------
    # Parameters
    # -----------------------------
    total_frames = 300  # more frames = slower animation
    interval = 50       # milliseconds per frame
    ball_radius = 20

    # Start and end positions
    start_pos = np.array([-350, -300])
    end_pos = np.array([350, 300])

    # Storage for positions
    positions_list = []

    # -----------------------------
    # Setup figure
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-400, 400)
    ax.set_ylim(-350, 350)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title("Slow Motion Ball: Bottom-Left → Top-Right", fontsize=16)

    # Ball plot
    ball_plot = ax.scatter(*start_pos, s=ball_radius**2, c="black")

    # -----------------------------
    # Smooth interpolation function
    # -----------------------------
    def smooth_step(t):
        return t * t * (3 - 2 * t)  # smoother than linear

    # -----------------------------
    # Animation update
    # -----------------------------
    def update(frame_num):
        t = frame_num / total_frames
        t = smooth_step(min(t, 1.0))
        pos = (1 - t) * start_pos + t * end_pos

        positions_list.append(pos.copy())

        ball_plot.set_offsets(pos)
        return ball_plot,

    # -----------------------------
    # Run animation
    # -----------------------------
    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=True,
        cache_frame_data=False
    )

    # -----------------------------
    # Save to MP4
    # -----------------------------
    writer = FFMpegWriter(fps=20)
    ani.save("slow_ball.mp4", writer=writer)
    print("Saved animation as slow_ball.mp4!")

    # -----------------------------
    # Return positions for later use
    # -----------------------------
    positions_array = np.array(positions_list)
    return positions_array

# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    ball_positions = main()