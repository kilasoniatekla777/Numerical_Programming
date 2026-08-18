def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from PIL import Image, ImageDraw, ImageFont

    print("Running Drone Show: TEKLAKIL → Happy New Year! (Stable Smooth Interpolation)")


    # sample N points from image pixels

    def sample_points(img_np, N, threshold=120):
        ys, xs = np.where(img_np < threshold)
        all_points = np.vstack([xs, ys]).T
        if len(all_points) >= N:
            indices = np.random.choice(len(all_points), N, replace=False)
        else:
            indices = np.random.choice(len(all_points), N, replace=True)
        points = all_points[indices]
        points = points - np.mean(points, axis=0)  # center
        return points.astype(float)


    # Smooth easing function

    def ease_in_out_cubic(t):
        """Smooth S-curve easing"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            p = 2 * t - 2
            return 1 + p * p * p / 2


    # Load images

    name_img = Image.open("handwritten.png").convert("L").resize((300, 100))
    name_np = np.array(name_img)
    N = 700
    name_targets = sample_points(name_np, N, threshold=120)

    try:
        greeting_img = Image.open("happy_new_year.png").convert("L").resize((700, 200))
    except FileNotFoundError:
        greeting_img = Image.new("L", (700, 200), color=255)
        draw = ImageDraw.Draw(greeting_img)
        try:
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        draw.text((10, 80), "Happy New Year!", fill=0, font=font)

    greeting_np = np.array(greeting_img)
    greeting_targets = sample_points(greeting_np, N, threshold=120)


    # Add smooth paths for each drone

    # Add some randomness to make paths more interesting
    np.random.seed(42)
    mid_offsets = np.random.randn(N, 2) * 30  # Random mid-point offsets

    # Calculate midpoint for each drone
    midpoints = (name_targets + greeting_targets) / 2 + mid_offsets

    # Initialize positions

    positions = name_targets.copy()

    # Parameters

    total_frames = 300
    hold_start = 30  # Hold initial position
    transition_start = hold_start
    transition_end = 250
    hold_end = total_frames  # Hold final position

    # Figure setup

    fig, ax = plt.subplots(figsize=(12, 5))
    scat = ax.scatter(positions[:, 0], -positions[:, 1], s=30, c="cyan",
                      edgecolors='#00ffff', linewidths=0.3, alpha=0.9)
    ax.set_xlim(-450, 450)
    ax.set_ylim(-250, 250)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('#000814')
    fig.patch.set_facecolor('#001220')
    ax.set_frame_on(False)

    title = ax.text(0, 230, "TEKLAKIL", fontsize=16, color='white',
                    ha='center', weight='bold')


    # Update function using smooth interpolation

    def update(frame):
        nonlocal positions

        if frame < transition_start:
            # Hold at start position
            positions = name_targets.copy()
            title.set_text("TEKLAKIL")
            alpha = 0

        elif frame < transition_end:
            # Smooth transition
            progress = (frame - transition_start) / (transition_end - transition_start)
            alpha = ease_in_out_cubic(progress)

            # Use Bezier-like curve through midpoint for smooth arcs
            # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
            t = alpha
            positions = (1 - t) ** 2 * name_targets + 2 * (1 - t) * t * midpoints + t ** 2 * greeting_targets

            # Update title based on progress
            if alpha < 0.3:
                title.set_text("TEKLAKIL")
            elif alpha < 0.7:
                title.set_text("Transforming...")
            else:
                title.set_text("Happy New Year!")

        else:
            # Hold at end position
            positions = greeting_targets.copy()
            title.set_text("Happy New Year!")
            alpha = 1

        # Update scatter plot
        scat.set_offsets(np.c_[positions[:, 0], -positions[:, 1]])

        # Add slight glow effect based on motion
        if 0 < alpha < 1:
            scat.set_alpha(0.85 + 0.15 * np.sin(alpha * np.pi))
        else:
            scat.set_alpha(0.9)

        return scat, title


    # Run animation (NO LOOP)

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=33,  # ~30 fps
        blit=True,
        repeat=False
    )

    plt.tight_layout()
    plt.show()
    print("Animation complete!")


if __name__ == "__main__":
    main()