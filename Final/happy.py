def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from PIL import Image, ImageDraw, ImageFont

    print("Running TEKLAKIL → Happy New Year Animation (Physics-based)")


    # Parameters

    N = 450     # number of drones in the swarm
    dt = 0.1    # time step for each frame
    steps = 400  # total number of animation frames
    k_p = 1.0   # how strongly drones are pulled toward their target
    k_d = 1.1   # damping to slow drones down and prevent overshoot
    k_rep = 1.1  # repulsion strength to keep drones from colliding
    R_safe = 5.0  # minimum distance between drones for repulsion to act
    vmax = 20.0  #  Maximum speed keeps the drones moving smoothly, safely, and realistically.


    # Load "TEKLAKIL" starting points

    name_img = Image.open("handwritten.png").convert("L").resize((300, 100))
    name_np = np.array(name_img)
    threshold = 120
    ys, xs = np.where(name_np < threshold)
    points = np.vstack([xs, ys]).T
    points = points - np.mean(points, axis=0)
    indices = np.linspace(0, len(points)-1, N).astype(int)
    start_targets = points[indices]


    # Load "Happy New Year!" points

    try:
        greeting_img = Image.open("happy_new_year.png").convert("L").resize((700, 200))
    except FileNotFoundError:
        greeting_img = Image.new("L", (700, 200), color=255)
        draw = ImageDraw.Draw(greeting_img)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = ImageFont.load_default()
        draw.text((10, 80), "Happy New Year!", fill=0, font=font)

    greeting_np = np.array(greeting_img)
    ys, xs = np.where(greeting_np < threshold)
    greeting_points = np.vstack([xs, ys]).T
    greeting_points = greeting_points - np.mean(greeting_points, axis=0)
    indices = np.linspace(0, len(greeting_points)-1, N).astype(int)
    end_targets = greeting_points[indices]


    # Initialize positions at TEKLAKIL

    positions = start_targets.copy()
    velocities = np.zeros_like(positions)
    current_targets = start_targets.copy()  # initially, drones aim at "TEKLAKIL"


    # Setup figure

    fig, ax = plt.subplots(figsize=(12, 5))
    scat = ax.scatter(positions[:, 0], -positions[:, 1], s=25, c="cyan")
    ax.set_xlim(-450, 450)
    ax.set_ylim(-250, 250)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title("Drone Show — TEKLAKIL → Happy New Year", fontsize=14)


    # Repulsion (vectorized)

    def compute_repulsion(pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + 1e-5
        mask = (dist < R_safe) & (dist > 0)
        rep_vec = k_rep * diff / (dist[:, :, None]**3)
        rep_vec *= mask[:, :, None]
        repulsion = np.sum(rep_vec, axis=1)
        return repulsion

    # Animation update
    transition_frame = 50  # frame when drones start moving to greeting

    def update(frame):
        nonlocal positions, velocities, current_targets

        # Change targets at transition
        if frame == transition_frame:
            current_targets = end_targets.copy()

        # Physics
        a_target = 0.5 * k_p * (current_targets - positions)  # weaker force
        a_damping = -0.5 * k_d * velocities  # stronger damping
        a_repulsion = compute_repulsion(positions)

        acceleration = a_target + a_damping + a_repulsion  #The drone now knows how it should change its speed this frame.


        velocities += acceleration * dt # Updates each drone’s velocity using Euler’s method:

        # Velocity saturation
        speeds = np.linalg.norm(velocities, axis=1)  #Calculates how fast each drone is moving (the length of its velocity vector).
        mask = speeds > vmax  #	Finds which drones are moving faster than the allowed maximum speed.
        velocities[mask] = velocities[mask] / speeds[mask][:, None] * vmax  #	Slows down the drones that are too fast so their speed equals vmax.

        positions += velocities * dt #	•	Moves each drone according to its velocity for this frame.

        scat.set_offsets(np.c_[positions[:, 0], -positions[:, 1]])  #Updates the animation to show the drones at their new positions.
        return scat,

    ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()