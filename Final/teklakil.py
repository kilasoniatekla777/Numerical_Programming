def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from PIL import Image

    print("Running TEKLAKIL animation (physics-based)")


    # Load image and sample points

    img = Image.open("handwritten.png").convert("L").resize((300, 100))
    img_np = np.array(img)
    threshold = 120
    ys, xs = np.where(img_np < threshold)
    points = np.vstack([xs, ys]).T
    points = points - np.mean(points, axis=0)

    N = 450
    indices = np.linspace(0, len(points)-1, N).astype(int)
    targets = points[indices]


    # Initial positions (scattered)

    np.random.seed(7)
    positions = np.random.uniform([-400, -200], [400, 200], (N, 2))
    velocities = np.zeros_like(positions)


    # Physics parameters

    k_p = 1.0       # target tracking gain
    k_d = 0.2       # damping
    k_rep = 5.0     # weak repulsion
    R_safe = 5.0    # small safety distance
    vmax = 20.0     # max speed
    dt = 0.1
    steps = 400


    # Setup figure

    fig, ax = plt.subplots(figsize=(11, 4))
    scat = ax.scatter(positions[:, 0], -positions[:, 1], s=25, c="cyan")
    ax.set_xlim(-450, 450)   # all drones stay visible
    ax.set_ylim(-250, 250)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title("Drone Show — TEKLAKIL (Physics-based Motion)", fontsize=14)


    # Vectorized repulsion function

    def compute_repulsion(pos):
        diff = pos[:, None, :] - pos[None, :, :]         # shape (N, N, 2) This tells us direction and distance from drone j to drone i.

        dist = np.linalg.norm(diff, axis=2) + 1e-5       # Euclidean distance for each pair (i, j)
        mask = (dist < R_safe) & (dist > 0)              # Only drones that are closer than R_safe should repel each other.
        rep_vec = k_rep * diff / (dist[:, :, None]**3)
        rep_vec *= mask[:, :, None]                      # zero out outside radius
        repulsion = np.sum(rep_vec, axis=1)              # Each drone i feels the sum of all repulsion vectors from nearby drones.
        return repulsion


    # Animation update

    def update(frame):
        nonlocal positions, velocities

        # Target tracking
        a_target = k_p * (targets - positions)  # k_p determines how strongly it moves toward the target.

        # Damping
        a_damping = -k_d * velocities           #Slows down the drone as it approaches the target.

        # Collision avoidance
        a_repulsion = compute_repulsion(positions)     #Checks if drones are too close to each other.

        # Total acceleration
        acceleration = a_target + a_damping + a_repulsion     #This gives the final force on each drone this frame.


        # Update velocity
        velocities += acceleration * dt                      #Speed changes depending on forces acting on me.

        # Velocity saturation
        speeds = np.linalg.norm(velocities, axis=1)    #limit speed
        mask = speeds > vmax
        velocities[mask] = velocities[mask] / speeds[mask][:, None] * vmax

        # Update positions
        positions += velocities * dt                         #New position = old position + how far the drone moved this frame

        scat.set_offsets(np.c_[positions[:, 0], -positions[:, 1]])
        return scat,


    # Run animation

    ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    main()