import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Parameters
# -----------------------------
N = 600
dt = 0.05
steps = 400

kv = 2.0       # velocity tracking gain
kd = 1.2       # damping
vmax = 25.0    # speed limit

# -----------------------------
# Load Happy New Year shape
# -----------------------------
try:
    img = Image.open("happy_new_year.png").convert("L").resize((700, 200))
except:
    img = Image.new("L", (700, 200), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default()
    d.text((20, 80), "Happy New Year!", fill=0, font=f)

img_np = np.array(img)
ys, xs = np.where(img_np < 120)
points = np.vstack([xs, ys]).T
points = points.astype(float)
points -= np.mean(points, axis=0)

idx = np.linspace(0, len(points) - 1, N).astype(int)
shape = points[idx]

# -----------------------------
# Initial state
# -----------------------------
positions = shape.copy()
velocities = np.zeros_like(positions)

# -----------------------------
# Velocity field (moving ball)
# -----------------------------
def velocity_field(t):
    center = np.array([
        200 * np.sin(0.02 * t),
        80 * np.cos(0.02 * t)
    ])
    return center

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 5))
scat = ax.scatter(positions[:, 0], -positions[:, 1], s=20, c="cyan")
ax.set_xlim(-450, 450)
ax.set_ylim(-250, 250)
ax.axis("off")
ax.set_title("Happy New Year — Dynamic Tracking (IVP-VT)")

# -----------------------------
# Animation update (Explicit Euler)
# -----------------------------
def update(frame):
    global positions, velocities

    # Desired velocity field (same for all drones)
    V = velocity_field(frame)
    Vsat = V * min(1, vmax / (np.linalg.norm(V) + 1e-6))

    # IVP-VT acceleration
    a = kv * (Vsat - velocities) - kd * velocities

    # Explicit Euler
    velocities += a * dt

    # Speed saturation
    speeds = np.linalg.norm(velocities, axis=1)
    mask = speeds > vmax
    velocities[mask] *= vmax / speeds[mask][:, None]

    positions += velocities * dt

    scat.set_offsets(np.c_[positions[:, 0], -positions[:, 1]])
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=True)
plt.show()