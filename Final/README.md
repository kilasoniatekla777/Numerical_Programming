# Final — Physics-Based Drone Swarm Animations

A collection of particle-swarm animations where hundreds of simulated "drones" are steered by a spring-damper-repulsion physics model to form text, letters, and shapes, plus a simple ball-trajectory animation.

## What it does

Each drone is treated as a point mass pulled toward a target position by a proportional-derivative (PD) controller, with a repulsion force between nearby drones to prevent collisions and velocity clamping for realism:

```
F = k_p * (target - position) - k_d * velocity + repulsion(neighbors)
```

Target positions are sampled from the dark pixels of a source image (handwriting or rendered text), so the swarm morphs between shapes over time.

| File | Description |
|---|---|
| `teklakil.py` | Drones scattered randomly converge to form the handwritten "TEKLAKIL" signature (from `handwritten.png`). |
| `teklakil2.py` | Variant with different physics tuning (velocity tracking gain/damping) for the same signature animation. |
| `happy.py` | Drones morph from the "TEKLAKIL" shape into "Happy New Year!" text. |
| `happynew.py` | Variant of the TEKLAKIL → "Happy New Year!" transition with smoother interpolation between shapes. |
| `ball.py` | Drones form a solid ball shape and follow a curved trajectory across the scene while also spelling "Happy New Year!". |
| `video.py` | Simple non-swarm animation: a single ball moving smoothly (ease in/out) across the frame, saved to `slow_ball.mp4`. |
| `main.py` | Master entry point; imports all the above and runs the ball animation by default. |

## Run it

```bash
pip install numpy matplotlib pillow
python main.py          # runs the ball animation
python happy.py         # or run any script individually
```

## Assets

- `handwritten.png` — source image for the "TEKLAKIL" signature shape
- `happy_new_year.png` — source image for the "Happy New Year!" text shape
- `final.pdf`, `final2.pdf` — write-up/report for the assignment

## Output

Pre-rendered results included in this folder: `ball.mp4`, `slow_ball.mp4`, `happy_new_year_animation.mp4`, `happy_new_year_drone_on_ball_path.mp4`. Running a script directly opens a live matplotlib animation window.
