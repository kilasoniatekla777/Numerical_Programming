# Numerical Programming

A collection of numerical methods and scientific computing projects: derivatives, splines, ODE solvers, clustering, computational geometry, and computer-vision-driven simulations. Each project lives in its own folder with its own `README.md`, source code, and (where applicable) sample data and generated outputs.

## Projects

| Folder | Project | Topics |
|---|---|---|
| [AP1](AP1/) | [Vector & Matrix Norms](AP1/README.md) | L1/L2/Frobenius norms, unit ball visualization |
| [AP2](AP2/) | [Food Clustering](AP2/README.md) | K-Means, DBSCAN, PCA |
| [AP3](AP3/) | [Numerical Differentiation: Tangents & Normals](AP3/README.md) | Finite differences, tangent lines & planes |
| [AP4](AP4/) | [Voronoi & Delaunay Diagrams of Music Artists](AP4/README.md) | Computational geometry |
| [AP5](AP5/) | [Parametric Splines for Letter Shapes](AP5/README.md) | Linear/cubic/B-splines, curvature analysis |
| [AP6](AP6/) | [3D Candle Volume Reconstruction](AP6/README.md) | Edge detection, spline fitting, numerical integration |
| [CP1](CP1/) | [Video Object Tracking & Motion Derivatives](CP1/README.md) | Optical motion detection, finite differences, KMeans |
| [CP2](CP2/) | [Evaporation Pond ODE Simulation](CP2/README.md) | Implicit Euler, DIRK2, Newton vs. fixed-point iteration |
| [Final](Final/) | [Physics-Based Drone Swarm Animations](Final/README.md) | PD control, particle simulation, animation |

## Setup

Each project folder has its own dependencies, listed in that folder's README. In general, across the repo you'll need:

```bash
pip install numpy scipy pandas matplotlib scikit-learn opencv-python pillow
```

Then run any project's main script from inside its folder, e.g.:

```bash
cd AP5
python main.py
```

## Notes

- Every project folder includes a `README.md` explaining what it does, how to run it, and what output to expect.
- `AP4/tracks.csv` (~111 MB) is excluded from this repo since it exceeds GitHub's file size limit — see [AP4/README.md](AP4/README.md) for how to obtain it.
- Virtual environments (`venv/`), IDE files (`.idea/`), and Python caches are excluded via `.gitignore`.
