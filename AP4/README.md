# AP4 — Voronoi & Delaunay Diagrams of Music Artists

Builds a 2D "artist space" from a Spotify tracks dataset and visualizes it with Voronoi and Delaunay diagrams.

## What it does

- Loads `tracks.csv` and keeps the `artists`, `energy`, and `acousticness` columns.
- Selects the top 10 artists by track count (kept the point count low enough that the diagrams don't crash).
- Averages `energy` and `acousticness` per artist.
- Computes a Voronoi diagram and a Delaunay triangulation over the artist points.
- Plots both overlaid, with each artist labeled.

## Run it

```bash
pip install pandas matplotlib scipy
python ap4.py
```

## Data

`tracks.csv` (~111 MB) is **not included** in this repo — it exceeds GitHub's 100 MB file limit and is listed in `.gitignore`. To run this script, place a Spotify tracks CSV with `artists`, `energy`, and `acousticness` columns in this folder as `tracks.csv`.

## Output

A matplotlib figure titled "Music Artist Space - Top 10 Artists" showing the Voronoi cells, Delaunay edges, and labeled artist points.

`ap4.pdf` in this folder is the write-up/report for the assignment.
