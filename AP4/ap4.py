import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay


df = pd.read_csv("tracks.csv")
print("Columns:", df.columns)
print(df.head())


df = df[['artists', 'energy', 'acousticness']]


# i Picked top 10 artists by number of tracks  to ensure the Voronoi and Delaunay diagrams have enough points to plot as it was crashing
artist_counts = df['artists'].value_counts()
top_artists = artist_counts.head(10).index
df = df[df['artists'].isin(top_artists)]
print("Top artists included:", list(top_artists))


artist_df = df.groupby('artists').mean().reset_index()
print("Number of artists after selection:", len(artist_df))

points = artist_df[['energy', 'acousticness']].to_numpy()
labels = artist_df['artists'].tolist()


if len(points) == 0:
    print("No points to plot. Try selecting a different column or top N artists.")
else:
     # computin Voronoi & Delaunay
    vor = Voronoi(points)
    tri = Delaunay(points)


    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Music Artist Space - Top 10 Artists", fontsize=14)

    #  Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black',
                    line_width=1, line_alpha=0.7, point_size=0)

    # Delaunay triangulation
    ax.triplot(points[:, 0], points[:, 1], tri.simplices.copy(),
               color='gray', linestyle='--', linewidth=1)

    #  points
    ax.scatter(points[:, 0], points[:, 1], c='red', s=80, zorder=3)


    for i, label in enumerate(labels):
        ax.text(points[i, 0]+0.01, points[i, 1]+0.01, label,
                fontsize=10, weight='bold')


    ax.set_xlabel("Energy", fontsize=12)
    ax.set_ylabel("Acousticness", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.show()
