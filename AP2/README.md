# AP2 — Food Clustering (K-Means & DBSCAN)

Clusters foods by their nutritional profile using two different clustering algorithms and visualizes the results in 2D via PCA.

## What it does

- Loads `FOOD-DATA-GROUP1.csv` and selects the numeric nutrition columns.
- Standardizes the features with `StandardScaler`.
- Clusters foods with **K-Means** (k=3) and **DBSCAN** (eps=1.5, min_samples=5).
- Prints which foods fall into each cluster for both methods.
- Projects the scaled features to 2D with PCA and plots the K-Means and DBSCAN clusters side by side.

## Run it

```bash
pip install pandas scikit-learn matplotlib
python main.py
```

## Data

`FOOD-DATA-GROUP1.csv` — nutritional data per food item (included in this folder).

## Output

Console output listing cluster membership per food, plus a matplotlib figure comparing the K-Means and DBSCAN cluster assignments in PCA space.
