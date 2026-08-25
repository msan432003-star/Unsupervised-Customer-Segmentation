# Customer Segmentation — Unsupervised Learning & Clustering Analysis

Week 3 task: segmenting 300 customers into behaviorally meaningful groups using
K-Means clustering, validated with Agglomerative Hierarchical Clustering.

## Objective

Identify actionable customer personas based on income and spending behavior,
using unsupervised learning — no labeled target variable is available or needed.

## Dataset

`customer_segmentation_dataset.csv` — 300 rows, 4 columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier |
| `Age` | Customer age (18–69) |
| `Annual_Income_k$` | Annual income in $ thousands |
| `Spending_Score` | Composite spending/loyalty score (1–100) |

No missing values, no duplicate rows.

## Method

1. **Preprocessing** — standardized `Annual_Income_k$` and `Spending_Score` with
   `StandardScaler` so both features contribute equally to distance calculations.
2. **Choosing K** — swept K = 2–10 and evaluated with both:
   - **Elbow method** (within-cluster sum of squares)
   - **Silhouette score**

   Both independently pointed to **K = 5** (WCSS flattens after K=5; silhouette
   peaks at 0.834 at K=5).
3. **Clustering** — `KMeans(n_clusters=5, random_state=42, n_init=10)`.
4. **Validation** — Agglomerative Hierarchical Clustering (Ward linkage) on the
   same standardized features, to check the result isn't an artifact of
   K-Means' spherical-cluster assumption. The dendrogram shows the same
   ~5-branch structure, corroborating the K-Means result.

## Results — 5 Customer Segments

| Cluster | Segment | n | Avg. Age | Avg. Income ($k) | Avg. Spending Score |
|---|---|---|---|---|---|
| 0 | Budget-Conscious | 60 | 46.2 | 24.8 | 20.7 |
| 1 | Premium Target | 45 | 41.9 | 84.9 | 84.6 |
| 2 | Aspirational Spenders | 60 | 45.2 | 24.5 | 79.9 |
| 3 | Conservative Affluent | 45 | 44.0 | 84.8 | 14.3 |
| 4 | Standard / Average | 90 | 43.3 | 55.5 | 49.9 |

Age was checked across clusters and found not to vary meaningfully (41.9–46.2
across all five groups), confirming income and spending score — not age — are
the meaningful axes of segmentation in this dataset.

## Repository Structure

```
├── customer_segmentation_dataset.csv   # source data
├── clustering_analysis.ipynb           # full analysis notebook
├── cluster_optimization.png            # elbow + silhouette plots
├── kmeans_clusters.png                 # final K-Means segmentation plot
├── dendrogram.png                      # hierarchical clustering validation
├── Week3_Clustering_Analysis_Report.docx  # full written report
└── README.md
```

## Tech Stack

Python 3 · pandas · scikit-learn · SciPy · Matplotlib · Seaborn

## Key Takeaway

Two independent methods for choosing K and two independent clustering
algorithms converge on the same five-segment structure, giving confidence
this reflects real structure in customer behavior rather than an artifact of
one algorithm's assumptions. The segments map directly onto marketing
strategy — e.g., **Conservative Affluent** customers (high income, low
spend) are the strongest win-back target, while **Premium Target** customers
are the strongest retention/loyalty investment.
