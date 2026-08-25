# Customer Segmentation — Unsupervised Learning & Clustering Analysis

Week 3 task: segmenting real mall customers into behaviorally meaningful groups
using K-Means clustering, validated with Agglomerative Hierarchical Clustering.

## Objective

Identify actionable customer personas based on income and spending behavior,
using unsupervised learning — no labeled target variable is available or needed.

## Dataset

[**Mall Customer Segmentation Data**](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
(Kaggle, by Vijay Choudhary) — `Mall_Customers.csv`, 200 rows, 5 columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier |
| `Gender` | Male / Female |
| `Age` | Customer age (18–70) |
| `Annual Income (k$)` | Annual income in $ thousands (~15–137) |
| `Spending Score (1-100)` | Composite spending/loyalty score assigned by the mall |

No missing values, no duplicate rows.

## Method

1. **Preprocessing** — standardized `Annual_Income_k$` and `Spending_Score` with
   `StandardScaler` so both features contribute equally to distance calculations.
2. **Choosing K** — swept K = 2–10 and evaluated with both:
   - **Elbow method** (within-cluster sum of squares)
   - **Silhouette score**

   Both independently pointed to **K = 5** (WCSS drop-off softens sharply
   after K=5; silhouette peaks at 0.555 at K=5).
3. **Clustering** — `KMeans(n_clusters=5, random_state=42, n_init=10)`.
4. **Validation** — Agglomerative Hierarchical Clustering (Ward linkage) on the
   same standardized features, to check the result isn't an artifact of
   K-Means' spherical-cluster assumption. The dendrogram shows the same
   ~5-branch structure, and a cross-tabulation of the two algorithms' labels
   confirms strong agreement.

## Results — 5 Customer Segments

| Cluster | Segment | n | Avg. Age | Avg. Income ($k) | Avg. Spending Score |
|---|---|---|---|---|---|
| 0 | Standard / Average | 81 | 42.7 | 55.3 | 49.5 |
| 1 | Premium Target | 39 | 32.7 | 86.5 | 82.1 |
| 2 | Young Aspirational Spenders | 22 | 25.3 | 25.7 | 79.4 |
| 3 | Conservative Affluent | 35 | 41.1 | 88.2 | 17.1 |
| 4 | Budget-Conscious | 23 | 45.2 | 26.3 | 20.9 |

Unlike a synthetic dataset, this real data shows a genuine age signal: Cluster 2
(low income, high spend) is markedly younger on average (25.3) than every other
segment (32.7–45.2) — a real, data-driven finding rather than something built
into the data by construction.

## Repository Structure

```
├── Mall_Customers.csv                     # source data (Kaggle, public)
├── clustering_analysis.ipynb              # full analysis notebook, executed with outputs
├── mall_customers_clustered.csv           # output data with cluster labels
├── cluster_optimization.png               # elbow + silhouette plots
├── kmeans_clusters.png                    # final K-Means segmentation plot
├── dendrogram.png                         # hierarchical clustering validation
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
spend) are the strongest win-back target, **Premium Target** customers are
the strongest retention/loyalty investment, and **Young Aspirational
Spenders** — a pattern only visible because real data was used — point to a
distinct youth-focused campaign opportunity.
