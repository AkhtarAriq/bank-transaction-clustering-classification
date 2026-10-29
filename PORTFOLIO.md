# Portfolio copy

## Project title

Bank Transaction Clustering & Classification

## Public project summary

Developed a reproducible machine-learning study to segment bank transactions and predict the resulting cluster assignments. Explored 2,537 records and retained 1,945 samples through validated cleaning, outlier filtering, categorical encoding, scaling, and age-group feature engineering. Compared K-Means cluster counts, retained a three-cluster reference with a silhouette score of 0.495, and examined PCA projections, loadings, and readable cluster profiles. Built complete Decision Tree and Random Forest pipelines with training-only categorical preprocessing and five-fold hyperparameter tuning. Although the classifiers achieved perfect agreement on the 389-row classification holdout, a diagnostic showed that location encoding largely determined the labels. The project demonstrates reproducible experimentation, model evaluation, artifact management, and critical interpretation of machine-learning results.

## Short card explanation

Explored bank transaction clusters with reproducible pipelines and diagnosed the limits of perfect cluster-label prediction.

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook, JupyterLab.

## Suggested showcase order

| Placement | Output | What it communicates |
| --- | --- | --- |
| Cover | [PCA cluster projection](assets/pca-clusters.png) | Three distinct groups, centroid positions, and explained variance; discuss the location-dominated first component |
| Gallery | [Cluster-count comparison](assets/kmeans-elbow.png) | Inertia and silhouette evidence; two clusters score best while three are retained as the reference experiment |
| Gallery | [Correlation heatmap](assets/correlation-heatmap.png) | Exploratory relationships among numerical features |
| Gallery | [Feature distributions](assets/numerical-distributions.png) and [occupation boxplot](assets/transaction-amount-by-occupation.png) | Distribution shape and the motivation for examining preprocessing choices |
| Evaluation | [Model comparison](assets/classification-model-comparison.png) and [confusion matrices](assets/classification-confusion-matrices.png) | Actual held-out cluster-label predictions, presented with their limitations |
| Interpretation | [Cluster summary](reports/cluster_summary.csv) and the location diagnostic in the classification notebook | Measured differences in original units and an explanation for perfect label agreement |
| Reproducibility | Both notebooks, [run metadata](reports/clustering_metadata.json), [search results](reports/grid_search_results.csv), tests, and fitted model artifacts | A traceable workflow that another reader can execute and inspect |

## Presentation guidance

- Identify the task as segmentation and prediction of generated cluster labels. No verified fraud target is present.
- Perfect classification agreement is not evidence of a validated financial predictor. All 43 location categories map to one cluster each in the current run.
- Explain that the original integer category encoding strongly shapes the clusters, and treat alternative feature representations as future work.
- Use the current reports when citing results: the portfolio edition refits K-Means with ten initializations and fits classification encoders inside cross-validation.
- There is no live application or production inference service. Link to the executable notebooks and README.
