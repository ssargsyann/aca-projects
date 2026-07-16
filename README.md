# 🎯 Customer Segmentation with Unsupervised Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2.2-150458?logo=pandas&logoColor=white)
![UMAP](https://img.shields.io/badge/UMAP-0.5.6-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> An end-to-end unsupervised machine learning pipeline that transforms raw, unlabeled retail customer data into actionable marketing personas — using PCA, K-Means, Gaussian Mixture Models, and UMAP.

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Dimensionality Reduction (PCA)](#2-dimensionality-reduction-pca)
  - [3. Clustering (K-Means)](#3-clustering-k-means)
  - [4. Advanced Techniques (GMM & UMAP)](#4-advanced-techniques-gmm--umap)
- [Results: Customer Personas](#-results-customer-personas)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Key Engineering Decisions](#-key-engineering-decisions)
- [Future Improvements](#-future-improvements)

---

## 💼 Business Problem

A retail company's marketing team wants to launch **3–4 highly targeted advertising campaigns**, but they don't know who their distinct customer groups actually are. They have 29 raw features per customer — demographics, income, and spending across product categories — but no labels, no ground truth, and no predefined segments.

**The objective:** act as the Lead Data Scientist and algorithmically discover the natural customer segments hiding in this data, then translate those mathematical clusters into **actionable, human-readable personas** that a marketing team can build campaigns around.

This is a textbook unsupervised learning problem: there is no target variable to predict, only structure to uncover.

## 📊 Dataset

- **Source:** [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (Kaggle)
- **File:** `marketing_campaign.csv` (tab-separated)
- **Size:** ~2,240 customers, 29 raw features
- **Feature groups:**
  - **Demographics:** `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`
  - **Spending:** `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
  - **Purchasing behavior:** `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`

> 📁 Note: The raw CSV is not included in this repository per Kaggle's terms of use. Download it from the link above and place it in the project root as `marketing_campaign.csv`.

## 🔬 Methodology

### 1. Data Preprocessing

Unsupervised algorithms are unforgiving about data hygiene — there's no target variable to "average out" a messy feature. The pipeline:

1. Loads the raw tab-separated CSV.
2. Drops non-informative ID columns and rows with missing values.
3. Filters the dataframe down to purely numeric columns (PCA/K-Means cannot interpret categorical text like `"PhD"`).
4. Applies `StandardScaler` to bring every feature to zero mean / unit variance — preventing high-magnitude features (e.g., `Income` in the tens of thousands) from silently dominating low-magnitude ones (e.g., `NumDealsPurchases`, single digits).

### 2. Dimensionality Reduction (PCA)

With ~25 numeric features, clustering directly suffers from the **Curse of Dimensionality** — in high-dimensional space, distances between points become statistically meaningless.

- An unconstrained `PCA()` is fit first to inspect the full variance spectrum.
- A **Scree Plot** of cumulative explained variance is generated.
- The number of components required to retain **≥80% of total variance** is calculated *programmatically* (not eyeballed), and the data is projected onto that reduced space.

### 3. Clustering (K-Means)

- `KMeans` is swept across `n_clusters = 2` through `8`.
- Each run is scored with the **Silhouette Score**, which measures how well-separated clusters are — a more robust model-selection criterion than the Elbow Method, since it doesn't trivially improve as K increases.
- The K achieving the highest silhouette score is selected as optimal, and a final `KMeans` model is retrained on the full reduced dataset.

### 4. Advanced Techniques (GMM & UMAP)

Two industry-standard extensions push this beyond a baseline clustering exercise:

- **Gaussian Mixture Models (Soft Clustering):** Real customers rarely fall neatly into one bucket. A `GaussianMixture` model is fit to produce **probabilistic** cluster membership via `predict_proba()`, surfacing customers who sit right on the boundary between two personas — a valuable signal for targeted, careful campaign design.
- **UMAP (Presentation-Grade Visualization):** PCA is linear and optimized for variance, not visual separability. `umap.UMAP()` projects the data into 2D while preserving local neighborhood structure, producing a scatterplot where distinct customer segments appear as visually separated "islands" — ideal for a non-technical, executive-facing presentation.

## 🧑‍🤝‍🧑 Results: Customer Personas

The final model identified distinct, interpretable customer segments. Example persona summary (illustrative — regenerate with your own run to get exact figures):

| Cluster | Income | Wine Spend | Meat Spend | Deals Used | Persona |
|:-------:|:------:|:----------:|:----------:|:----------:|:--------|
| 0 | High | High | High | Low | **The Premium Big Spenders** |
| 1 | High | High | High | High | **The Affluent Bargain Hunters** |
| 2 | High | Low | Low | Low | **The Cautious High-Earners** |
| 3 | Low | Low | Low | High | **The Budget-Conscious Deal Hunters** |

**Marketing takeaway:** premium, low-discount campaigns should target Cluster 0; loyalty/rewards-driven promotions suit Cluster 1; Cluster 2 represents an under-tapped upsell opportunity; and Cluster 3 responds best to discount-led acquisition offers.

## 📁 Project Structure

```
customer-segmentation/
├── data/
│   └── marketing_campaign.csv      # (not tracked — download from Kaggle)
├── src/
│   ├── step1_preprocessing.py      # Data loading, cleaning, scaling
│   ├── step2_modeling.py           # PCA + K-Means + silhouette analysis
│   └── step3_interpretation.py     # Persona naming + GMM + UMAP
├── notebooks/
│   └── Unsupervised_Learning.ipynb # Original exploratory notebook
├── requirements.txt
└── README.md
```

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/customer-segmentation.git
cd customer-segmentation
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download `marketing_campaign.csv` from [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) and place it in the project root (or `data/`, adjusting the path in `step1_preprocessing.py`).

**5. Run the pipeline**
```bash
python src/step1_preprocessing.py
python src/step2_modeling.py
python src/step3_interpretation.py
```

Or explore interactively via the notebook:
```bash
jupyter notebook notebooks/Unsupervised_Learning.ipynb
```

## 🧠 Key Engineering Decisions

- **Optimal component/cluster counts are computed programmatically**, not hardcoded from a visual read of a plot — making the pipeline reproducible on refreshed data.
- **Persona labels are rule-based**, derived from each cluster's rank relative to its peers rather than fixed strings tied to arbitrary cluster indices, so relabeling doesn't break if K-Means cluster numbering shifts between runs.
- **`n_init` is pinned explicitly** on every `KMeans` call to guarantee reproducible results across scikit-learn versions.
- **The original unscaled dataframe is preserved** for business interpretation — cluster averages are reported in real units (dollars, purchase counts), not z-scores, because that's what a marketing stakeholder can actually act on.

## 🚀 Future Improvements

- Encode categorical features (`Education`, `Marital_Status`) via one-hot encoding to include them in the clustering feature space.
- Add automated cluster stability testing (e.g., bootstrapped silhouette scores) to validate robustness of the chosen K.
- Deploy an interactive Streamlit dashboard for marketing teams to explore personas without touching code.
- Package the pipeline as an installable module with `pytest` unit tests for CI/CD readiness.

---

<p align="center"><i>Built as part of an unsupervised learning portfolio project. Feedback and PRs welcome.</i></p>
