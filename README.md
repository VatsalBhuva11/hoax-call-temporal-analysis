# Network Evolution Analysis

## Assignment Overview

This project aims to build a **privacy-aware system** to analyze how hoax call networks evolve over time. Using **temporal network modeling**, **dynamic SNA metrics**, and **public datasets**, it helps detect structural changes, influencers, and suspicious trends. The system emphasizes **open-source tools** and **visual analytics** for **proactive threat detection**.

---

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone "https://github.com/VatsalBhuva11/hoax-call-temporal-analysis.git"
    cd hoax-call-temporal-analysis
    ```
2. Create a virtual environment (recommended via conda):

    ```bash
    python -m venv .
    ```

    If on Linux:

    ```bash
    source ./bin/activate
    ```

    For Windows:

    ```bash
    \bin\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Description of Directory Structure

```
.
├── data/                   # Input datasets (Enron, DBLP, etc.)
├── notebooks/              # Jupyter notebooks for various experiments
├── src/                    # Core Python scripts for modeling and analysis
│   ├── preprocessing/      # Data cleaning and transformation
│   ├── models/             # Growth models, prediction algorithms
│   └── visualization/      # Scripts for plotting and visual dashboards
├── results/                # Plots, metrics, and screenshots
├── README.md               # Project overview and instructions
└── requirements.txt        # Python dependencies
```

---

## Dependencies

-   Python 3.9+, Jupyter Notebook
-   NetworkX, DyNetX, graph-tool, igraph
-   ruptures, changepy, STUMPY, scikit-multiflow, pyts
-   NDlib, EoN, SimPy, matplotlib, seaborn, plotly, bokeh
    Use `requirements.txt` to install.

---

## Features Implemented

-   **Temporal Network Representation:** snapshot, interval graphs
-   **Change Detection:** global and local metrics, anomaly detection
-   **Growth Modeling:** BA, Watts-Strogatz, ER, SBM, communication-specific models
-   **Prediction Framework:** link prediction, role transitions, structural forecasts
-   **Visualization Dashboard:** animated evolution, community trends, metric timelines

---

## Instructions to Reproduce Results

1. Run `preprocessing/preprocess_data.ipynb` to prepare temporal data.
2. Execute `models/evolution_model.ipynb` to simulate network growth.
3. Use `visualization/dashboard.ipynb` to explore temporal graphs and metrics.
4. Evaluate predictions using `models/prediction_evaluation.ipynb`.

---

## Sample Outputs / Screenshots

-   Temporal graph evolution animation (in `/results/`)
-   Change detection plots: clustering coefficient over time
-   Community merging/splitting visualizations
-   Link prediction precision-recall plots

---

For more technical insights, refer to the full documentation in the `/notebooks/` directory.
