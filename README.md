Certainly! Here's the README rewritten without emojis:

---

# Oil Price Change Point Analysis & Statistical Modeling

This project presents an in-depth analysis of Brent crude oil prices, focusing on the influence of major political and economic events. It integrates statistical modeling, machine learning, and interactive dashboards to uncover actionable insights for investors, policymakers, and energy sector stakeholders.

The repository includes everything from raw data preprocessing to building forecasting models (ARIMA, LSTM), change point detection, and deployment of a web-based dashboard using Flask and React.


---

## Key Features

* Time Series Analysis – In-depth exploration of oil price trends over time.
* Change Point Detection – Identify significant economic or political events affecting price shifts.
* Forecasting Models – ARIMA and LSTM used to predict future oil prices.
* Indicator Correlation – Analyze the impact of macroeconomic indicators on oil prices.
* Machine Learning Integration – Use of LSTM and other models for predictive analysis.
* Interactive Dashboard – Built with Flask (backend) and React (frontend) for real-time visualization.
* Insightful Reporting – Findings summarized in a comprehensive report.

---

## Project Structure

```plaintext
.
├── .github/workflows/         # CI/CD pipelines using GitHub Actions
├── .vscode/                   # VS Code editor settings
├── notebooks/                 # Jupyter Notebooks for EDA and prototyping
├── dashboard/                 # Flask + React app for interactive visualization
├── scripts/                   # Python scripts for data cleaning, modeling, and plotting
├── tests/                     # Unit tests for data integrity and model accuracy
└── README.md                  # Project overview and setup guide
```

---

## Installation Guide

Follow the steps below to set up the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/doffn/Week-10-Change-point-analysis-and-statistical-modelling-of-time-series-data-.git
cd Week-10-Change-point-analysis-and-statistical-modelling-of-time-series-data-
```

### 2. Create and Activate a Virtual Environment

#### On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Live Demo & Report

The full report contains visual interpretations, metric breakdowns, and key takeaways. The dashboard can be launched locally and provides dynamic analysis based on real data.
---