# YouTube Engagement Analysis With SHAP

[Portfolio and case studies](https://alirazaportfolio2026.netlify.app/)

Independent academic portfolio project by Ali Raza, using public YouTube metadata. This work is not commissioned by or affiliated with MrBeast. Model associations and SHAP explanations do not establish causal effects on engagement.

🎥 YouTube Video Engagement Analysis — MrBeast Case Study

Author: Ali Raza
Course: BAN 612 — Business Analytics
Dataset: 250 public uploads from @MrBeast
 using YouTube Data API v3

🧠 Overview

This project analyzes YouTube engagement dynamics using MrBeast’s channel as a high-volume case study.
MrBeast (Jimmy Donaldson) is one of YouTube’s most influential creators with 200M+ subscribers. His content combines extreme challenges, philanthropy, and storytelling, making it an ideal dataset for studying video reach and audience behavior.

The study uses machine learning models (Linear, Logistic, Random Forest, Gradient Boosting) to understand how factors like video duration, upload time, and title features affect engagement.

🎯 Objectives

Collect and clean metadata from YouTube’s API.

Engineer features (publish time, duration, title length, etc.) for predictive analysis.

Compare baseline and advanced models on engagement prediction.

Interpret model behavior using SHAP explainability to derive actionable recommendations for creators.

🧩 Data Source & Pipeline

Source: YouTube Data API v3
Process:

Pulled data through channels → uploads playlist → videos.list endpoints (≤50 IDs per call).

Extracted ISO-8601 durations, title lengths, publish hour/day, and log-transformed view counts.

Saved cleaned dataset as data/youtube_dataset.csv.

Code Files:

fetch_youtube.py → Data pull via API

features.py → Feature engineering pipeline

eda.py → Exploratory data analysis

models.py → Baseline and advanced model training

shap_explain.py → Explainability visualizations

📊 Exploratory Data Analysis

Engagement (views) shows heavy-tailed distribution; log-transform stabilizes variance.

Duration and title length show weak but non-linear patterns.

Uploads between 18–22h local time perform better in early engagement velocity.

Shorts (≤60s) behave differently, requiring a binary flag (is_short).

🤖 Modeling Summary
Model Type	Task	Metric	Score
Linear Regression	Regression (log_views)	R²	−0.013
Logistic Regression	Classification	ROC-AUC	0.634
Random Forest	Regression (log_views)	R²	0.0135
Random Forest	Classification	ROC-AUC	0.6530
Gradient Boosting	Regression (log_views)	R²	−0.0856
Gradient Boosting	Classification	ROC-AUC	0.6646

Tree-based models outperform linear baselines and provide interpretability using SHAP values.

🔍 Model Interpretability (SHAP)

Global Insights:

duration_seconds, is_short, and publish_hour are top predictors of engagement.

Local Example:

Videos posted between 18–22h or those flagged as Shorts tend to have higher engagement probability.

Overly long uploads or off-peak timings reduce performance.

(See Figures 5–6 in report for SHAP summary and waterfall plots.)

📈 Key Insights & Recommendations

Upload Timing: Schedule between 18–22h for best reach.

Format Strategy: Treat Shorts and long-form content separately.

Title Optimization: Keep titles concise and descriptive.

Data-Driven Planning: Use interpretable models to inform content decisions.

⚙️ Future Work

Integrate sentiment and keyword analysis using lightweight NLP.

Extend dataset to include multiple creators.

Evaluate XGBoost, LightGBM, or neural network architectures.

Deploy an interactive YouTube Engagement Dashboard with timing and title recommendations.

🛠️ Requirements

Python 3.9+
Libraries:
pandas, numpy, matplotlib, seaborn, scikit-learn, requests, google-api-python-client, shap

🧮 Example Usage
python fetch_youtube.py --channel_id UCX6OQ3DkcsbYNE6H8uQQuVA
python features.py --input data/raw.json --output data/youtube_dataset.csv
python eda.py
python models.py
python shap_explain.py

🏁 Conclusion

This project demonstrates an end-to-end data science workflow—from data acquisition and feature engineering to modeling and explainability.
The findings contribute to a practical framework for content creators seeking data-informed upload strategies, focusing on transparency, timing, and performance optimization.
