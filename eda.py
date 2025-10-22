import os
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt


def save_fig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def make_figures(df):
    # 1) Raw vs log views
    sns.histplot(df["views"], bins=50)
    plt.title("Figure 1. Raw View Count Distribution")
    save_fig("figs/fig1_views_raw.png")

    sns.histplot(df["log_views"], bins=50)
    plt.title("Figure 1b. Log(Views) Distribution")
    save_fig("figs/fig1b_views_log.png")

    # 2) Duration vs log_views
    sns.scatterplot(data=df, x="duration_seconds", y="log_views", alpha=0.5)
    plt.title("Figure 2. Duration vs Log(Views)")
    save_fig("figs/fig2_duration_vs_logviews.png")

    # 3) Publish hour profile
    hour_median = df.groupby("publish_hour")["log_views"].median()
    hour_median.plot(kind="bar", rot=0, title="Figure 3. Median Log(Views) by Publish Hour")
    save_fig("figs/fig3_logviews_by_hour.png")

    # 4) Title length quartiles
    df["title_len_bin"] = pd.qcut(df["title_len"], q=4, labels=["Q1","Q2","Q3","Q4"])
    sns.boxplot(data=df, x="title_len_bin", y="log_views")
    plt.title("Figure 4. Log(Views) by Title Length Quartile")
    save_fig("figs/fig4_logviews_by_titlelen.png")
