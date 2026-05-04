# Copyright (c) 2026 Sinem Turkoglu - MIT License
# # ==========================================
# Project: Moltbook Analysis Pipeline (Final Publication Standard)
# Author: Sinem Türkoğlu
# Description: Unified analysis pipeline with English visualizations for academic publication.
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prophet import Prophet
import logging
import os
from clean_pipeline import clean_and_integrate_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
CLEANED_DATA_PATH = 'moltbook_final_v4.csv'
RAW_DATA_PATH = 'moltbook_temiz.csv'
FINAL_DATA_PATH = 'moltbook_temiz_final.csv'
PLOTS_DIR = 'plots'
DPI = 300

# Topic Mapping (English for Publication)
TOPIC_MAPPING = {
    'A': 'A (General Discussion)', 'B': 'B (Politics)', 'C': 'C (Economy)',
    'D': 'D (Technology)', 'E': 'E (Sports)', 'F': 'F (Culture & Arts)',
    'G': 'G (Health & Lifestyle)', 'H': 'H (Education & Science)', 'I': 'I (Other)'
}

def setup_environment():
    """Ensures directories and styles are set."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    plt.style.use('fivethirtyeight')
    sns.set_theme(style="whitegrid")

def run_cleaning_pipeline():
    """Runs the H5 cleaning and integration pipeline."""
    logging.info("--- Stage 1: Data Cleaning & Integration ---")
    clean_and_integrate_pipeline(CLEANED_DATA_PATH, RAW_DATA_PATH, FINAL_DATA_PATH)

def perform_general_analysis(df):
    """
    Performs basic descriptive analysis and saves plots in English.
    
    Args:
        df (pd.DataFrame): Cleaned and integrated dataframe.
    """
    logging.info("--- Stage 2: General Analysis ---")
    
    df['topic_display'] = df['topic_label'].map(TOPIC_MAPPING).fillna(df['topic_label'])
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    # 1. Daily Activity
    plt.figure(figsize=(12, 6))
    daily_counts = df.groupby('date').size().reset_index(name='post_count')
    sns.lineplot(data=daily_counts, x='date', y='post_count', marker='o', color='dodgerblue')
    plt.title('Daily Post Activity', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Post Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/01_daily_activity.png', dpi=DPI)
    plt.close()

    # 2. Topic Distribution
    plt.figure(figsize=(12, 6))
    topic_counts = df['topic_display'].value_counts().reset_index()
    topic_counts.columns = ['topic_display', 'count']
    sns.barplot(data=topic_counts, x='topic_display', y='count', hue='topic_display', palette='viridis', legend=False)
    plt.title('Topic Distribution', fontsize=16)
    plt.xlabel('Topic')
    plt.ylabel('Post Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/02_topic_distribution.png', dpi=DPI)
    plt.close()

    # 3. Toxicity Distribution
    plt.figure(figsize=(8, 5))
    toxic_counts = df['toxic_level'].value_counts().sort_index().reset_index()
    toxic_counts.columns = ['toxic_level', 'count']
    sns.barplot(data=toxic_counts, x='toxic_level', y='count', hue='toxic_level', palette='Reds', legend=False)
    plt.title('Toxicity Level Distribution', fontsize=16)
    plt.xlabel('Toxicity Level')
    plt.ylabel('Post Count')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/03_toxicity_distribution.png', dpi=DPI)
    plt.close()

def perform_clustering(df, n_clusters=4):
    """
    Performs feature engineering and KMeans clustering.
    
    Args:
        df (pd.DataFrame): Dataframe.
        n_clusters (int): Number of clusters.
    """
    logging.info("--- Stage 3: Clustering ---")
    
    # Feature Engineering
    df['text_len'] = df['content_body'].fillna('').apply(len)
    df['word_count'] = df['content_body'].fillna('').apply(lambda x: len(x.split()))
    df['special_char_count'] = df['content_body'].fillna('').apply(lambda x: sum(not c.isalnum() for c in x))
    df['is_code'] = df['content_body'].fillna('').apply(lambda x: 1 if '```' in x else 0)
    
    # Topic Encoding
    df['topic_encoded'] = pd.factorize(df['topic_label'])[0]
    
    features = ['toxic_level', 'upvotes_count', 'topic_encoded', 'text_len', 'word_count', 'special_char_count', 'is_code']
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualization with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['cluster'], palette='Set1', s=60, alpha=0.6)
    plt.title('Post Clusters (PCA Visualization)', fontsize=16)
    plt.legend(title='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/04_clustering_pca.png', dpi=DPI)
    plt.close()
    
    # Feature distribution by cluster
    cluster_means = df.groupby('cluster')[features].mean()
    logging.info(f"Cluster Means:\n{cluster_means}")
    
    return df

def perform_forecasting(df):
    """
    Performs 30-day activity and toxicity forecasting using Prophet.
    
    Args:
        df (pd.DataFrame): Dataframe with 'timestamp' and 'toxic_level'.
    """
    logging.info("--- Stage 4: Time-Series Forecasting ---")
    
    # 1. Activity Forecast
    df_prophet = df.set_index(pd.to_datetime(df['timestamp'])).resample('D').size().reset_index(name='y')
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    m_activity = Prophet(weekly_seasonality=True, daily_seasonality=False)
    m_activity.fit(df_prophet)
    
    future_activity = m_activity.make_future_dataframe(periods=30)
    forecast_activity = m_activity.predict(future_activity)
    
    fig1 = m_activity.plot(forecast_activity)
    plt.title('30-Day Activity Forecast', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Post Count')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/05_activity_forecast.png', dpi=DPI)
    plt.close()

    # 2. Toxicity Forecast
    df_toxic = df.set_index(pd.to_datetime(df['timestamp'])).resample('D')['toxic_level'].mean().reset_index()
    df_toxic.columns = ['ds', 'y']
    df_toxic['ds'] = df_toxic['ds'].dt.tz_localize(None)
    
    m_toxic = Prophet(weekly_seasonality=True, daily_seasonality=False)
    m_toxic.fit(df_toxic)
    
    future_toxic = m_toxic.make_future_dataframe(periods=30)
    forecast_toxic = m_toxic.predict(future_toxic)
    
    fig2 = m_toxic.plot(forecast_toxic)
    plt.title('30-Day Toxicity Trend Forecast', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Average Toxicity Level')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/06_toxicity_forecast.png', dpi=DPI)
    plt.close()

def main():
    """Main execution flow."""
    setup_environment()
    
    # 1. Cleaning
    run_cleaning_pipeline()
    
    # Load the fresh data
    if not os.path.exists(FINAL_DATA_PATH):
        logging.error("Final dataset could not be created!")
        return
        
    df = pd.read_csv(FINAL_DATA_PATH)
    
    # 2. General Analysis
    perform_general_analysis(df)
    
    # 3. Clustering
    df = perform_clustering(df)
    
    # 4. Forecasting
    perform_forecasting(df)
    
    logging.info("--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")
    logging.info(f"Plots saved to '{PLOTS_DIR}/' directory with 300 DPI resolution.")

if __name__ == "__main__":
    main()

