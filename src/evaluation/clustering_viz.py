"""
clustering_viz.py - Latent Space Visualisation & Clustering
CSE425 Neural Networks | Ummay Maimona Chaman | 22301719 | Section 1

This script performs PCA, K-Means, and DBSCAN on the latent spaces of the
Autoencoder and VAE models to visualize how genres are separated.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import *
from src.models.autoencoder import LSTMAutoencoder
from src.models.vae import MusicVAE

def plot_clusters(data, labels, genre_labels, method_name, model_name, save_path):
    """
    Plots 2D clusters using PCA.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=[GENRES[l] for l in genre_labels], 
                    palette='viridis', style=labels, markers=True, s=60, alpha=0.7)
    
    plt.title(f"{model_name} Latent Space Clustering ({method_name})\nExplained Var: {np.sum(pca.explained_variance_ratio_):.2f}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {save_path}")

def run_clustering_analysis(device='cpu'):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load test data
    try:
        data = np.load(os.path.join(TRAIN_TEST_DIR, 'ae_test.npy'))[:300]
        genres = np.load(os.path.join(TRAIN_TEST_DIR, 'genres_test.npy'))[:300]
    except FileNotFoundError:
        print("Error: Test data not found. Run preprocess_data.py first.")
        return

    # 1. Autoencoder Latent Space
    print("[Clustering] Analyzing Autoencoder latent space...")
    model_ae = LSTMAutoencoder().to(device)
    # Extract latents
    with torch.no_grad():
        x = torch.tensor(data).to(device)
        latents_ae = model_ae.encoder(x).cpu().numpy()
    
    # Agglomerative/K-Means Clustering
    kmeans = KMeans(n_clusters=NUM_GENRES, n_init=10, random_state=42)
    ae_clusters = kmeans.fit_predict(latents_ae)
    
    plot_clusters(latents_ae, ae_clusters, genres, "K-Means", "LSTM Autoencoder", 
                  os.path.join(PLOTS_DIR, "autoencoder_clusters_kmeans.png"))

    # 2. VAE Latent Space
    print("[Clustering] Analyzing VAE latent space...")
    model_vae = MusicVAE().to(device)
    with torch.no_grad():
        g_tensor = torch.tensor(genres).to(device)
        mu, _ = model_vae.encoder(x, g_tensor)
        latents_vae = mu.cpu().numpy()
        
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    vae_clusters = dbscan.fit_predict(latents_vae)
    
    plot_clusters(latents_vae, vae_clusters, genres, "DBSCAN", "Music VAE", 
                  os.path.join(PLOTS_DIR, "vae_clusters_dbscan.png"))
    
    # 3. PCA Baseline
    print("[Clustering] Analyzing Raw Data (PCA Baseline)...")
    raw_flat = data.reshape(len(data), -1)
    plot_clusters(raw_flat, genres, genres, "Direct Genres", "Raw Piano-Roll", 
                  os.path.join(PLOTS_DIR, "baseline_pca_genres.png"))

if __name__ == "__main__":
    run_clustering_analysis()
