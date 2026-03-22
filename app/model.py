import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from .utils import get_embedding_from_bytes, preprocess_tabular

# Load saved models
scaler = joblib.load("models/scaler.pkl")

# Load embeddings
embeddings = np.load("models/embeddings.npy")

# Load dataset
dataset = pd.read_csv("data/dataset.csv")


def compute_image_similarity(user_embedding, dataset_embeddings):
    sims = cosine_similarity([user_embedding], dataset_embeddings)
    return sims.flatten()


def compute_tabular_similarity(user_metrics):

    # Example: dataset should contain same columns
    tabular_data = dataset.drop(columns=["image_name"])

    user_scaled = preprocess_tabular(user_metrics, scaler)

    sims = cosine_similarity(user_scaled, tabular_data)
    return sims.flatten()

def hybrid_recommend(image_bytes, user_metrics):

    # 1. Get user embedding
    user_embedding = get_embedding_from_bytes(image_bytes)

    # 2. Compute similarities
    img_sim = compute_image_similarity(user_embedding, embeddings)
    tab_sim = compute_tabular_similarity(user_metrics)

    # 3. Combine
    final_score = 0.7 * img_sim + 0.3 * tab_sim

    # 4. Get top results
    top_idx = np.argsort(final_score)[-5:][::-1]

    # 5. Return image names
    return dataset.iloc[top_idx]["image_name"].tolist()


