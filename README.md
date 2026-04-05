# 👕 Hybrid Clothing Recommendation System

A machine learning system that recommends clothing items based on:

- 🖼 User image (visual similarity)
- 📊 Body measurements (tabular data)

The system combines computer vision and structured data using a hybrid recommendation approach

---

## 🚀 Features

- 🔥 Image-based similarity using deep learning (ResNet / CLIP)
- 📊 Body measurement-based clustering & similarity
- 🧠 Hybrid recommendation engine (image + tabular fusion)


---

## 🏗️ Tech Stack (Using)

- Python, NumPy, Pandas
- Scikit-learn (KMeans, preprocessing)
- TensorFlow / PyTorch (image embeddings)
- FastAPI (backend)
- Docker (containerization)
- Google Cloud Run + GCS (deployment)
- FAISS (optional for fast similarity search)

---

## 🎯 Use Case

This system can be used for:
- Fashion recommendation platforms
- Personalized styling assistants
- E-commerce product matching
- Virtual try-on systems (future extension)

---

## 📌 Project Status

🚧 In Development → Moving towards production deployment

---

## 🤝 Contributions

Open to improvements, optimizations, and feature extensions.

```
clothing-recommender
│
├── app/
│   ├── main.py              # FastAPI app
│   ├── model.py             # ML pipeline
│   ├── utils.py             # preprocessing
│   ├── config.py
│
├── models/
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── embeddings.npy
│
├── data/
│   ├── dataset.csv
│   ├── images/
│
├── notebooks/
│   ├── Collectivism.ipynb
│   ├── UserImageModel.ipynb

│
├── requirements.txt
├── Dockerfile
├── README.md

```
