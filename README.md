
# 🌾 Agri ML System (Full Version)

End-to-end Machine Learning project for:
- Crop Disease Classification (CNN - MobileNetV2)
- Crop Yield Prediction (GBM + GridSearch)
- Farm Zone Clustering (KMeans + Silhouette Score)

## Setup
pip install -r requirements.txt

## Download Data
python src/data/download_data.py

## Train Models
python src/training/train_disease.py
python src/training/train_yield.py
python src/training/train_cluster.py

## Run App
streamlit run app/streamlit_app.py
