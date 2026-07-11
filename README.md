# 🌿 CropGuard AI — Precision Agriculture Intelligence System

**An AI-powered platform for crop disease detection and yield forecasting with real-time weather integration and intelligent diagnostics.**

## Features

### 🔬 Disease Classification (Tab 1)
- **CNN-based leaf disease detection** using a pre-trained deep learning model
- **Grad-CAM heatmaps** to visualize which leaf regions triggered the disease detection (Explainable AI)
- Supports **3 PlantVillage crops** — **Pepper, Potato, and Tomato** — across **15 disease classes**
- Real-time confidence scores with 88%+ accuracy
- AI-powered diagnosis via Groq LLM with treatment recommendations
- Weather risk assessment with disease-specific alerts
- Automatic crop detection → linked yield forecasting

### 📊 Yield Prediction (Tab 2)
- **Multi-input Random Forest model** (R² = 97.7%) considering:
  - Crop type & regional climate
  - Rainfall, temperature, pesticide inputs
  - Farm area & historical trends
- Supports **3 crops**: **Pepper, Potato, Tomato**
- 95% confidence intervals showing prediction uncertainty
- Disease-adjusted yield estimates (shows healthy vs disease-impacted yields)
- 10-year historical trend visualization with climate noise
- AI-powered economic insights from Groq LLM
- Actionable recommendations per crop/region

### 🗺️ Farm Zone Clustering
- **KMeans Clustering** to group farm zones based on sensor data.
- Evaluates field conditions to identify similar agricultural zones based on local traits.

### 🌦️ Weather Integration
- Real-time weather data via Open-Meteo API (no API key required)
- Disease-specific risk alerts (e.g., Late Blight favors cool, wet conditions)
- Weather-aware treatment recommendations
- Integration with disease detection for contextual advice

### 🤖 AI Diagnostics
- **Groq LLaMA 3.3 integration** for:
  - Detailed disease explanations
  - Immediate action steps for farmers
  - Climate-specific treatment strategies
  - Economic yield insights & optimization tips
- All responses tailored to detected conditions

## Supported Crops & Diseases

| Crop | Disease Classes |
|------|----------------|
| **Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

> **Note:** Full 14-crop support (Apple, Blueberry, Cherry, Corn, Grape, etc.) is planned for a future release after retraining on the complete PlantVillage dataset.

## Project Structure

```text
agri-ml-system-full/
├── app/
│   └── streamlit_app.py           # Main Streamlit application
├── src/                           # Source code
│   ├── data/                      # Data downloading and preprocessing scripts
│   ├── training/                  # Training scripts for different models
│   └── utils/                     # Helper functions
├── data/                          # Datasets
├── outputs/                       # Output logs, metrics, or graphs
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml               # Your Groq API key (DO NOT COMMIT)
└── models/
    ├── disease_model.h5           # CNN disease classifier (224×224 RGB)
    ├── disease_classes.json       # Disease class mappings (15 classes)
    ├── yield_model.pkl            # Random Forest yield predictor
    ├── yield_columns.pkl          # Feature names for yield model
    ├── yield_crop_encoder.pkl     # Crop name encoder
    ├── yield_country_encoder.pkl  # Country name encoder
    ├── crop_list.json             # Available crops (Pepper, Potato, Tomato)
    └── country_list.json          # Available countries
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/niharikaprasad1906/cropguard-ai.git
cd cropguard-ai
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data & Train Models (Optional)
If you wish to train the models from scratch:
```bash
python src/data/download_data.py
python src/training/train_disease.py
python src/training/train_yield.py
python src/training/train_cluster.py
```

### 5. Add your Groq API key
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
Get a free key at: https://console.groq.com

### 6. Run the app
```bash
streamlit run app/streamlit_app.py
```
App will open at `http://localhost:8501`

## Models & Data

### Disease Model
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Dataset:** PlantVillage — Pepper, Potato, Tomato subsets
- **Accuracy:** 88%+ on test set
- **Input:** 224×224 RGB leaf images
- **Output:** 15 disease classes across 3 crops

### Yield Model
- **Algorithm:** Random Forest (200 trees, max_depth=20)
- **Dataset:** Synthetic agronomic data (20,000 samples)
- **R² Score:** 97.7% | MAE: 1.42 t/ha
- **Features:** Crop, country, year, rainfall, temperature, pesticides, area
- **Target:** Crop yield (tonnes/hectare)
- **Training range:** 1990–2024

### Clustering Model
- **Algorithm:** KMeans
- **Application:** Grouping farm zones using sensor data attributes.

## API Keys Required

### Groq (Free tier)
- **Purpose:** AI diagnoses and economic insights
- **Get key:** https://console.groq.com
- **Cost:** Free (includes $25/month credits)
- **Model used:** llama-3.3-70b-versatile

### Open-Meteo
- **Purpose:** Real-time weather data
- **No API key required**
- **Service:** https://open-meteo.com

## Technology Stack

- **Frontend:** Streamlit
- **ML/AI:** TensorFlow (disease CNN), scikit-learn (yield RF, clustering), Groq LLM
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **Weather:** Open-Meteo API
- **Deployment:** Streamlit Cloud / self-hosted

## Limitations

1. **Disease detection limited to 3 crops** — Pepper, Potato, Tomato only
2. **Yield predictions available for 3 crops only**
3. **Synthetic yield data** — generated using agronomic parameters (not real farm data)
4. **Real-world variations** — soil type, irrigation, disease pressure not modeled

## Future Enhancements

- [ ] Expand to full 14-crop PlantVillage dataset
- [ ] Top-3 disease predictions (confidence ranking)
- [ ] Scan history & analytics dashboard
- [ ] Multi-language UI (Hindi, Gujarati for Indian farmers)
- [ ] Mobile PWA app
- [ ] Real yield dataset integration

## License

MIT License — see LICENSE file

---

**Built with ❤️ for farmers & agricultural data scientists**

*CropGuard AI — Making precision agriculture accessible.*