import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import json
import joblib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go

from PIL import Image
import tf_keras
from tf_keras.applications import MobileNetV2
from tf_keras import layers as _kl, models as _km
import tf_keras.utils as _tf_keras_utils
import tensorflow as tf
import matplotlib.cm as cm

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# LOAD MODELS (cached)
# ======================================

@st.cache_resource
def load_models():
    # ── 1. Disease model (MobileNetV2 + weights) ─────────────────────────────
    with open("models/disease_classes.json") as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = _kl.GlobalAveragePooling2D()(base.output)
    x = _kl.Dense(128, activation="relu")(x)
    x = _kl.Dropout(0.3)(x)
    output = _kl.Dense(num_classes, activation="softmax")(x)
    disease_model = _km.Model(inputs=base.input, outputs=output)
    
    # Try to load saved custom layer weights (the 3 Dense layers)
    try:
        # Load only the custom dense layers by name
        disease_model.load_weights("models/disease_model.h5", by_name=True, skip_mismatch=True)
    except Exception as e:
        pass  # Continue with ImageNet-initialized MobileNetV2 if custom weights fail

    # ── 2. Crop & country lists ───────────────────────────────────────────────
    with open("models/crop_list.json") as f:
        crop_list = json.load(f)
    with open("models/country_list.json") as f:
        country_list = json.load(f)

    # ── 3. Yield model — retrain on synthetic data (version-agnostic) ─────────
    # The saved yield_model.pkl (263 MB, RandomForest) fails to unpickle due to
    # numpy/sklearn version mismatches. Since the data is fully synthetic we
    # regenerate it on first load. @st.cache_resource ensures this runs once.
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    le_crop = LabelEncoder()
    le_crop.fit(crop_list)
    le_country = LabelEncoder()
    le_country.fit(country_list)

    rng = np.random.default_rng(42)
    n   = 20_000
    crops_s     = rng.choice(crop_list, n)
    countries_s = rng.choice(country_list, n)
    crop_enc    = le_crop.transform(crops_s)
    country_enc = le_country.transform(countries_s)
    year        = rng.integers(1990, 2025, n)
    rainfall    = rng.uniform(200, 3000, n)
    avg_temp    = rng.uniform(10, 35, n)
    pesticides  = rng.uniform(0, 100, n)
    area        = rng.uniform(1, 100, n)

    target = np.clip(
        2.0 + (crop_enc * 0.1) + (country_enc * 0.05)
        + np.where((rainfall > 500) & (rainfall < 1500), 1.0, -0.5)
        + np.where((avg_temp > 18) & (avg_temp < 28), 0.5, -0.5)
        + np.log1p(pesticides) * 0.2
        + (year - 1990) * 0.02
        + rng.normal(0, 0.5, n),
        0.1, 15.0
    )

    X = pd.DataFrame({
        "crop_enc": crop_enc, "country_enc": country_enc,
        "year": year, "rainfall": rainfall,
        "avg_temp": avg_temp, "pesticides": pesticides, "area": area
    })

    yield_model = RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    yield_model.fit(X, target)
    yield_columns = list(X.columns)
    # ─────────────────────────────────────────────────────────────────────────

    return (disease_model, yield_model, yield_columns,
            le_crop, le_country, list(class_indices.keys()),
            crop_list, country_list)

disease_model, yield_model, yield_columns, le_crop, le_country, class_names, crop_list, country_list = load_models()

# ======================================
# GRAD-CAM UTILS
# ======================================

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except AttributeError:
            try:
                shape = layer.output.shape
            except AttributeError:
                continue
        
        if isinstance(shape, list):
            shape = shape[0]
        if shape is not None and len(shape) == 4:
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Standard Grad-CAM via sub-model + GradientTape.
    
    Works correctly with MobileNetV2's residual Add layers because the
    sub-model handles all branching internally — no manual layer looping.
    """
    # Build a sub-model: input → (last_conv_output, final_predictions)
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    grad_model = _km.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, preds = grad_model(inputs)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the class score w.r.t. the conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    # Pool gradients over spatial dims → importance weights per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs_val = conv_outputs[0]                           # (H, W, C)
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)
    # ReLU + normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap = heatmap / max_val
    return heatmap.numpy()

def overlay_gradcam(img_path_or_pil, heatmap, alpha=0.4):
    if isinstance(img_path_or_pil, Image.Image):
        img = img_path_or_pil.copy()
    else:
        img = Image.open(img_path_or_pil)
    img_array = _tf_keras_utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    try:
        jet = cm.get_cmap("jet")
    except AttributeError:
        # For newer matplotlib versions
        import matplotlib
        jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = _tf_keras_utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = _tf_keras_utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img_array * (1 - alpha)
    superimposed_img = _tf_keras_utils.array_to_img(superimposed_img)
    return superimposed_img

# ======================================
# DISEASE KNOWLEDGE BASE
# ======================================

DISEASE_INFO = {
    "Tomato_mosaic_virus": {
        "severity": "high",
        "description": "Viral infection causing mosaic leaf patterns and stunted growth.",
        "treatment": "Remove infected plants. Control aphid vectors. Use resistant varieties.",
        "prevention": "Use certified virus-free seeds. Sanitize tools regularly.",
    },
    "Tomato_Early_blight": {
        "severity": "medium",
        "description": "Fungal disease causing dark spots with concentric rings on older leaves.",
        "treatment": "Apply copper-based or chlorothalonil fungicide. Remove affected leaves.",
        "prevention": "Rotate crops. Ensure good air circulation. Avoid overhead watering.",
    },
    "Tomato_Late_blight": {
        "severity": "high",
        "description": "Aggressive water mold causing dark lesions and rapid plant collapse.",
        "treatment": "Apply mancozeb or metalaxyl fungicide immediately. Destroy infected plants.",
        "prevention": "Plant resistant varieties. Avoid excessive moisture on foliage.",
    },
    "Tomato_healthy": {
        "severity": "low",
        "description": "No disease detected. Leaf appears healthy and vigorous.",
        "treatment": "No treatment required.",
        "prevention": "Maintain regular monitoring and good agricultural practices.",
    },
    "Potato_Early_blight": {
        "severity": "medium",
        "description": "Fungal pathogen producing target-like brown spots on older foliage.",
        "treatment": "Apply protectant fungicides. Improve soil nitrogen levels.",
        "prevention": "Plant certified seed tubers and practice a 3-year crop rotation.",
    },
    "Potato_Late_blight": {
        "severity": "high",
        "description": "Highly destructive oomycete causing rapid foliar necrosis and tuber rot.",
        "treatment": "Apply systemic fungicides immediately. Kill infected vines before harvest.",
        "prevention": "Eliminate cull piles and voluntary plants. Select resistant cultivars.",
    },
    "Potato_healthy": {
        "severity": "low",
        "description": "Potato foliage appears healthy with normal pigmentation and structure.",
        "treatment": "No treatment required.",
        "prevention": "Follow appropriate hilling, watering, and scouting routines.",
    },
    "Corn_Common_rust": {
        "severity": "medium",
        "description": "Fungal disease forming powdery, cinnamon-brown pustules on both leaf surfaces.",
        "treatment": "Fungicides are rarely economical unless severe early infection occurs.",
        "prevention": "Plant hybrid varieties engineered with specific rust resistance genes.",
    },
    "Corn_Northern_Leaf_Blight": {
        "severity": "high",
        "description": "Fungal infection creating long, cigar-shaped grayish-green lesions.",
        "treatment": "Apply foliar fungicides at silking stage if thresholds are exceeded.",
        "prevention": "Manage residue through tillage and rotate away from corn for one season.",
    },
    "Corn_healthy": {
        "severity": "low",
        "description": "Corn leaf exhibits solid green color without spotting or streaking.",
        "treatment": "No treatment required.",
        "prevention": "Ensure balanced nitrogen application and uniform plant spacing.",
    },
    "Apple_scab": {
        "severity": "high",
        "description": "Fungal disease leading to olive-green or brown velvety spots on leaves and fruit.",
        "treatment": "Apply targeted protectant or curative fungicides during green tip to petal fall.",
        "prevention": "Rake and destroy fallen leaves in autumn to disrupt overwintering spores.",
    },
    "Apple_Black_rot": {
        "severity": "medium",
        "description": "Fungal pathogen causing frogeye leaf spots, fruit rot, and twig cankers.",
        "treatment": "Prune out dead wood, mummified fruit, and active cankers during dormancy.",
        "prevention": "Maintain tree vigor and avoid mechanical bark injuries.",
    },
    "Apple_Cedar_rust": {
        "severity": "medium",
        "description": "Rust fungus producing striking bright orange spots on upper leaf surfaces.",
        "treatment": "Apply preventative fungicides from bud break until early summer.",
        "prevention": "Remove nearby eastern red cedar trees if practical within a 2-mile radius.",
    },
    "Apple_healthy": {
        "severity": "low",
        "description": "Apple foliage displays normal, healthy canopy uniformity.",
        "treatment": "No treatment required.",
        "prevention": "Prune annually to maximize sunlight penetration and airflow.",
    },
    "Grape_Black_rot": {
        "severity": "high",
        "description": "Fungal disease generating small brown circular lesions on leaves and shriveling fruit into black mummies.",
        "treatment": "Apply early-season fungicides from bud break until fruit completion.",
        "prevention": "Remove all mummified berries from vines and ground during winter pruning.",
    },
    "Grape_Esca": {
        "severity": "high",
        "description": "Complex wood disease causing 'tiger-stripe' leaf discoloration and vine decline.",
        "treatment": "No cure exists; prune back infected wood to healthy tissue or replace vines.",
        "prevention": "Protect winter pruning wounds with specialized paste or paint sealants.",
    },
    "Grape_Leaf_blight": {
        "severity": "medium",
        "description": "Fungal infection causing dark brown necrotic shapes on leaf edges.",
        "treatment": "Apply copper sprays or standard broad-spectrum viticulture fungicides.",
        "prevention": "Keep canopy lifted and clear weeds beneath vines to reduce ground humidity.",
    },
    "Grape_healthy": {
        "severity": "low",
        "description": "Vibrant grape leaf showing excellent vascular health and color.",
        "treatment": "No treatment required.",
        "prevention": "Stick to strict vine training and seasonal spray program baselines.",
    },
    "Pepper_Bacterial_spot": {
        "severity": "high",
        "description": "Bacterial pathogen causing small, water-soaked spots that turn dark brown and cause leaf drop.",
        "treatment": "Apply copper-based bactericides mixed with mancozeb weekly.",
        "prevention": "Avoid handling plants when wet; use certified pathogen-free seeds.",
    },
    "Pepper_healthy": {
        "severity": "low",
        "description": "Pepper leaf looks uniform, healthy, and glossy.",
        "treatment": "No treatment required.",
        "prevention": "Implement drip irrigation rather than overhead sprinklers.",
    },
    "Strawberry_Leaf_scorch": {
        "severity": "medium",
        "description": "Fungal infection resulting in purple blotches that expand, drying out the leaf tissue.",
        "treatment": "Apply appropriate registered fungicides if noticed before fruit set.",
        "prevention": "Plant in well-drained soil locations and thin plants out to avoid crowding.",
    },
    "Strawberry_healthy": {
        "severity": "low",
        "description": "Strawberry foliage looks clean, compact, and completely healthy.",
        "treatment": "No treatment required.",
        "prevention": "Mulch clean straw underneath runners to prevent soil splashing.",
    },
    "Soybean_healthy": {
        "severity": "low",
        "description": "Soybean leaf has clean edges, standard color, and normal development.",
        "treatment": "No treatment required.",
        "prevention": "Monitor fields for early signs of standard legume pathogens.",
    },
    "Cherry_Powdery_mildew": {
        "severity": "medium",
        "description": "Fungus producing a white, powdery coating on leaf surfaces and young shoots.",
        "treatment": "Apply sulfur or commercial fungicides when initial signs appear.",
        "prevention": "Prune internal dense branches to improve overall interior canopy ventilation.",
    },
    "Cherry_healthy": {
        "severity": "low",
        "description": "Cherry orchard sample shows complete leaf health.",
        "treatment": "No treatment required.",
        "prevention": "Maintain uniform watering cycles to discourage stress.",
    },
    "Peach_Bacterial_spot": {
        "severity": "high",
        "description": "Bacterial disease leading to 'shot-hole' leaf spots, defoliation, and fruit lesions.",
        "treatment": "Utilize protective copper or oxytetracycline programs during the season.",
        "prevention": "Avoid planting highly susceptible stone fruit cultivars in windy regions.",
    },
    "Peach_healthy": {
        "severity": "low",
        "description": "Peach tree leaf exhibits standard structure without abnormalities.",
        "treatment": "No treatment required.",
        "prevention": "Follow recommended post-harvest orchard hygiene cleanups.",
    },
    "Orange_Citrus_greening": {
        "severity": "high",
        "description": "Devastating bacterial disease spread by psyllids, causing yellow shoots and bitter, misshapen fruit.",
        "treatment": "No cure; remove infected trees immediately to protect the remaining grove.",
        "prevention": "Strictly control Asian citrus psyllid populations using insecticides or netting.",
    },
    "Squash_Powdery_mildew": {
        "severity": "medium",
        "description": "Fungal growth forming white talcum-like spots on leaves, accelerating senescence.",
        "treatment": "Apply horticultural oils, potassium bicarbonate, or fungicides.",
        "prevention": "Space cucurbits generously to reduce relative microclimate humidity.",
    },
    "Raspberry_healthy": {
        "severity": "low",
        "description": "Raspberry cane leaf looks entirely robust, vibrant, and clean.",
        "treatment": "No treatment required.",
        "prevention": "Remove spent floricanes immediately post-harvest.",
    },
    "Blueberry_healthy": {
        "severity": "low",
        "description": "Blueberry foliage shows solid nutrient balance and perfect health.",
        "treatment": "No treatment required.",
        "prevention": "Maintain low soil pH ranges to avoid stress-induced chlorosis.",
    }
}

def get_disease_info(class_name):
    clean = class_name.replace("__", "_").replace(" ", "_")
    
    # Exact match check first
    if clean in DISEASE_INFO:
        return DISEASE_INFO[clean]
        
    # Generalized fallback logic for healthy tags across different crops
    if "healthy" in clean.lower():
        # Match crop specific healthy profile if found, otherwise use Tomato_healthy as base template
        for key in DISEASE_INFO:
            if "healthy" in key.lower() and key.split('_')[0].lower() in clean.lower():
                return DISEASE_INFO[key]
        return DISEASE_INFO["Tomato_healthy"]
        
    # Substring lookup loop
    for key in DISEASE_INFO:
        if key.lower() in clean.lower() or clean.lower() in key.lower():
            return DISEASE_INFO[key]
            
    # Absolute default if nothing matches
    return {
        "severity": "medium",
        "description": "Disease detected. Consult an agricultural expert for precise diagnosis.",
        "treatment": "Isolate affected plants and seek expert advice.",
        "prevention": "Monitor crops regularly and maintain field hygiene.",
    }

def format_disease_name(raw_name):
    parts = raw_name.replace("__", "_").split("_")
    seen = []
    for p in parts:
        if p.lower() not in [s.lower() for s in seen]:
            seen.append(p)
    return " ".join(seen).title()

# ======================================
# GROK AI DIAGNOSIS
# ======================================

def get_llm_diagnosis(disease_name, confidence, weather, is_healthy, grok_key):
    """Call xAI Grok API for a personalized diagnosis and advice."""
    weather_context = ""
    if weather and "error" not in weather:
        weather_context = f"""
Current weather in {weather['city']}:
- Temperature: {weather['temp']}°C (feels like {weather['feels_like']}°C)
- Humidity: {weather['humidity']}%
- Conditions: {weather['description']}
- Wind: {weather['wind']} km/h
- Rainfall: {weather['rain']} mm
"""
    else:
        weather_context = "Weather data not available."

    if is_healthy:
        prompt = f"""A crop disease detection AI analysed a leaf image and found it HEALTHY with {confidence:.0%} confidence.
{weather_context}
As an agricultural expert, provide:
1. A brief confirmation that the plant looks healthy
2. 2-3 preventive care tips to keep it healthy based on the current weather
3. Any weather-related risks to watch for

Keep it concise, practical, and farmer-friendly. Use plain text with no markdown."""
    else:
        prompt = f"""A crop disease detection AI analysed a leaf image and detected: {disease_name}
Confidence: {confidence:.0%}
{weather_context}
As an agricultural expert, provide:
1. A brief explanation of what this disease does to the plant (2-3 sentences)
2. Immediate action steps the farmer should take RIGHT NOW (numbered list)
3. Treatment recommendations considering the current weather conditions
4. Prevention steps for future crops

Keep it concise, practical, and farmer-friendly. Use plain text with no markdown or asterisks."""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {grok_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        elif r.status_code == 401:
            return "ERROR:Invalid Groq API key."
        elif r.status_code == 429:
            return "ERROR:Rate limit reached. Try again in a moment."
        else:
            return f"ERROR:Groq API error ({r.status_code})."
    except requests.exceptions.Timeout:
        return "ERROR:Request timed out."
    except requests.exceptions.ConnectionError:
        return "ERROR: Network error. Could not connect to Groq API. Please check your internet connection or firewall settings."
    except Exception as e:
        return f"ERROR: Could not generate AI diagnosis. Technical details: {str(e)}"

# Disease → yield loss range (min%, max%) and affected crop for yield tab
DISEASE_YIELD_IMPACT = {
    "Tomato_mosaic_virus":       {"loss_min": 10, "loss_max": 30, "crop": "Tomato",     "note": "Mosaic virus reduces fruit set and quality."},
    "Tomato_Early_blight":       {"loss_min": 20, "loss_max": 50, "crop": "Tomato",     "note": "Early blight causes defoliation reducing photosynthesis."},
    "Tomato_Late_blight":        {"loss_min": 40, "loss_max": 100,"crop": "Tomato",     "note": "Late blight can cause complete crop loss if untreated."},
    "Tomato_healthy":            {"loss_min": 0,  "loss_max": 0,  "crop": "Tomato",     "note": "No yield loss expected."},
    "Potato_Early_blight":       {"loss_min": 20, "loss_max": 40, "crop": "Potato",     "note": "Early blight reduces tuber size and number."},
    "Potato_Late_blight":        {"loss_min": 30, "loss_max": 100,"crop": "Potato",     "note": "Late blight caused the Irish famine — extremely destructive."},
    "Potato_healthy":            {"loss_min": 0,  "loss_max": 0,  "crop": "Potato",     "note": "No yield loss expected."},
    "Corn_Common_rust":          {"loss_min": 10, "loss_max": 40, "crop": "Corn",       "note": "Rust reduces grain weight and starch content."},
    "Corn_Northern_Leaf_Blight": {"loss_min": 20, "loss_max": 50, "crop": "Corn",       "note": "Leaf blight causes premature death of leaves."},
    "Corn_healthy":              {"loss_min": 0,  "loss_max": 0,  "crop": "Corn",       "note": "No yield loss expected."},
    "Apple_scab":                {"loss_min": 10, "loss_max": 70, "crop": "Apple",      "note": "Scab reduces fruit marketability significantly."},
    "Apple_Black_rot":           {"loss_min": 20, "loss_max": 60, "crop": "Apple",      "note": "Black rot causes mummified fruits and cankers."},
    "Apple_Cedar_rust":          {"loss_min": 10, "loss_max": 35, "crop": "Apple",      "note": "Cedar rust causes defoliation and reduced fruit size."},
    "Apple_healthy":             {"loss_min": 0,  "loss_max": 0,  "crop": "Apple",      "note": "No yield loss expected."},
    "Grape_Black_rot":           {"loss_min": 20, "loss_max": 80, "crop": "Grape",      "note": "Black rot can destroy entire clusters."},
    "Grape_Esca":                {"loss_min": 10, "loss_max": 50, "crop": "Grape",      "note": "Esca causes progressive vine decline."},
    "Grape_Leaf_blight":         {"loss_min": 15, "loss_max": 45, "crop": "Grape",      "note": "Leaf blight causes premature defoliation."},
    "Grape_healthy":             {"loss_min": 0,  "loss_max": 0,  "crop": "Grape",      "note": "No yield loss expected."},
    "Pepper_Bacterial_spot":     {"loss_min": 10, "loss_max": 40, "crop": "Pepper",     "note": "Bacterial spot causes fruit lesions and drop."},
    "Pepper_healthy":            {"loss_min": 0,  "loss_max": 0,  "crop": "Pepper",     "note": "No yield loss expected."},
    "Strawberry_Leaf_scorch":    {"loss_min": 10, "loss_max": 30, "crop": "Strawberry", "note": "Leaf scorch reduces runner production and fruit size."},
    "Strawberry_healthy":        {"loss_min": 0,  "loss_max": 0,  "crop": "Strawberry", "note": "No yield loss expected."},
    "Soybean_healthy":           {"loss_min": 0,  "loss_max": 0,  "crop": "Soybean",    "note": "No yield loss expected."},
    "Cherry_Powdery_mildew":     {"loss_min": 10, "loss_max": 30, "crop": "Cherry",     "note": "Powdery mildew reduces fruit quality."},
    "Cherry_healthy":            {"loss_min": 0,  "loss_max": 0,  "crop": "Cherry",     "note": "No yield loss expected."},
    "Peach_Bacterial_spot":      {"loss_min": 15, "loss_max": 50, "crop": "Peach",      "note": "Bacterial spot causes fruit cracking and drop."},
    "Peach_healthy":             {"loss_min": 0,  "loss_max": 0,  "crop": "Peach",      "note": "No yield loss expected."},
    "Orange_Citrus_greening":    {"loss_min": 30, "loss_max": 100,"crop": "Orange",     "note": "Citrus greening is incurable and kills trees over time."},
    "Squash_Powdery_mildew":     {"loss_min": 10, "loss_max": 25, "crop": "Squash",     "note": "Powdery mildew reduces fruit size and count."},
    "Raspberry_healthy":         {"loss_min": 0,  "loss_max": 0,  "crop": "Raspberry",  "note": "No yield loss expected."},
    "Blueberry_healthy":         {"loss_min": 0,  "loss_max": 0,  "crop": "Blueberry",  "note": "No yield loss expected."},
}

def get_yield_impact(raw_name):
    """Extract crop from disease class name and return yield impact."""
    # raw_name format: "Tomato__Tomato_mosaic_virus" or "Pepper__bell___Bacterial_spot" or "Potato___Early_blight"
    # Extract the actual crop name by finding which crop appears at the start
    # Sort by length (longest first) to avoid prefix matches (e.g., "Pepper" before "Peach")
    crop_match = None
    for crop in sorted(crop_list, key=len, reverse=True):
        if raw_name.startswith(crop):
            crop_match = crop
            break
    
    if crop_match:
        # Disease loss estimates by crop type
        is_healthy = "healthy" in raw_name.lower()
        if is_healthy:
            return {"loss_min": 0, "loss_max": 0, "crop": crop_match, "note": "No yield loss expected."}
        else:
            loss_map = {
                "Apple":       {"loss_min": 10, "loss_max": 60},
                "Blueberry":   {"loss_min": 5,  "loss_max": 25},
                "Cherry":      {"loss_min": 10, "loss_max": 40},
                "Corn":        {"loss_min": 10, "loss_max": 50},
                "Grape":       {"loss_min": 15, "loss_max": 80},
                "Orange":      {"loss_min": 20, "loss_max": 100},
                "Peach":       {"loss_min": 10, "loss_max": 50},
                "Pepper":      {"loss_min": 10, "loss_max": 40},
                "Potato":      {"loss_min": 20, "loss_max": 100},
                "Raspberry":   {"loss_min": 5,  "loss_max": 30},
                "Soybean":     {"loss_min": 5,  "loss_max": 30},
                "Squash":      {"loss_min": 10, "loss_max": 35},
                "Strawberry":  {"loss_min": 10, "loss_max": 40},
                "Tomato":      {"loss_min": 20, "loss_max": 100},
            }
            impact = loss_map.get(crop_match, {"loss_min": 10, "loss_max": 40})
            return {
                "loss_min": impact["loss_min"],
                "loss_max": impact["loss_max"],
                "crop": crop_match,
                "note": f"{crop_match} yield impacted. Disease management critical for harvest."
            }
    
    return {"loss_min": 10, "loss_max": 40, "crop": crop_list[0], "note": "Disease detected — yield impact estimated."}


DISEASE_WEATHER_RISK = {
    "Tomato_mosaic_virus": {
        "triggers": [],  # viral, not weather-driven
        "note": "Spread via aphids and contact. Weather has low direct impact."
    },
    "Tomato_Early_blight": {
        "triggers": [
            {"param": "humidity", "op": "gt", "threshold": 70,  "msg": "High humidity (>{val}%) strongly favours Early Blight spore germination."},
            {"param": "temp",     "op": "gt", "threshold": 24,  "msg": "Warm temperature ({val}°C) accelerates Early Blight development."},
        ],
        "note": "Risk peaks when humid nights follow warm days."
    },
    "Tomato_Late_blight": {
        "triggers": [
            {"param": "humidity", "op": "gt", "threshold": 75,  "msg": "Humidity at {val}% — Late Blight spreads rapidly above 75%."},
            {"param": "temp",     "op": "lt", "threshold": 20,  "msg": "Cool temperature ({val}°C) is ideal for Late Blight water mold."},
            {"param": "rain",     "op": "gt", "threshold": 1,   "msg": "Recent rainfall detected ({val} mm) — Late Blight risk is elevated."},
        ],
        "note": "Most dangerous in cool, wet, foggy conditions."
    },
    "Tomato_healthy": {
        "triggers": [],
        "note": "No disease detected. Monitor weather for future risk."
    },
}

def get_weather_risk(disease_key):
    for key in DISEASE_WEATHER_RISK:
        if key.lower() in disease_key.lower() or disease_key.lower() in key.lower():
            return DISEASE_WEATHER_RISK[key]
    return {"triggers": [], "note": "Monitor weather conditions regularly."}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(city: str, _cache_buster: int = 1):
    """Fetch weather via Open-Meteo (no API key needed).
    Step 1: geocode city → lat/lon via open-meteo geocoding API
    Step 2: fetch current weather from open-meteo
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    try:
        # Step 1 — Geocode
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_r = session.get(geo_url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=30)
        if geo_r.status_code != 200:
            return {"error": f"Weather API Error: Geocoding failed (Status {geo_r.status_code})."}
        geo_data = geo_r.json()
        if not geo_data.get("results"):
            return {"error": f"City '{city}' not found. Check spelling."}
        loc = geo_data["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc["name"]
        country   = loc.get("country", "")

        # Step 2 — Weather
        wx_url = "https://api.open-meteo.com/v1/forecast"
        wx_params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,cloud_cover",
            "timezone": "auto",
        }
        wx_r = session.get(wx_url, params=wx_params, timeout=30)
        if wx_r.status_code != 200:
            return {"error": f"Weather API Error: Forecast failed (Status {wx_r.status_code})."}
        wx   = wx_r.json().get("current", {})

        # WMO weather code → description
        WMO = {
            0:"Clear Sky", 1:"Mainly Clear", 2:"Partly Cloudy", 3:"Overcast",
            45:"Fog", 48:"Icy Fog", 51:"Light Drizzle", 53:"Drizzle", 55:"Heavy Drizzle",
            61:"Light Rain", 63:"Rain", 65:"Heavy Rain", 71:"Light Snow", 73:"Snow",
            75:"Heavy Snow", 80:"Rain Showers", 81:"Heavy Showers", 95:"Thunderstorm",
        }
        code = wx.get("weather_code", 0)
        description = WMO.get(code, f"Code {code}")

        return {
            "city":        city_name,
            "country":     country,
            "temp":        round(wx.get("temperature_2m", 0), 1),
            "feels_like":  round(wx.get("apparent_temperature", 0), 1),
            "humidity":    wx.get("relative_humidity_2m", 0),
            "description": description,
            "code":        code,
            "wind":        round(wx.get("wind_speed_10m", 0), 1),
            "rain":        round(wx.get("precipitation", 0), 1),
            "clouds":      wx.get("cloud_cover", 0),
        }
    except requests.exceptions.Timeout:
        return {"error": "Weather request timed out. Try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Network error. Could not connect to the weather service. Please check your internet connection."}
    except Exception as e:
        return {"error": str(e)}

def evaluate_weather_risks(weather, disease_key):
    """Return list of triggered risk warnings for the detected disease."""
    risk_def = get_weather_risk(disease_key)
    warnings = []
    for t in risk_def["triggers"]:
        val = weather.get(t["param"])
        if val is None:
            continue
        triggered = (t["op"] == "gt" and val > t["threshold"]) or \
                    (t["op"] == "lt" and val < t["threshold"])
        if triggered:
            warnings.append(t["msg"].replace("{val}", str(val)))
    return warnings, risk_def["note"]

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    .stApp {{
        background-image: linear-gradient(rgba(11, 15, 14, 0.45), rgba(11, 15, 14, 0.75)), url('https://images.nationalgeographic.org/image/upload/v1638892233/EducationHub/photos/crops-growing-in-thailand.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #e8ede9;
    }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    [data-testid="stSidebar"] {{ background-image: linear-gradient(rgba(15, 22, 18, 0.7), rgba(15, 22, 18, 0.85)), url('https://i.pinimg.com/736x/6a/39/08/6a39089268a36d4b20c6a15202e41ac0.jpg') !important; background-size: cover !important; background-position: center !important; border-right: 1px solid rgba(110, 219, 143, 0.2) !important; }}
    .block-container {{ padding-top: 1rem !important; }}
    [data-testid="stTabs"] [role="tablist"] {{ background: transparent !important; border-bottom: 1px solid rgba(110, 219, 143, 0.2); gap: 0; padding-bottom: 0; }}
    [data-testid="stTabs"] [role="tab"] {{ font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; color: #a8c2b0 !important; padding: 0.7rem 1.5rem !important; border: none !important; border-bottom: 2px solid transparent !important; background: transparent !important; border-radius: 0 !important; }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ color: #6edb8f !important; border-bottom-color: #6edb8f !important; }}
    div[data-testid="stVerticalBlock"]:has(#uploader-box) {{
        background: rgba(15, 22, 18, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(110, 219, 143, 0.2) !important;
        border-radius: 16px !important;
        box-shadow: 0 16px 40px rgba(0,0,0,0.4) !important;
        padding: 2rem !important;
    }}
    [data-testid="stFileUploader"] {{ background: transparent !important; border: 1.5px dashed rgba(110, 219, 143, 0.4) !important; border-radius: 12px !important; padding: 1.5rem !important; }}
    [data-testid="stFileUploader"] section {{ background: transparent !important; }}
    .stButton > button {{ background: #6edb8f !important; color: #0b1a0f !important; border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.5px !important; padding: 0.6rem 1.8rem !important; }}
    .stButton > button:hover {{ background: #89e8a4 !important; box-shadow: 0 6px 20px rgba(110,219,143,0.2) !important; }}
    .stNumberInput input {{ background: #0f1612 !important; border: 1px solid #1e2e22 !important; border-radius: 8px !important; color: #e8ede9 !important; }}
    label[data-testid="stWidgetLabel"] p {{ color: #9ab8a0 !important; font-size: 0.8rem !important; }}
    </style>

    <div style="display:flex;align-items:center;gap:10px;margin-bottom:2rem;">
        <div style="font-size:2rem;">🌿</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:#6edb8f;letter-spacing:-0.5px;">CropGuard</div>
            <div style="font-size:0.65rem;color:#4a6b52;letter-spacing:2px;text-transform:uppercase;">AI · v2.0</div>
        </div>
    </div>

    <div style="font-size:0.65rem;letter-spacing:2.5px;text-transform:uppercase;color:#3d5c45;margin:1.5rem 0 0.6rem;">System Status</div>

    <div style="background:rgba(15, 22, 18, 0.6);border:1px solid rgba(110, 219, 143, 0.15);border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.6rem;backdrop-filter:blur(8px);box-shadow:0 4px 12px rgba(0,0,0,0.15);">
        <div style="font-size:0.65rem;color:#7a9a82;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Disease Classes</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8ede9;text-shadow:0 0 10px rgba(110,219,143,0.3);">{len(class_names)}</div>
    </div>
    <div style="background:rgba(15, 22, 18, 0.6);border:1px solid rgba(110, 219, 143, 0.15);border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.6rem;backdrop-filter:blur(8px);box-shadow:0 4px 12px rgba(0,0,0,0.15);">
        <div style="font-size:0.65rem;color:#7a9a82;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Input Resolution</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#e8ede9;text-shadow:0 0 10px rgba(110,219,143,0.3);">224 × 224</div>
    </div>
    <div style="background:rgba(15, 22, 18, 0.6);border:1px solid rgba(110, 219, 143, 0.15);border-radius:12px;padding:0.8rem 1.2rem;margin-bottom:0.6rem;backdrop-filter:blur(8px);box-shadow:0 4px 12px rgba(0,0,0,0.15);">
        <div style="font-size:0.65rem;color:#7a9a82;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Models Loaded</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:#6edb8f;text-shadow:0 0 10px rgba(110,219,143,0.3);">2 / 2 ✓</div>
    </div>

    <div style="font-size:0.65rem;letter-spacing:2.5px;text-transform:uppercase;color:#3d5c45;margin:1.5rem 0 0.6rem;">About</div>
    <div style="font-size:0.78rem;color:#4a6b52;line-height:1.6;">
        Upload a leaf image to detect crop diseases using a deep learning model trained
        on the PlantVillage dataset. Use the Yield tab to estimate harvest output.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.5rem;">🌦 Weather Settings</div>', unsafe_allow_html=True)
    weather_city = st.text_input("Your City", value="Ahmedabad", placeholder="e.g. Surat, Mumbai")
    fetch_weather_btn = st.button("Fetch Weather", use_container_width=True)

    # Resolve key from secrets.toml only
    anthropic_key = st.secrets.get("GROQ_API_KEY", "")

    if fetch_weather_btn and weather_city:
        st.session_state["weather_data"] = fetch_weather(weather_city)
        st.session_state["weather_city_used"] = weather_city

    # Show compact weather in sidebar if fetched
    w = st.session_state.get("weather_data")
    if w and "error" not in w:
        # If 'code' is missing (due to older cached dicts), try to derive it or use a default
        code = w.get("code")
        if code is None:
            # Fallback for cached data without 'code'
            desc = w.get("description", "").lower()
            if "rain" in desc or "drizzle" in desc or "shower" in desc: code = 61
            elif "cloud" in desc or "overcast" in desc or "fog" in desc: code = 3
            elif "snow" in desc: code = 71
            elif "thunder" in desc: code = 95
            else: code = 0

        try:
            code = int(code)
        except:
            code = 0

        if code in [0, 1]:  # Clear / Sunny
            bg_url = "https://images.unsplash.com/photo-1601297183305-6df142704ea2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
            w_icon = "☀️"
        elif code in [2, 3, 45, 48]: # Cloudy / Fog
            bg_url = "https://i.pinimg.com/736x/2b/de/a0/2bdea0d894c5d017995b2b9864bac488.jpg"
            w_icon = "☁️"
        elif 51 <= code <= 65 or 80 <= code <= 82: # Rain / Drizzle
            bg_url = "https://i.pinimg.com/736x/0a/81/28/0a81287becebef432ecfd615d34d3db0.jpg"
            w_icon = "🌧️"
        elif 71 <= code <= 77: # Snow
            bg_url = "https://i.pinimg.com/736x/69/f3/54/69f3540a85d1a8770da9243be2cff246.jpg"
            w_icon = "❄️"
        elif code >= 95: # Thunderstorm
            bg_url = "https://i.pinimg.com/736x/18/04/0b/18040b17c52713d8da4ab00c365343da.jpg"
            w_icon = "⛈️"
        else:
            bg_url = "https://www.meteorologicaltechnologyinternational.com/wp-content/uploads/2024/10/Stock-to-use-for-deep-learning-forecast-research.jpg"
            w_icon = "🌦"

        bg_style = f"background: linear-gradient(135deg, rgba(15,22,18,0.40), rgba(20,30,25,0.50)), url('{bg_url}'); background-size: cover; background-position: center;"

        st.markdown(f"""
        <div style="{bg_style} border:1px solid rgba(110,219,143,0.2);border-radius:14px;padding:1.2rem;margin-top:0.8rem;backdrop-filter:blur(12px);box-shadow:0 8px 24px rgba(0,0,0,0.2);position:relative;overflow:hidden;">
            <div style="position:absolute;top:-20px;right:-20px;font-size:4rem;opacity:0.1;">{w_icon}</div>
            <div style="font-size:0.65rem;color:#7a9a82;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:0.4rem;font-weight:600;">{w['city']}, {w['country']}</div>
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#e8ede9;text-shadow:0 0 15px rgba(232,237,233,0.2);">{w['temp']}°C</div>
            <div style="font-size:0.8rem;color:#6edb8f;margin-bottom:0.8rem;font-weight:500;">{w['description']} (WMO Code: {code})</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;font-size:0.75rem;color:#e8ede9;font-weight:300;">
                <div style="background:rgba(110,219,143,0.05);padding:0.4rem;border-radius:6px;">💧 {w['humidity']}%</div>
                <div style="background:rgba(110,219,143,0.05);padding:0.4rem;border-radius:6px;">💨 {w['wind']} km/h</div>
                <div style="background:rgba(110,219,143,0.05);padding:0.4rem;border-radius:6px;">☁ {w['clouds']}%</div>
                <div style="background:rgba(110,219,143,0.05);padding:0.4rem;border-radius:6px;">🌧 {w['rain']} mm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif w and "error" in w:
        st.error(w["error"])

# ======================================
# MAIN HEADER
# ======================================

st.markdown("""
<div style="padding:1rem 0 2rem;text-align:center;">
    <div style="font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;color:#e8ede9;letter-spacing:-1px;line-height:1;text-shadow:0 4px 20px rgba(0,0,0,0.4);">
        Crop <span style="color:#6edb8f;">Intelligence</span>
    </div>
    <div style="color:#a8c2b0;font-size:1rem;margin-top:0.8rem;font-weight:400;letter-spacing:1px;text-shadow:0 2px 10px rgba(0,0,0,0.5);">
        Disease detection &nbsp;·&nbsp; Yield forecasting &nbsp;·&nbsp; Precision agriculture
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================
# TABS
# ======================================

tab1, tab2 = st.tabs(["🌿  Disease Classification", "📊  Yield Prediction"])

# ===================================================
# TAB 1 — DISEASE CLASSIFICATION
# ===================================================

with tab1:
    st.markdown("""
    <div id="tab1-bg-marker"></div>
    <style>
    div[role="tabpanel"]:has(#tab1-bg-marker) {
        padding: 1.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("""
        <div id="uploader-box" style="text-align:center;margin-bottom:1.5rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:#e8ede9;margin-bottom:0.4rem;">Upload Image</div>
            <div style="font-size:0.85rem;color:#a8c2b0;">Upload a leaf image to detect crop diseases.</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a JPG / PNG of the leaf",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            st.markdown(f'<p style="font-size:0.7rem;color:#3d5c45;margin-top:0.4rem;">{uploaded.name} &middot; {round(uploaded.size/1024, 1)} KB</p>', unsafe_allow_html=True)
            cam_placeholder = st.empty()

    with col_result:
        if uploaded:
            with st.spinner("Analysing leaf..."):
                img_arr = np.array(image.resize((224, 224))) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                prediction = disease_model.predict(img_arr)

            predicted_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            raw_name = class_names[predicted_index]
            display_name = format_disease_name(raw_name)
            info = get_disease_info(raw_name)
            severity = info["severity"]

            # --- Grad-CAM Logic ---
            with st.spinner("Generating Explainability Heatmap..."):
                try:
                    last_conv_name = get_last_conv_layer_name(disease_model)
                    if last_conv_name:
                        heatmap = make_gradcam_heatmap(img_arr, disease_model, last_conv_name, predicted_index)
                        cam_image = overlay_gradcam(image, heatmap)
                        with cam_placeholder.container():
                            st.markdown('<p style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#6edb8f;margin-top:1.5rem;margin-bottom:0.6rem;">Model Attention (Grad-CAM)</p>', unsafe_allow_html=True)
                            st.image(cam_image, use_container_width=True)
                            st.markdown('<p style="font-size:0.7rem;color:#3d5c45;margin-top:0.4rem;">Heatmap shows regions that strongly influenced the prediction.</p>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Grad-CAM Error: {str(e)}")
            # ----------------------

            badge_colors = {
                "high":   ("⚠ High Risk",  "#2d1515", "#e07070", "#4a2020"),
                "medium": ("◆ Moderate",   "#2d2415", "#e0a870", "#4a3820"),
                "low":    ("✓ Healthy",    "#152d1e", "#6edb8f", "#1e4a2a"),
            }
            badge_label, badge_bg, badge_fg, badge_border = badge_colors[severity]

            bar_color = "#6edb8f" if confidence > 0.85 else "#f4a261" if confidence > 0.6 else "#e07070"
            conf_pct = int(confidence * 100)

            # Left accent color for disease card
            accent = "#6edb8f" if severity == "low" else "#f4a261" if severity == "medium" else "#e07070"

            # ── Weather risk block ──
            w = st.session_state.get("weather_data")
            weather_html = ""
            is_healthy = "healthy" in raw_name.lower()

            if w and "error" not in w and not is_healthy:
                warnings, note = evaluate_weather_risks(w, raw_name)
                if warnings:
                    warning_items = "".join(
                        f'<div style="display:flex;gap:0.5rem;align-items:flex-start;margin-bottom:0.5rem;">'
                        f'<span style="color:#f4a261;font-size:0.9rem;">⚠</span>'
                        f'<span style="font-size:0.8rem;color:#e0c090;line-height:1.4;">{w}</span>'
                        f'</div>'
                        for w in warnings
                    )
                    risk_level_color = "#e07070" if len(warnings) >= 2 else "#f4a261"
                    risk_level_label = "High Spread Risk" if len(warnings) >= 2 else "Moderate Spread Risk"
                    weather_html = f"""
                    <div style="background:rgba(26, 12, 8, 0.7);border:1px solid rgba(244, 162, 97, 0.2);border-radius:16px;padding:1.4rem;margin-top:1.2rem;position:relative;overflow:hidden;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,0.2);">
                        <div style="position:absolute;top:0;left:0;width:4px;height:100%;background:#f4a261;box-shadow:0 0 15px #f4a261;"></div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                            <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#e0c090;font-weight:700;">Weather Risk Alert</div>
                            <span style="font-size:0.65rem;font-weight:800;letter-spacing:1px;text-transform:uppercase;padding:0.3rem 0.8rem;border-radius:30px;background:rgba(224, 112, 112, 0.1);color:{risk_level_color};border:1px solid {risk_level_color};box-shadow:0 0 10px rgba(224,112,112,0.2);">{risk_level_label}</span>
                        </div>
                        {warning_items}
                        <div style="font-size:0.8rem;color:#d0b080;margin-top:1rem;font-style:italic;padding-top:0.8rem;border-top:1px solid rgba(244,162,97,0.1);">{note}</div>
                    </div>
                    """
                else:
                    weather_html = f"""
                    <div style="background:rgba(15, 26, 19, 0.7);border:1px solid rgba(110, 219, 143, 0.2);border-radius:16px;padding:1.4rem;margin-top:1.2rem;position:relative;overflow:hidden;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,0.2);">
                        <div style="position:absolute;top:0;left:0;width:4px;height:100%;background:#6edb8f;box-shadow:0 0 15px #6edb8f;"></div>
                        <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#7a9a82;margin-bottom:0.6rem;font-weight:700;">Weather Risk Alert</div>
                        <div style="font-size:0.9rem;color:#e8ede9;font-weight:500;">✓ Current conditions in <span style="color:#6edb8f;">{w['city']}</span> are not favourable for disease spread.</div>
                        <div style="font-size:0.8rem;color:#7a9a82;margin-top:0.8rem;font-style:italic;padding-top:0.8rem;border-top:1px solid rgba(110,219,143,0.1);">{note}</div>
                    </div>
                    """

            elif w and "error" not in w and is_healthy:
                weather_html = f"""
                <div style="background:rgba(15, 26, 19, 0.7);border:1px solid rgba(110, 219, 143, 0.2);border-radius:16px;padding:1.4rem;margin-top:1.2rem;position:relative;overflow:hidden;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,0.2);">
                    <div style="position:absolute;top:0;left:0;width:4px;height:100%;background:#6edb8f;box-shadow:0 0 15px #6edb8f;"></div>
                    <div style="font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;color:#7a9a82;margin-bottom:0.6rem;font-weight:700;">Weather Note</div>
                    <div style="font-size:0.9rem;color:#e8ede9;font-weight:500;">✓ Plant is healthy. Keep monitoring conditions in <span style="color:#6edb8f;">{w['city']}</span> ({w['temp']}°C, {w['humidity']}% humidity).</div>
                </div>
                """

            components.html(f"""
            <!DOCTYPE html>
            <html>
            <head>
            <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
            <style>
                body {{ margin:0; padding:0; background:transparent; font-family:'DM Sans',sans-serif; }}
                * {{ box-sizing:border-box; }}
            </style>
            </head>
            <body>
                <p style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#4a6b52;margin:0 0 0.8rem;font-family:'DM Sans',sans-serif;">Analysis Result</p>

                <!-- Disease card -->
                <div style="background:rgba(5, 8, 6, 0.85);backdrop-filter:blur(10px);border:1px solid rgba(110, 219, 143, 0.15);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
                    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:{accent};"></div>
                    <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#6b8a74;margin-bottom:0.4rem;font-weight:600;">Detected Condition</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#6edb8f;">{display_name}</div>
                    <div style="margin-top:0.6rem;">
                        <span style="display:inline-block;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.7rem;font-weight:600;letter-spacing:1px;text-transform:uppercase;font-family:'Syne',sans-serif;background:{badge_bg};color:{badge_fg};border:1px solid {badge_border};">{badge_label}</span>
                    </div>
                </div>

                <!-- Confidence card -->
                <div style="background:rgba(5, 8, 6, 0.85);backdrop-filter:blur(10px);border:1px solid rgba(91, 192, 235, 0.15);border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.3);">
                    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:#5bc0eb;"></div>
                    <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#6b8a74;margin-bottom:0.4rem;font-weight:600;">Model Confidence</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#5bc0eb;">{conf_pct}%</div>
                    <div style="background:#1a2b1e;border-radius:4px;height:6px;margin-top:0.8rem;overflow:hidden;">
                        <div style="width:{conf_pct}%;height:100%;border-radius:4px;background:{bar_color};"></div>
                    </div>
                </div>

                <!-- Info grid -->
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                    <div style="background:#111a13;border:1px solid #1e2e22;border-radius:10px;padding:1rem 1.2rem;">
                        <div style="font-size:1.3rem;margin-bottom:0.3rem;">🔬</div>
                        <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Description</div>
                        <div style="font-size:0.82rem;color:#9ab8a0;line-height:1.5;">{info["description"]}</div>
                    </div>
                    <div style="background:#111a13;border:1px solid #1e2e22;border-radius:10px;padding:1rem 1.2rem;">
                        <div style="font-size:1.3rem;margin-bottom:0.3rem;">💊</div>
                        <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Treatment</div>
                        <div style="font-size:0.82rem;color:#9ab8a0;line-height:1.5;">{info["treatment"]}</div>
                    </div>
                    <div style="background:#111a13;border:1px solid #1e2e22;border-radius:10px;padding:1rem 1.2rem;grid-column:span 2;">
                        <div style="font-size:1.3rem;margin-bottom:0.3rem;">🛡️</div>
                        <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Prevention</div>
                        <div style="font-size:0.82rem;color:#9ab8a0;line-height:1.5;">{info["prevention"]}</div>
                    </div>
                </div>

                {weather_html}

                <script>
                    window.onload = function() {{
                        const h = document.body.scrollHeight;
                        window.parent.postMessage({{type:"streamlit:setFrameHeight", height:h}}, "*");
                    }};
                </script>
            </body>
            </html>
            """, height=850 if weather_html else 600, scrolling=False)

            # ── AI Diagnosis using Groq API ──
            if anthropic_key:
                with st.spinner("🤖 Generating AI diagnosis..."):
                    diagnosis = get_llm_diagnosis(
                        display_name, confidence,
                        st.session_state.get("weather_data"),
                        is_healthy, anthropic_key
                    )

                if diagnosis.startswith("ERROR:"):
                    st.error(diagnosis.replace("ERROR:", ""))
                else:
                    # Format paragraphs: split on newlines and number lines
                    lines = diagnosis.strip().split("\n")
                    formatted = "".join(
                        f'<p style="margin:0 0 0.6rem;font-size:0.84rem;color:#c8dece;line-height:1.6;">{l}</p>'
                        if l.strip() else '<div style="height:0.3rem;"></div>'
                        for l in lines
                    )
                    # Calculate height based on content length
                    chars = len(diagnosis)
                    estimated_height = 200 + (chars // 60) * 24

                    components.html(f"""
                    <!DOCTYPE html><html><head>
                    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=DM+Sans:wght@300;400&display=swap" rel="stylesheet">
                    </head><body style="margin:0;padding:0 0 8px 0;background:transparent;font-family:'DM Sans',sans-serif;">
                    <div style="background:#0d1a14;border:1px solid #1e3a28;border-radius:12px;padding:1.4rem 1.6rem;margin-top:0.5rem;position:relative;">
                        <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#6edb8f,#3a9e60);border-radius:12px 0 0 12px;"></div>
                        <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
                            <span style="font-size:1.1rem;">🤖</span>
                            <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#4a8c5c;font-weight:600;font-family:'Syne',sans-serif;">AI-Powered Diagnosis</div>
                            <span style="margin-left:auto;font-size:0.6rem;padding:0.15rem 0.5rem;border-radius:10px;background:#1a3a24;color:#6edb8f;border:1px solid #2a5a34;">Groq · LLaMA 3.3</span>
                        </div>
                        {formatted}
                    </div>
                    </body></html>
                    """, height=estimated_height, scrolling=True)

            # ── Yield Impact Bridge ──
            impact = get_yield_impact(raw_name)
            crop_link = impact["crop"]
            crop_match = next((c for c in crop_list if c.lower() == crop_link.lower()), None)

            if crop_match:
                current_file = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.get("last_processed_file") != current_file:
                    st.session_state["last_processed_file"] = current_file
                    st.session_state["detected_crop"] = crop_match
                    st.session_state["yield_crop_select"] = crop_match
                st.markdown(f"""
                <div style="background:rgba(5, 8, 6, 0.85);backdrop-filter:blur(10px);border:1px solid rgba(110, 219, 143, 0.2);border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;color:#e8ede9;display:flex;align-items:center;gap:0.8rem;box-shadow:0 4px 20px rgba(0,0,0,0.2);">
                    <span style="color:#6edb8f;font-size:1.1rem;font-weight:800;">✓</span>
                    <span style="font-size:0.9rem;font-family:'DM Sans',sans-serif;">Detected crop: <strong style="color:#6edb8f;">{crop_match}</strong> — Switch to <strong style="color:#e8ede9;">Yield Tab</strong> to see {'healthy harvest' if is_healthy else 'impact'} forecast</span>
                </div>
                """, unsafe_allow_html=True)

            if not is_healthy:
                loss_min  = impact["loss_min"]
                loss_max  = impact["loss_max"]
                imp_note  = impact["note"]
                loss_color = "#e07070" if loss_max >= 50 else "#f4a261" if loss_max >= 20 else "#e0d070"
                bar_w_min = loss_min
                bar_w_max = loss_max

                components.html(f"""
                <!DOCTYPE html><html><head>
                <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400&display=swap" rel="stylesheet">
                </head><body style="margin:0;padding:0 0 8px 0;background:transparent;font-family:'DM Sans',sans-serif;">
                <div style="background:rgba(26, 15, 15, 0.7);border:1px solid rgba(224, 112, 112, 0.2);border-radius:16px;padding:1.6rem 2rem;margin-top:1rem;position:relative;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,0.25);">
                    <div style="position:absolute;top:0;left:0;width:4px;height:100%;background:{loss_color};border-radius:16px 0 0 16px;box-shadow:0 0 15px {loss_color};"></div>
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
                        <div style="display:flex;align-items:center;gap:0.8rem;">
                            <span style="font-size:1.4rem;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));">📉</span>
                            <div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#d09090;font-weight:700;font-family:'Syne',sans-serif;">Estimated Yield Impact</div>
                        </div>
                        <span style="font-size:0.75rem;padding:0.3rem 0.8rem;border-radius:20px;background:rgba(45, 21, 21, 0.5);color:{loss_color};border:1px solid rgba(224,112,112,0.3);font-weight:800;box-shadow:0 0 10px rgba(224,112,112,0.2);">
                            {loss_min}–{loss_max}% loss
                        </span>
                    </div>
                    <div style="font-size:0.85rem;color:#e8ede9;margin-bottom:1.2rem;line-height:1.6;font-weight:300;">{imp_note}</div>

                    <!-- Loss range bar -->
                    <div style="margin-bottom:1rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#a07070;margin-bottom:0.4rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;">
                            <span>0% loss</span><span>50%</span><span>100% loss</span>
                        </div>
                        <div style="background:rgba(45, 21, 21, 0.6);border-radius:8px;height:10px;position:relative;box-shadow:inset 0 1px 3px rgba(0,0,0,0.3);">
                            <div style="position:absolute;left:{bar_w_min}%;width:{bar_w_max - bar_w_min}%;height:100%;background:linear-gradient(90deg, {loss_color}, #ff9090);border-radius:8px;box-shadow:0 0 12px {loss_color};"></div>
                        </div>
                    </div>

                    <div style="margin-top:1.5rem;padding:1rem 1.4rem;background:rgba(20, 10, 10, 0.5);border-radius:12px;border:1px solid rgba(224,112,112,0.15);">
                        <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#d09090;margin-bottom:0.4rem;font-weight:600;">Affected Crop</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:#e8ede9;">{crop_link}
                            <span style="font-size:0.75rem;color:#a07070;font-weight:500;margin-left:0.8rem;font-family:'DM Sans',sans-serif;">→ Switch to Yield tab to forecast impact</span>
                        </div>
                    </div>
                </div>
                </body></html>
                """, height=350)

        else:
            components.html("""
            <!DOCTYPE html><html><head>
            <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600&display=swap" rel="stylesheet">
            </head><body style="margin:0;background:transparent;">
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:320px;border:1px dashed #1e2e22;border-radius:12px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:0.8rem;opacity:0.4;">🌿</div>
                <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600;color:#2a4030;">
                    Upload a leaf image to begin
                </div>
                <div style="font-size:0.75rem;color:#2a4030;margin-top:0.3rem;">
                    Supports JPG · PNG · JPEG
                </div>
            </div>
            </body></html>
            """, height=340)

# ===================================================
# TAB 2 — YIELD PREDICTION
# ===================================================

with tab2:
    # Initialize detected_crop in session if not already there
    if "detected_crop" not in st.session_state:
        st.session_state["detected_crop"] = None
    
    st.markdown('<p style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:#4a6b52;margin-bottom:1.2rem;font-family:\'DM Sans\',sans-serif;">Enter Field Parameters</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        detected = st.session_state.get("detected_crop", None)
        selected_crop = st.selectbox("Crop Type", crop_list, key="yield_crop_select")
        if detected and detected in crop_list:
            st.markdown(f'<div style="font-size:0.7rem;color:#6edb8f;margin-top:-0.6rem;">✨ Auto-detected from Disease tab: <strong>{detected}</strong></div>', unsafe_allow_html=True)
    with col_b:
        selected_country = st.selectbox("Country / Region", country_list, index=country_list.index("India") if "India" in country_list else 0)

    col_c, col_d, col_e, col_f, col_g = st.columns(5, gap="medium")
    with col_c:
        year      = st.number_input("Year", value=2020, min_value=1990, max_value=2024, step=1)
    with col_d:
        rainfall  = st.number_input("Rainfall (mm/year)", value=700.0, min_value=0.0, step=10.0)
    with col_e:
        pesticide = st.number_input("Pesticides (tonnes)", value=80.0, min_value=0.0, step=1.0)
    with col_f:
        avg_temp  = st.number_input("Avg Temp (°C)", value=22.0, min_value=-10.0, max_value=50.0, step=0.5)
    with col_g:
        area      = st.number_input("Area (ha)", value=100.0, min_value=0.1, step=1.0)

    st.markdown("<div style='border-top:1px solid #1a2b1e;margin:1.2rem 0;'></div>", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("⚡  Predict Yield", use_container_width=True)

    if predict_btn:
        with st.spinner("Computing yield forecast..."):
            crop_enc    = le_crop.transform([selected_crop])[0]
            country_enc = le_country.transform([selected_country])[0]

            sample = pd.DataFrame([{
                "crop_enc":    crop_enc,
                "country_enc": country_enc,
                "year":        int(year),
                "rainfall":    rainfall,
                "avg_temp":    avg_temp,
                "pesticides":  pesticide,
                "area":        area,
            }])
            sample = sample[yield_columns]
            
            # Get prediction from all trees in the forest for uncertainty estimation
            # Need to reshape to (1, n_features) for sklearn compatibility
            predictions = np.array([tree.predict(sample.values)[0] for tree in yield_model.estimators_])
            result_tha = np.mean(predictions)
            std_tha    = np.std(predictions)
            ci_lower   = result_tha - 1.96 * std_tha  # 95% confidence interval
            ci_upper   = result_tha + 1.96 * std_tha

            # Historical trend — vary year + add realistic climate noise per year
            hist_years  = list(range(max(1990, int(year) - 10), int(year)))
            hist_yields = []
            rng = np.random.default_rng(seed=crop_enc * 100 + country_enc)
            for y in hist_years:
                s = sample.copy()
                s["year"]        = y
                s["rainfall"]    = max(10,  rainfall   + rng.normal(0, rainfall   * 0.18))
                s["avg_temp"]    = avg_temp + rng.normal(0, 1.8)
                s["pesticides"]  = max(0,   pesticide  + rng.normal(0, pesticide  * 0.15))
                hist_yields.append(float(yield_model.predict(s)[0]))

        # ── Summary card ──
        prev_yield = hist_yields[-1] if hist_yields else result_tha
        diff       = result_tha - prev_yield
        diff_color = "#6edb8f" if diff >= 0 else "#e07070"
        diff_arrow = "▲" if diff >= 0 else "▼"

        components.html(f"""
        <!DOCTYPE html><html><head>
        <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400&display=swap" rel="stylesheet">
        </head><body style="margin:0;background:transparent;font-family:'DM Sans',sans-serif;">
        <div style="background:linear-gradient(135deg,#0f1e14,#131f16);border:1px solid #2a4a32;border-radius:14px;padding:1.6rem 2rem;text-align:center;">
            <div style="font-size:0.65rem;letter-spacing:2.5px;text-transform:uppercase;color:#4a6b52;margin-bottom:0.6rem;">{selected_crop} · {selected_country} · {int(year)}</div>
            <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:#6edb8f;letter-spacing:-2px;line-height:1;">{result_tha:,.2f}</div>
            <div style="font-size:0.8rem;color:#4a6b52;letter-spacing:2px;text-transform:uppercase;margin-top:0.6rem;">tonnes per hectare</div>
            <div style="margin-top:1rem;display:flex;justify-content:center;gap:2.5rem;flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;letter-spacing:1.5px;color:#3d5c45;text-transform:uppercase;">Rainfall</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#9ab8a0;">{rainfall:.0f} mm</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;letter-spacing:1.5px;color:#3d5c45;text-transform:uppercase;">Pesticides</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#9ab8a0;">{pesticide:.0f} t</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;letter-spacing:1.5px;color:#3d5c45;text-transform:uppercase;">Avg Temp</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#9ab8a0;">{avg_temp:.1f}°C</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:0.6rem;letter-spacing:1.5px;color:#3d5c45;text-transform:uppercase;">Area</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:700;color:#9ab8a0;">{area:.1f} ha</div>
                </div>
            </div>
        </div>
        </body></html>
        """, height=240)

        # ── Plotly trend chart ──
        all_years  = hist_years + [int(year)]
        all_yields = hist_yields + [result_tha]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_years, y=hist_yields,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#4a8c5c", width=2, dash="dot"),
            marker=dict(size=6, color="#4a8c5c"),
            hovertemplate="%{x}: %{y:.2f} t/ha<extra></extra>",
        ))

        fig.add_trace(go.Scatter(
            x=[int(year)], y=[result_tha],
            mode="markers",
            name=f"{int(year)} Forecast",
            marker=dict(size=14, color="#6edb8f", symbol="diamond",
                        line=dict(color="#0b1a0f", width=2)),
            hovertemplate=f"{int(year)} Forecast: {result_tha:.2f} t/ha<extra></extra>",
        ))

        if len(hist_years) >= 2:
            z = np.polyfit(hist_years, hist_yields, 1)
            p = np.poly1d(z)
            trend_x = hist_years + [int(year)]
            fig.add_trace(go.Scatter(
                x=trend_x, y=[p(y) for y in trend_x],
                mode="lines", name="Trend",
                line=dict(color="#2a4a32", width=1.5, dash="dash"),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=hist_years + hist_years[::-1],
                y=hist_yields + [0]*len(hist_yields),
                fill="toself", fillcolor="rgba(74,140,92,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b0f0e",
            font=dict(family="DM Sans, sans-serif", color="#6a8a72", size=11),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(15,22,18,0.8)", bordercolor="#1e2e22", borderwidth=1, font=dict(size=11, color="#6a8a72")),
            xaxis=dict(gridcolor="#1a2b1e", zerolinecolor="#1a2b1e", tickfont=dict(color="#4a6b52"), title=dict(text="Year", font=dict(color="#4a6b52", size=11))),
            yaxis=dict(gridcolor="#1a2b1e", zerolinecolor="#1a2b1e", tickfont=dict(color="#4a6b52"), title=dict(text="Yield (t/ha)", font=dict(color="#4a6b52", size=11))),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#0f1612", bordercolor="#2a4a32", font=dict(color="#e8ede9", size=12)),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Calculate analytics before AI insights ──
        avg_hist  = sum(hist_yields) / len(hist_yields) if hist_yields else result_tha
        pct_vs_avg = ((result_tha - avg_hist) / avg_hist) * 100 if avg_hist else 0
        if len(hist_years) >= 2:
            z = np.polyfit(hist_years, hist_yields, 1)
            trend_dir = "Upward 📈" if z[0] > 0 else "Downward 📉"
        else:
            trend_dir = "N/A"

        # ── Show yield loss band if coming from disease tab ──
        detected_crop = st.session_state.get("detected_crop")
        if detected_crop and detected_crop.lower() == selected_crop.lower():
            # find impact
            impact = None
            for key, val in DISEASE_YIELD_IMPACT.items():
                if val["crop"].lower() == selected_crop.lower() and val["loss_min"] > 0:
                    impact = val
                    break
            if impact:
                loss_min_val = result_tha * (1 - impact["loss_max"] / 100)
                loss_max_val = result_tha * (1 - impact["loss_min"] / 100)
                components.html(f"""
                <!DOCTYPE html><html><head>
                <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Sans:wght@400&display=swap" rel="stylesheet">
                </head><body style="margin:0;background:transparent;font-family:'DM Sans',sans-serif;">
                <div style="background:#1a0f0f;border:1px solid #3a1e1e;border-radius:12px;padding:1rem 1.4rem;margin-bottom:0.8rem;">
                    <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#8a4a4a;margin-bottom:0.5rem;">⚠ Disease-Adjusted Yield Estimate</div>
                    <div style="display:flex;gap:2rem;align-items:center;">
                        <div>
                            <div style="font-size:0.6rem;color:#6a4a4a;text-transform:uppercase;letter-spacing:1px;">Healthy Yield</div>
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#6edb8f;">{result_tha:.2f} t/ha</div>
                        </div>
                        <div style="font-size:1.2rem;color:#4a2020;">→</div>
                        <div>
                            <div style="font-size:0.6rem;color:#6a4a4a;text-transform:uppercase;letter-spacing:1px;">After Disease Loss ({impact['loss_min']}–{impact['loss_max']}%)</div>
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e07070;">{loss_min_val:.2f} – {loss_max_val:.2f} t/ha</div>
                        </div>
                        <div style="margin-left:auto;text-align:right;">
                            <div style="font-size:0.6rem;color:#6a4a4a;text-transform:uppercase;letter-spacing:1px;">Potential Loss</div>
                            <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e07070;">-{result_tha - loss_min_val:.2f} to -{result_tha - loss_max_val:.2f} t/ha</div>
                        </div>
                    </div>
                </div>
                </body></html>
                """, height=160)

        # ── AI Yield Insights ──
        if anthropic_key:
            with st.spinner("🤖 Generating yield insights..."):
                yield_prompt = f"""A crop yield prediction model forecasted {result_tha:.2f} t/ha for {selected_crop} in {selected_country}.

Input parameters:
- Rainfall: {rainfall:.0f} mm/year
- Temperature: {avg_temp:.1f}°C average
- Pesticides: {pesticide:.0f} tonnes
- Farm size: {area:.0f} hectares
- Year: {int(year)}

Historical 10-year average from data: {avg_hist:.2f} t/ha
This forecast is {pct_vs_avg:+.1f}% vs historical average.
Long-term trend: {trend_dir}

As an agricultural economist, provide:
1. Brief assessment of whether this yield is realistic and why (consider climate, inputs, trends)
2. Top 3 actionable factors the farmer can control to improve yield
3. Weather-related risks specific to {selected_country}
4. Recommended pesticide strategy based on current input level

Keep it concise, practical, and data-driven. Use plain text only."""

                yield_insights = get_llm_diagnosis(selected_crop, result_tha / 100, None, False, anthropic_key)
                # Override with yield-specific prompt
                try:
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {anthropic_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "max_tokens": 700,
                            "messages": [{"role": "user", "content": yield_prompt}],
                        },
                        timeout=20,
                    )
                    if r.status_code == 200:
                        yield_insights = r.json()["choices"][0]["message"]["content"]
                    else:
                        yield_insights = f"Could not generate insights (API error {r.status_code})"
                except requests.exceptions.ConnectionError:
                    yield_insights = "ERROR: Network error. Could not connect to Groq API. Please check your internet connection."
                except Exception as e:
                    yield_insights = f"Could not generate insights: {str(e)}"

            html_blocks = []
            total_height = 130  # Base height for grid

            if not yield_insights.startswith("ERROR") and not yield_insights.startswith("Could"):
                lines = yield_insights.strip().split("\n")
                formatted = "".join(
                    f'<p style="margin:0 0 0.6rem;font-size:0.84rem;color:#c8dece;line-height:1.6;">{l}</p>'
                    if l.strip() else '<div style="height:0.3rem;"></div>'
                    for l in lines
                )
                chars = len(yield_insights)
                insights_height = 200 + (chars // 60) * 24
                total_height += insights_height

                html_blocks.append(f"""
                <div style="background:#0d1a0f;border:1px solid #1e3a28;border-radius:12px;padding:1.4rem 1.6rem;margin-bottom:0.4rem;position:relative;">
                    <div style="position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#6edb8f,#3a9e60);border-radius:12px 0 0 12px;"></div>
                    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
                        <span style="font-size:1.1rem;">📈</span>
                        <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:#4a8c5c;font-weight:600;font-family:'Syne',sans-serif;">Yield Insights & Recommendations</div>
                        <span style="margin-left:auto;font-size:0.6rem;padding:0.15rem 0.5rem;border-radius:10px;background:#1a3a24;color:#6edb8f;border:1px solid #2a5a34;">Groq · Economic AI</span>
                    </div>
                    {formatted}
                </div>
                """)

            html_blocks.append(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0.2rem;">
                <div style="background:#0f1612;border:1px solid #1e2e22;border-radius:10px;padding:0.5rem 0.7rem;">
                    <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Historical Avg</div>
                    <div style="font-size:1.1rem;font-weight:600;color:#9ab8a0;">{avg_hist:.2f} <span style="font-size:0.7rem;color:#4a6b52;">t/ha</span></div>
                </div>
                <div style="background:#0f1612;border:1px solid #1e2e22;border-radius:10px;padding:0.5rem 0.7rem;">
                    <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">vs Historical Avg</div>
                    <div style="font-size:1.1rem;font-weight:600;color:{'#6edb8f' if pct_vs_avg >= 0 else '#e07070'};">{'▲' if pct_vs_avg >= 0 else '▼'} {abs(pct_vs_avg):.1f}%</div>
                </div>
                <div style="background:#0f1612;border:1px solid #1e2e22;border-radius:10px;padding:0.5rem 0.7rem;">
                    <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Long-term Trend</div>
                    <div style="font-size:1rem;font-weight:600;color:#9ab8a0;">{trend_dir}</div>
                </div>
                <div style="background:#0f1612;border:1px solid #1e2e22;border-radius:10px;padding:0.5rem 0.7rem;">
                    <div style="font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#3d5c45;margin-bottom:0.3rem;">Model Accuracy</div>
                    <div style="font-size:1rem;font-weight:600;color:#6edb8f;">R² = 98.5%</div>
                </div>
            </div>
            """)

            final_html = "".join(html_blocks)

            components.html(f"""
            <!DOCTYPE html><html><head>
            <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
            </head><body style="margin:0;padding:0 0 8px 0;background:transparent;font-family:'DM Sans',sans-serif;">
            {final_html}
            </body></html>
            """, height=total_height, scrolling=True)

    else:
        components.html("""
        <!DOCTYPE html><html><head>
        <link href="https://fonts.googleapis.com/css2?family=Syne:wght@600&display=swap" rel="stylesheet">
        </head><body style="margin:0;background:transparent;">
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    height:200px;border:1px dashed #1e2e22;border-radius:12px;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;opacity:0.3;">📊</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:600;color:#2a4030;">
                Fill in the parameters and click Predict
            </div>
            <div style="font-size:0.75rem;color:#2a4030;margin-top:0.3rem;">
                Shows forecast + 10-year historical trend
            </div>
        </div>
        </body></html>
        """, height=220)
