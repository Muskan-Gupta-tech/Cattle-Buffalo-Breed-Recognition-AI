"""
CattleAI — Final Flask Backend (Fixed)
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
import tensorflow as tf
import numpy as np
from PIL import Image
import requests, os, cv2, json, uuid
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

MONGO_URI      = "mongodb://localhost:27017/"
WEATHER_KEY    = "YOUR_OPENWEATHERMAP_KEY"
IMG_SIZE       = 224
UPLOAD_FOLDER  = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db      = client["cattle_ai"]
    users   = db["users"]
    history = db["history"]
    print("✅ MongoDB connected")
except Exception as e:
    print(f"⚠️  MongoDB offline — running without DB")
    db = None

def load_model_safe(name):
    for ext in ['.keras', '.h5']:
        path = f"models/{name}{ext}"
        if os.path.exists(path):
            try:
                m = tf.keras.models.load_model(path)
                print(f"✅ {name} loaded ({ext})")
                return m
            except Exception as e:
                print(f"⚠️  {name}{ext} failed: {e}")
    print(f"⚠️  {name} not found in models/")
    return None

breed_model  = load_model_safe("breed_model")
health_model = load_model_safe("health_model")
bcs_model    = load_model_safe("bcs_model")

def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

breed_classes  = load_json("breed_classes.json",  {})
health_classes = load_json("health_classes.json", {})
bcs_classes    = load_json("bcs_classes.json",    {})

idx_to_breed  = {v: k for k, v in breed_classes.items()}
idx_to_health = {v: k for k, v in health_classes.items()}
idx_to_bcs    = {v: k for k, v in bcs_classes.items()}

print(f"✅ Breed  classes ({len(idx_to_breed)}): {list(idx_to_breed.values())}")
print(f"✅ Health classes: {list(idx_to_health.values())}")
print(f"✅ BCS    classes: {list(idx_to_bcs.values())}")

HEALTH_DISPLAY = {
    'healthy':        '✅ Healthy',
    'lumpy':          '⚠️ Lumpy Skin Disease',
    'foot-and-mouth': '⚠️ Foot & Mouth Disease',
}
def format_health(raw):
    return HEALTH_DISPLAY.get(raw.lower().strip(), f'⚠️ {raw}')

BCS_MEANING = {
    'BCS_2': {'score':2, 'label':'Thin',      'advice':'Increase feed immediately.'},
    'BCS_3': {'score':3, 'label':'Ideal',      'advice':'Body condition is optimal.'},
    'BCS_4': {'score':4, 'label':'Good',       'advice':'Good condition. Monitor feeding.'},
    'BCS_5': {'score':5, 'label':'Overweight', 'advice':'Reduce feed, increase exercise.'},
}

MILK_BASE = {
    'Ayrshire':18,'Jersey':20,'Guernesy':17,
    'Sahiwal':14,'Gir':12,'Red_Sindhi':10,
    'Rathi':7,'Nagpuri':7,'Bhadwari':6,
    'Khillari':4,'Bargur':4,'Dangi':4,
    'Alambadi':4,'Amritmahal':3,'Banni':5,'Vechur':3,
    'default':6
}
HEAT_TOL = {
    'Gir':'high','Sahiwal':'high','Red_Sindhi':'high',
    'Rathi':'high','Bargur':'high','Banni':'high',
    'Alambadi':'high','Khillari':'high',
    'Jersey':'low','Ayrshire':'low','Guernesy':'low',
    'Nagpuri':'moderate','Bhadwari':'moderate',
}

def predict_milk(breed, temp):
    base = MILK_BASE.get(breed, MILK_BASE['default'])
    tol  = HEAT_TOL.get(breed, 'moderate')
    if temp <= 22:   r = 0.0
    elif temp <= 27: r = 0.05
    elif temp <= 32: r = {'high':0.08,'moderate':0.15,'low':0.22}[tol]
    elif temp <= 38: r = {'high':0.12,'moderate':0.25,'low':0.35}[tol]
    else:            r = {'high':0.18,'moderate':0.35,'low':0.50}[tol]
    return round(base*(1-r),1), base, round(r*100,1)

BREED_CLIMATE = {
    'Gir':      {'t':(18,35),'h':(40,80)},
    'Sahiwal':  {'t':(15,38),'h':(35,85)},
    'Jersey':   {'t':(8,24), 'h':(45,75)},
    'Ayrshire': {'t':(5,22), 'h':(40,70)},
    'Nagpuri':  {'t':(15,38),'h':(40,80)},
    'default':  {'t':(10,35),'h':(30,80)},
}

def climate_advice(breed, temp, hum):
    p = BREED_CLIMATE.get(breed, BREED_CLIMATE['default'])
    issues = []
    if temp > p['t'][1]: issues.append(f"heat stress (>{p['t'][1]}°C)")
    if temp < p['t'][0]: issues.append(f"cold stress (<{p['t'][0]}°C)")
    if hum  > p['h'][1]: issues.append(f"high humidity (>{p['h'][1]}%)")
    if not issues:
        return "Excellent", f"{breed} suits current conditions ({temp}°C, {hum}% RH)."
    suit   = "Poor" if len(issues) >= 2 else "Moderate"
    advice = f"Issues: {'; '.join(issues)}. "
    if temp > p['t'][1]: advice += "Provide shade and water. "
    if hum  > p['h'][1]: advice += "Improve ventilation. "
    return suit, advice.strip()

def get_weather(city="Chennai"):
    if WEATHER_KEY == "YOUR_OPENWEATHERMAP_KEY":
        return {"temp":30.0,"humidity":70,"city":city,"mock":True}
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?q={city}&appid={WEATHER_KEY}&units=metric")
        d = requests.get(url, timeout=5).json()
        return {"temp":round(d["main"]["temp"],1),
                "humidity":d["main"]["humidity"],
                "city":d["name"],"mock":False}
    except:
        return {"temp":30.0,"humidity":70,"city":city,"mock":True}

def generate_gradcam(model, img_array, class_idx, save_path):
    try:
        img_t = tf.Variable(tf.cast(img_array, tf.float32))
        with tf.GradientTape() as tape:
            tape.watch(img_t)
            preds = model(img_t, training=False)
            loss  = preds[:, class_idx]
        grads   = tape.gradient(loss, img_t)
        grads   = tf.abs(grads)
        heatmap = tf.reduce_max(grads, axis=-1)[0].numpy()
        orig    = (img_array[0]*255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
        h_r     = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        h_col   = cv2.applyColorMap(np.uint8(255*h_r), cv2.COLORMAP_JET)
        overlaid = cv2.addWeighted(orig_bgr, 0.6, h_col, 0.4, 0)
        cv2.imwrite(save_path, overlaid)
        return True
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return False

def preprocess(file_path):
    img = Image.open(file_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── ROUTES ──────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

# FIX 1: Accept GET (page load) AND POST (form submit)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("index.html")  # just show the app

    data     = request.json or {}
    username = data.get("username", "").strip()
    if not username:
        return jsonify({"success": False, "message": "Username required"})

    if db is not None:
        if users.find_one({"username": username}):
            return jsonify({"success": True, "message": "Welcome back!", "newUser": False})
        users.insert_one({"username": username, "created_at": datetime.now()})
        return jsonify({"success": True, "message": "Account created!", "newUser": True})

    return jsonify({"success": True, "message": "Logged in (offline mode)", "newUser": False})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success":False,"message":"No image uploaded"})
    file     = request.files["image"]
    username = request.form.get("username","guest")
    city     = request.form.get("city","Chennai")
    if file.filename == "":
        return jsonify({"success":False,"message":"Empty filename"})

    uid      = str(uuid.uuid4())[:8]
    ext      = os.path.splitext(file.filename)[1] or ".jpg"
    filename = f"{uid}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    img_array = preprocess(filepath)

    result = {
        "success":True,
        "image_path":        f"static/uploads/{filename}",
        "heatmap_path":      None,
        "breed":             "Unknown",
        "confidence":        0.0,
        "top3":              [],
        "health":            "Unknown",
        "health_raw":        "unknown",
        "health_confidence": 0.0,
        "bcs_label":         "BCS_3",
        "bcs_score":         3,
        "bcs_meaning":       "Ideal",
        "bcs_advice":        "Normal body condition.",
        "temperature":       None,
        "humidity":          None,
        "city":              city,
        "climate_suitability":"Unknown",
        "climate_advice":    "Weather data unavailable.",
        "milk_prediction":   None,
        "milk_base":         None,
        "milk_reduction_pct":None,
        "timestamp":         datetime.now().isoformat()
    }

    if breed_model is not None:
        try:
            preds = breed_model.predict(img_array, verbose=0)
            top3  = []
            for idx in np.argsort(preds[0])[::-1][:3]:
                top3.append({
                    "breed":      idx_to_breed.get(int(idx), f'Class_{idx}'),
                    "confidence": round(float(preds[0][idx])*100, 2)
                })
            result.update({"breed":top3[0]['breed'],"confidence":top3[0]['confidence'],"top3":top3})
            hm_path = os.path.join(HEATMAP_FOLDER, f"heatmap_{uid}.jpg")
            if generate_gradcam(breed_model, img_array, int(np.argmax(preds[0])), hm_path):
                result["heatmap_path"] = f"static/heatmaps/heatmap_{uid}.jpg"
        except Exception as e:
            print(f"Breed error: {e}")

    if health_model is not None:
        try:
            h_pred    = health_model.predict(img_array, verbose=0)
            h_idx     = int(np.argmax(h_pred[0]))
            raw_label = idx_to_health.get(h_idx, 'unknown')
            h_conf    = round(float(np.max(h_pred[0]))*100, 1)
            result.update({"health":format_health(raw_label),"health_raw":raw_label,"health_confidence":h_conf})
        except Exception as e:
            print(f"Health error: {e}")

    if bcs_model is not None:
        try:
            bcs_pred  = bcs_model.predict(img_array, verbose=0)
            bcs_idx   = int(np.argmax(bcs_pred[0]))
            bcs_label = idx_to_bcs.get(bcs_idx, 'BCS_3')
            info      = BCS_MEANING.get(bcs_label, {'score':3,'label':'Ideal','advice':'Normal.'})
            result.update({"bcs_label":bcs_label,"bcs_score":info['score'],"bcs_meaning":info['label'],"bcs_advice":info['advice']})
        except Exception as e:
            print(f"BCS error: {e}")
    else:
        if "healthy" in result.get("health_raw",""):
            result.update({"bcs_score":3,"bcs_meaning":"Ideal (est.)"})
        else:
            result.update({"bcs_score":2,"bcs_meaning":"Thin (est.)"})

    weather = get_weather(city)
    temp    = weather["temp"]
    hum     = weather["humidity"]
    suit, advice = climate_advice(result["breed"], temp, hum)
    result.update({"temperature":temp,"humidity":hum,"city":weather["city"],
                   "climate_suitability":suit,"climate_advice":advice,"weather_mock":weather.get("mock",True)})

    milk, base_milk, red_pct = predict_milk(result["breed"], temp)
    result.update({"milk_prediction":milk,"milk_base":base_milk,"milk_reduction_pct":red_pct})

    if db is not None:
        try:
            rec = {k:v for k,v in result.items() if k != "success"}
            rec["username"] = username
            history.insert_one(rec)
        except Exception as e:
            print(f"History error: {e}")

    return jsonify(result)

@app.route("/history/<username>")
def get_history(username):
    if db is None:
        return jsonify([])
    records = list(history.find({"username":username},{"_id":0}).sort("timestamp",-1).limit(20))
    return jsonify(records)

@app.route("/health-check")
def health_check():
    return jsonify({
        "status":   "running",
        "models":   {"breed":breed_model is not None,"health":health_model is not None,"bcs":bcs_model is not None},
        "database":  db is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
