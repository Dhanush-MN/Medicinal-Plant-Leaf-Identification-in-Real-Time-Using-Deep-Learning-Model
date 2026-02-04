"""
Flask app for MedicinalLeaf
- register / login (passwords hashed)python -m pip install --upgrade pip setuptools wheel

- dashboard endpoints for upload and webcam base64 prediction
- history saved to DB (username, filename, label, confidence, timestamp)
- optionally preprocess uploaded image with rembg/OpenCV before prediction
"""

import os
import io
import sqlite3
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

# local model loader
from model_loader import predict_image, preprocess_image_if_available

# --- Config ---
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "leaf_info.db"
UPLOAD_FOLDER = Path(__file__).resolve().parent / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# --- Flask app setup ---
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("MEDICINALLEAF_SECRET", "replace_this_with_secure_random")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# --- Simple User class for Flask-Login ---
class User(UserMixin):
    def __init__(self, uid, username):
        self.id = str(uid)
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    # We'll simply return a User if username exists in session or DB.
    # Flask-Login requires this function; our session stores username after login.
    username = session.get("username")
    return User(user_id, username) if username else None

# --- DB helper ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Routes ---
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password")
        if not username or not password:
            return "Username and password required", 400
        hashed = generate_password_hash(password)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)")
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return "Username already exists", 400
        conn.close()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row["password"], password):
            user = User(row["id"], username)
            login_user(user)
            session["username"] = username
            return redirect(url_for("dashboard"))
        return "Invalid credentials", 401
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)

@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, filename TEXT, label TEXT, confidence REAL, created_at TEXT)")
    cur.execute("SELECT filename, label, confidence, created_at FROM history WHERE username=? ORDER BY created_at DESC", (current_user.username,))
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", rows=rows)

# allow serving user-uploaded images
@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# helper to validate extension
def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXT

# Upload file endpoint
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_name = f"{current_user.username}_{timestamp}_{filename}"
    save_path = UPLOAD_FOLDER / save_name
    file.save(save_path)

    # Optional preprocessing step (segmentation) if available
    try:
        preprocess_image_if_available(str(save_path))
    except Exception as e:
        # don't fail if preprocessing not available or fails — continue to prediction
        app.logger.warning("Preprocess failed: %s", e)

    # Predict
    try:
        label, confidence = predict_image(str(save_path))
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    # retrieve info from DB
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT scientific_name, benefits FROM LeafInfo WHERE name=?", (label,))
    info = cur.fetchone()
    conn.close()
    if info:
        scientific_name = info["scientific_name"]
        benefits = info["benefits"]
    else:
        scientific_name = "N/A"
        benefits = "N/A"

    # save to history
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, filename TEXT, label TEXT, confidence REAL, created_at TEXT)")
        cur.execute("INSERT INTO history (username, filename, label, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                    (current_user.username, save_name, label, float(confidence), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    except Exception as e:
        app.logger.warning("Failed to save history: %s", e)

    return jsonify({
        "label": label,
        "confidence": float(confidence),
        "info": {"scientific_name": scientific_name, "benefits": benefits}
    })

# Webcam base64 predict endpoint
@app.route("/predict_base64", methods=["POST"])
@login_required
def predict_base64():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    img_b64 = data["image"]
    if "," in img_b64:
        header, img_b64 = img_b64.split(",", 1)
    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    save_name = f"{current_user.username}_cam_{timestamp}.jpg"
    save_path = UPLOAD_FOLDER / save_name
    img.save(save_path)

    # Optional preprocessing
    try:
        preprocess_image_if_available(str(save_path))
    except Exception as e:
        app.logger.warning("Preprocess failed: %s", e)

    try:
        label, confidence = predict_image(str(save_path))
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

    # fetch leaf info
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT scientific_name, benefits FROM LeafInfo WHERE name=?", (label,))
    info = cur.fetchone()
    conn.close()
    if info:
        scientific_name = info["scientific_name"]
        benefits = info["benefits"]
    else:
        scientific_name = "N/A"
        benefits = "N/A"

    # save history
    try:
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute("INSERT INTO history (username, filename, label, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                    (current_user.username, save_name, label, float(confidence), datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
    except Exception as e:
        app.logger.warning("Failed to save history: %s", e)

    return jsonify({"label": label, "confidence": float(confidence), "info": {"scientific_name": scientific_name, "benefits": benefits}})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
