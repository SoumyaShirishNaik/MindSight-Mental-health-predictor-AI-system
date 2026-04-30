"""
MindSight – Mental Health Prediction API (Market-Ready)
New: /api/export, /api/weekly, /api/feedback, /api/compare, mood journal endpoint
"""

import os, json, time, sqlite3, hashlib, csv, io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predictor import get_predictor, LABELS, LABEL_COLORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

DB_PATH   = "data/predictions.db"
MODEL_DIR = "models/bert_mental_health"

# ── DB init ───────────────────────────────────────────────────────────────────
def init_db():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash   TEXT,
            text_len    INTEGER,
            prediction  TEXT,
            confidence  REAL,
            severity    TEXT,
            top3        TEXT,
            created_at  TEXT,
            session_id  TEXT,
            user_note   TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_id     INTEGER,
            actual_label TEXT,
            rating      INTEGER,
            comment     TEXT,
            created_at  TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mood_score  INTEGER,
            mood_label  TEXT,
            note        TEXT,
            created_at  TEXT
        )""")
    conn.commit(); conn.close()

def save_prediction(result: Dict[str, Any], session_id="", note=""):
    try:
        conn = sqlite3.connect(DB_PATH)
        text_hash = hashlib.md5(result.get("text","").encode()).hexdigest()
        cur = conn.execute(
            "INSERT INTO predictions (text_hash,text_len,prediction,confidence,severity,top3,created_at,session_id,user_note) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (text_hash, len(result.get("text","")), result.get("prediction"),
             result.get("confidence"), result.get("severity"),
             json.dumps(result.get("top3",[])), datetime.now().isoformat(),
             session_id, note))
        pred_id = cur.lastrowid
        conn.commit(); conn.close()
        return pred_id
    except Exception as e:
        print(f"DB error: {e}"); return None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health")
def health():
    predictor = get_predictor(MODEL_DIR)
    return jsonify({"status":"ok","model_ready":predictor.is_ready(),
                    "timestamp":datetime.now().isoformat()})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error":"Missing 'text' field"}), 400
    text = data["text"].strip()
    if len(text) < 5:
        return jsonify({"error":"Text too short"}), 400

    predictor = get_predictor(MODEL_DIR)
    if not predictor.is_ready():
        return jsonify({
            "demo":True,"prediction":"Depression","confidence":72.5,
            "severity":"Moderate","color":"#2196F3",
            "description":"Demo mode – train the model for real predictions.",
            "recommendations":["Run: python src/train_bert.py","Then restart: python app.py"],
            "top3":[{"label":"Depression","confidence":0.725,"color":"#2196F3"},
                    {"label":"Anxiety","confidence":0.155,"color":"#FF9800"},
                    {"label":"Stress","confidence":0.082,"color":"#FF5722"}],
            "distribution":{l:0.0 for l in LABELS.values()},
            "tokens":[],"token_scores":[]})

    t0 = time.time()
    result = predictor.predict(text)
    result["inference_time_ms"] = round((time.time()-t0)*1000,1)

    if "error" not in result:
        session_id = data.get("session_id","")
        note       = data.get("note","")
        pred_id    = save_prediction(result, session_id, note)
        result["pred_id"] = pred_id

    return jsonify(result)

@app.route("/api/batch", methods=["POST"])
def batch_predict():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error":"Missing 'texts' field"}), 400
    texts = data["texts"]
    if not isinstance(texts,list) or len(texts)==0:
        return jsonify({"error":"texts must be non-empty list"}), 400
    if len(texts)>20:
        return jsonify({"error":"Max 20 texts per batch"}), 400

    predictor = get_predictor(MODEL_DIR)
    if not predictor.is_ready():
        return jsonify({"error":"Model not trained yet"}), 503

    results = predictor.batch_predict(texts)
    for r in results:
        if "error" not in r:
            save_prediction(r)
    return jsonify({"results":results,"count":len(results)})

@app.route("/api/stats")
def stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        rows   = conn.execute("SELECT prediction,COUNT(*) FROM predictions GROUP BY prediction").fetchall()
        total  = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        recent = conn.execute(
            "SELECT id,prediction,confidence,severity,created_at FROM predictions ORDER BY id DESC LIMIT 15"
        ).fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
        conn.close()
        return jsonify({
            "total_predictions": total,
            "avg_confidence": round(avg_conf,2),
            "label_distribution": {r[0]:r[1] for r in rows},
            "recent": [{"id":r[0],"prediction":r[1],"confidence":r[2],
                        "severity":r[3],"at":r[4]} for r in recent],
        })
    except Exception as e:
        return jsonify({"error":str(e),"total_predictions":0})

@app.route("/api/weekly")
def weekly():
    """Return day-by-day prediction counts + category breakdown for last 7 days."""
    try:
        conn = sqlite3.connect(DB_PATH)
        days = []
        for i in range(6,-1,-1):
            d = (datetime.now()-timedelta(days=i)).strftime("%Y-%m-%d")
            label = (datetime.now()-timedelta(days=i)).strftime("%a")
            rows = conn.execute(
                "SELECT prediction,COUNT(*) FROM predictions WHERE created_at LIKE ? GROUP BY prediction",
                (d+"%",)).fetchall()
            total_day = sum(r[1] for r in rows)
            by_cat    = {r[0]:r[1] for r in rows}
            days.append({"date":d,"day":label,"total":total_day,"by_category":by_cat})
        conn.close()
        return jsonify({"days":days})
    except Exception as e:
        return jsonify({"error":str(e)})

@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Accept user correction / rating for a prediction."""
    data = request.get_json()
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO feedback (pred_id,actual_label,rating,comment,created_at) VALUES (?,?,?,?,?)",
            (data.get("pred_id"), data.get("actual_label",""),
             data.get("rating",3), data.get("comment",""),
             datetime.now().isoformat()))
        conn.commit(); conn.close()
        return jsonify({"status":"ok","message":"Feedback saved. Thank you!"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/journal", methods=["GET","POST"])
def journal():
    """Mood journal — GET returns last 14 entries, POST adds one."""
    if request.method=="POST":
        data = request.get_json()
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO journal (mood_score,mood_label,note,created_at) VALUES (?,?,?,?)",
                (data.get("mood_score",5), data.get("mood_label",""),
                 data.get("note",""), datetime.now().isoformat()))
            conn.commit(); conn.close()
            return jsonify({"status":"ok"})
        except Exception as e:
            return jsonify({"error":str(e)}), 500
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute(
                "SELECT id,mood_score,mood_label,note,created_at FROM journal ORDER BY id DESC LIMIT 14"
            ).fetchall()
            conn.close()
            return jsonify({"entries":[{"id":r[0],"mood_score":r[1],"mood_label":r[2],
                                        "note":r[3],"at":r[4]} for r in rows]})
        except Exception as e:
            return jsonify({"error":str(e)})

@app.route("/api/export")
def export_csv():
    """Export all predictions as CSV download."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT id,prediction,confidence,severity,text_len,created_at FROM predictions ORDER BY id DESC"
        ).fetchall()
        conn.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID","Prediction","Confidence(%)","Severity","Text Length","Created At"])
        writer.writerows(rows)
        return Response(output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition":"attachment;filename=mindsight_predictions.csv"})
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/compare", methods=["POST"])
def compare():
    """Compare up to 3 texts side-by-side."""
    data = request.get_json()
    texts = data.get("texts",[])
    if not texts or len(texts)>3:
        return jsonify({"error":"Send 1–3 texts"}), 400
    predictor = get_predictor(MODEL_DIR)
    if not predictor.is_ready():
        return jsonify({"error":"Model not ready"}), 503
    results = [predictor.predict(t) for t in texts]
    return jsonify({"results":results})

@app.route("/api/model_info")
def model_info():
    p = Path(MODEL_DIR)/"results.json"
    if p.exists():
        with open(p) as f: return jsonify(json.load(f))
    return jsonify({"error":"Model not trained yet"})

if __name__ == "__main__":
    init_db()
    print("\n"+"="*55)
    print("  🧠 MindSight — Mental Health Prediction System")
    print("  URL: http://localhost:5000")
    print("="*55+"\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
