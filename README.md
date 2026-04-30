# 🧠 MindSight – Mental Health Prediction with BERT

An end-to-end AI/ML system for mental health detection using fine-tuned BERT,
with a stunning dark-theme web interface, explainability (gradient-based token
importance), analytics dashboard, and REST API.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Model** | `bert-base-uncased` fine-tuned for 7-class classification |
| **Categories** | Normal · Depression · Anxiety · Bipolar · PTSD · OCD · Stress |
| **Explainability** | Gradient-based token importance highlighting |
| **Confidence** | Animated ring + severity scoring (Low / Moderate / High) |
| **Dashboard** | Live analytics, distribution charts, prediction history |
| **API** | REST endpoints: `/api/predict`, `/api/batch`, `/api/stats` |
| **UI** | Beautiful dark interface with DM Serif typography |
| **Storage** | SQLite prediction history (no external DB needed) |

---

## 🗂️ Project Structure

```
mental_health_bert/
├── app.py                    ← Flask web server (entry point)
├── requirements.txt          ← Python dependencies
├── src/
│   ├── data_generator.py     ← Dataset creation & preprocessing
│   ├── train_bert.py         ← BERT fine-tuning pipeline
│   └── predictor.py          ← Inference engine + explainability
├── templates/
│   └── index.html            ← Main web UI
├── static/
│   ├── css/style.css         ← Dark theme styles
│   └── js/app.js             ← Frontend logic
├── scripts/
│   └── quick_setup.py        ← Fast demo setup (no training needed)
├── notebooks/
│   └── analysis.ipynb        ← EDA and result visualization
├── models/                   ← Saved model (after training)
└── data/                     ← Datasets (generated automatically)
```

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- 4 GB RAM minimum (8 GB for training)
- Internet connection (first run downloads BERT ~440 MB)

### Step 1 – Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Quick Demo Setup (UI testing, no training)

```bash
python scripts/quick_setup.py
```

This downloads BERT and prepares the demo in ~2 minutes.

### Step 4 – Start the server

```bash
python app.py
```

### Step 5 – Open browser

```
http://localhost:5000
```

---

## 🏋️ Full Training (for accurate predictions)

Training takes 15-45 minutes on CPU, 3-8 minutes on GPU.

```bash
# Step 1: Generate dataset
python src/data_generator.py

# Step 2: Fine-tune BERT
python src/train_bert.py

# Step 3: Start server
python app.py
```

### GPU Training (recommended)

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🌐 API Reference

### `POST /api/predict`
```json
// Request
{ "text": "I feel overwhelmed and anxious all the time." }

// Response
{
  "prediction": "Anxiety",
  "confidence": 78.4,
  "severity": "Moderate",
  "color": "#FF9800",
  "description": "...",
  "recommendations": ["..."],
  "top3": [...],
  "distribution": {...},
  "tokens": [...],
  "token_scores": [...]
}
```

### `POST /api/batch`
```json
{ "texts": ["text1", "text2", "..."] }   // max 20
```

### `GET /api/stats`
Returns dashboard analytics and prediction history.

### `GET /api/health`
Returns server and model status.

### `GET /api/model_info`
Returns training metrics (accuracy, F1, etc.).

---

## ⚙️ Configuration

Edit `src/train_bert.py` → `CONFIG` dict:

```python
CONFIG = {
    "model_name":    "bert-base-uncased",  # or bert-large-uncased
    "max_length":    256,                  # token sequence length
    "batch_size":    16,                   # reduce if OOM
    "num_epochs":    5,                    # increase for better accuracy
    "learning_rate": 2e-5,                 # standard for BERT fine-tuning
    "dropout":       0.3,                  # regularization
}
```

---

## 📊 Expected Performance

| Metric | Expected Range |
|---|---|
| Test Accuracy | 82–92% |
| Weighted F1 | 0.83–0.93 |
| Inference Time | ~50–200ms per text |

*Results vary depending on hardware and training epochs.*

---

## 🔬 Model Architecture

```
Input Text
    ↓
BERT Tokenizer (WordPiece, max 256 tokens)
    ↓
bert-base-uncased
  - 12 Transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - 110M parameters
    ↓
[CLS] token representation (768-dim)
    ↓
Dropout (p=0.3)
    ↓
Linear Classifier (768 → 7)
    ↓
Softmax → 7-class probabilities
```

---

## ⚠️ Disclaimer

**MindSight is for educational and research purposes only.**

- It is NOT a medical diagnostic tool
- Results should NOT replace professional mental health assessment
- If you or someone you know is in crisis:
  - 🇺🇸 **988** – Suicide & Crisis Lifeline (US)
  - 🌍 **findahelpline.com** – International resources

---

## 📦 Dependencies

- `torch` – Deep learning framework
- `transformers` – HuggingFace BERT implementation
- `scikit-learn` – Metrics and data splitting
- `Flask` – Web framework
- `pandas / numpy` – Data processing
- `matplotlib / seaborn` – Visualization

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `CUDA out of memory` | Reduce `batch_size` to 8 or 4 |
| `Connection refused` | Make sure `python app.py` is running |
| Model not found | Run `python scripts/quick_setup.py` |
| Slow on CPU | Normal – BERT is large. Use GPU for speed. |

---

*Built with ❤️ using BERT + Flask + Vanilla JS*
