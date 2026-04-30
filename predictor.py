"""
Mental Health Prediction Inference Engine
Supports: BERT prediction + confidence scores + basic explainability
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import BertTokenizerFast, BertForSequenceClassification


# ── Label metadata ─────────────────────────────────────────────────────────────
LABELS = {
    0: "Normal",
    1: "Depression",
    2: "Anxiety",
    3: "Bipolar Disorder",
    4: "PTSD",
    5: "OCD",
    6: "Stress",
}

LABEL_COLORS = {
    "Normal":           "#4CAF50",
    "Depression":       "#2196F3",
    "Anxiety":          "#FF9800",
    "Bipolar Disorder": "#9C27B0",
    "PTSD":             "#F44336",
    "OCD":              "#00BCD4",
    "Stress":           "#FF5722",
}

LABEL_DESCRIPTIONS = {
    "Normal": "Your text indicates a healthy mental state with positive indicators.",
    "Depression": "Your text shows signs associated with depressive symptoms such as low mood, loss of interest, or feelings of hopelessness.",
    "Anxiety": "Your text reflects anxiety-related patterns including worry, fear, and restlessness.",
    "Bipolar Disorder": "Your text contains patterns that may be associated with mood cycling typical of bipolar disorder.",
    "PTSD": "Your text shows indicators linked to trauma responses including hypervigilance and avoidance.",
    "OCD": "Your text suggests patterns related to obsessive thinking or compulsive behaviors.",
    "Stress": "Your text indicates high stress levels and overwhelm.",
}

RECOMMENDATIONS = {
    "Normal": [
        "Continue your healthy habits and self-care routines.",
        "Stay connected with friends and family.",
        "Maintain regular sleep, exercise, and nutrition.",
    ],
    "Depression": [
        "Consider speaking with a licensed therapist or counselor.",
        "Reach out to a trusted person in your life.",
        "Crisis support: National Suicide Prevention Lifeline 988 (US).",
        "Small daily activities like walks can help.",
    ],
    "Anxiety": [
        "Practice deep breathing or grounding exercises (5-4-3-2-1 technique).",
        "Consider Cognitive Behavioral Therapy (CBT).",
        "Limit caffeine and screen time before bed.",
        "Talk to a mental health professional.",
    ],
    "Bipolar Disorder": [
        "A psychiatrist can evaluate mood stabilization options.",
        "Keep a mood journal to track patterns.",
        "Maintain a consistent sleep schedule.",
        "Avoid alcohol and recreational drugs.",
    ],
    "PTSD": [
        "Trauma-focused therapy (EMDR, CPT) has strong evidence.",
        "Contact a PTSD helpline or veteran support service.",
        "Practice safety grounding exercises.",
        "Give yourself compassion – healing is non-linear.",
    ],
    "OCD": [
        "Exposure and Response Prevention (ERP) is highly effective.",
        "Speak with a psychiatrist about treatment options.",
        "Join an OCD support community (IOCDF).",
        "Avoid reassurance-seeking, which can reinforce OCD.",
    ],
    "Stress": [
        "Identify and reduce stressors where possible.",
        "Try mindfulness or meditation apps.",
        "Set boundaries and practice saying no.",
        "Ensure regular breaks and recovery time.",
    ],
}

SEVERITY_THRESHOLDS = {
    "Low": 0.40,
    "Moderate": 0.65,
    "High": 0.85,
}


class MentalHealthPredictor:
    """Loads the fine-tuned BERT model and runs inference."""

    def __init__(self, model_dir: str = "models/bert_mental_health"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.model: Optional[BertForSequenceClassification] = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"Loading model from {self.model_dir} on {self.device}...")
        self.tokenizer = BertTokenizerFast.from_pretrained(str(self.model_dir))
        self.model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print("Model loaded successfully.")

    def is_ready(self) -> bool:
        return self.model_dir.exists() and (self.model_dir / "config.json").exists()

    def predict(self, text: str, max_length: int = 256) -> Dict:
        """Run full prediction pipeline for a single text."""
        if not self._loaded:
            self.load()

        if not text or len(text.strip()) < 5:
            return {"error": "Text too short. Please enter at least a sentence."}

        enc = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = LABELS[pred_id]
        confidence = float(probs[pred_id])

        # Severity
        severity = "Low"
        for level, thresh in sorted(SEVERITY_THRESHOLDS.items(), key=lambda x: x[1]):
            if confidence >= thresh:
                severity = level

        # Top-3 predictions
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"label": LABELS[i], "confidence": float(probs[i]), "color": LABEL_COLORS[LABELS[i]]}
            for i in top3_idx
        ]

        # Full distribution
        distribution = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

        # Attention-based token importance (simplified)
        tokens, token_scores = self._get_token_importance(text, max_length)

        return {
            "text": text,
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "severity": severity,
            "color": LABEL_COLORS[pred_label],
            "description": LABEL_DESCRIPTIONS[pred_label],
            "recommendations": RECOMMENDATIONS[pred_label],
            "top3": top3,
            "distribution": distribution,
            "tokens": tokens,
            "token_scores": token_scores,
        }

    def _get_token_importance(self, text: str, max_length: int) -> Tuple[List[str], List[float]]:
        """Return tokens with simplified gradient-based importance scores."""
        try:
            enc = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            # Enable gradients for input embeddings
            embeddings = self.model.bert.embeddings.word_embeddings(enc["input_ids"])
            embeddings.retain_grad()

            outputs_with_grad = self.model(
                inputs_embeds=embeddings,
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
            pred = outputs_with_grad.logits.argmax(dim=-1)
            outputs_with_grad.logits[0, pred].backward()

            grads = embeddings.grad.abs().mean(dim=-1).squeeze().cpu().numpy()
            # Normalize
            grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-9)

            token_ids = enc["input_ids"].squeeze().cpu().numpy()
            tokens_raw = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Filter special tokens
            filtered = [
                (t, float(s))
                for t, s in zip(tokens_raw, grads)
                if t not in ("[CLS]", "[SEP]", "[PAD]")
            ][:20]

            tokens = [t for t, _ in filtered]
            scores = [s for _, s in filtered]
            return tokens, scores
        except Exception:
            return [], []

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]


# ── Singleton predictor ────────────────────────────────────────────────────────
_predictor: Optional[MentalHealthPredictor] = None


def get_predictor(model_dir: str = "models/bert_mental_health") -> MentalHealthPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MentalHealthPredictor(model_dir)
    return _predictor


if __name__ == "__main__":
    predictor = get_predictor()
    if predictor.is_ready():
        predictor.load()
        sample = "I have been feeling extremely sad and hopeless lately, nothing brings me joy anymore."
        result = predictor.predict(sample)
        print(json.dumps({k: v for k, v in result.items() if k not in ("tokens", "token_scores")}, indent=2))
    else:
        print("Model not found. Run training first: python src/train_bert.py")
