"""
Quick Setup Script – creates a demo model for testing UI without full training.
Run this if you want to test the interface immediately.
For production accuracy, run: python src/train_bert.py
"""

import os
import json
import torch
import numpy as np
from pathlib import Path

def create_demo_model():
    """Download BERT and save it with random classification head for demo."""
    print("\n" + "="*55)
    print("  MindSight – Quick Demo Setup")
    print("  This creates a DEMO model for UI testing only.")
    print("  For accurate predictions, run: python src/train_bert.py")
    print("="*55 + "\n")

    save_dir = Path("models/bert_mental_health")
    save_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Step 1: Generate dataset
    print("[1/3] Generating dataset...")
    os.system("python src/data_generator.py")

    # Step 2: Load BERT and save with classification head
    print("\n[2/3] Downloading & saving BERT model...")
    try:
        from transformers import BertTokenizerFast, BertForSequenceClassification
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=7
        )
        tokenizer.save_pretrained(str(save_dir))
        model.save_pretrained(str(save_dir))
        print(f"   Model saved to {save_dir}/")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure transformers is installed: pip install transformers")
        return False

    # Step 3: Save mock results
    print("\n[3/3] Saving demo results metadata...")
    results = {
        "config": {"model_name": "bert-base-uncased", "num_epochs": 0},
        "best_epoch": 0,
        "best_val_f1": 0.0,
        "test_accuracy": 0.0,
        "test_f1_weighted": 0.0,
        "label_names": ["Normal", "Depression", "Anxiety", "Bipolar Disorder", "PTSD", "OCD", "Stress"],
        "note": "DEMO MODEL – run train_bert.py for real accuracy",
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  ✅ Demo setup complete!")
    print("  Start server: python app.py")
    print("  Open browser: http://localhost:5000")
    print("\n  For real accuracy, run: python src/train_bert.py")
    print("="*55 + "\n")
    return True


if __name__ == "__main__":
    create_demo_model()
