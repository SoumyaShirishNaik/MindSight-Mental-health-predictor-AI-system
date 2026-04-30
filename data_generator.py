"""
Mental Health Dataset Generator & Preprocessor
Generates a synthetic dataset for training + preprocessing pipeline
"""

import pandas as pd
import numpy as np
import json
import os
import re
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)

# ── Label schema ──────────────────────────────────────────────────────────────
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
    "Normal": "#4CAF50",
    "Depression": "#2196F3",
    "Anxiety": "#FF9800",
    "Bipolar Disorder": "#9C27B0",
    "PTSD": "#F44336",
    "OCD": "#00BCD4",
    "Stress": "#FF5722",
}

# ── Sample texts per class ────────────────────────────────────────────────────
SAMPLES = {
    0: [
        "I had a great day today. Feeling energetic and positive about life.",
        "Just finished a wonderful workout and feel amazing. Life is good.",
        "Spent quality time with my family. Feeling grateful and content.",
        "Work is going well and I have good relationships with my colleagues.",
        "I enjoy my hobbies and feel balanced in my daily routine.",
        "Had a productive meeting and accomplished all my goals for the day.",
        "Feeling optimistic about the future and excited about new opportunities.",
        "I slept well and woke up refreshed, ready for the day.",
        "Enjoyed a nice meal with friends and laughed a lot.",
        "Everything seems to be falling into place. Feeling at peace.",
    ],
    1: [
        "I can't get out of bed. Everything feels pointless and empty.",
        "I haven't felt joy in weeks. Nothing seems to matter anymore.",
        "I'm constantly exhausted, even after sleeping for 12 hours.",
        "I feel like a burden to everyone around me. They'd be better off without me.",
        "Lost interest in things I used to love. Can't remember the last time I smiled.",
        "The darkness feels overwhelming. I don't see a way out.",
        "I've been crying for no reason. I feel so hollow inside.",
        "Stopped eating properly. Just don't have the appetite or energy.",
        "I isolate myself because I don't want to bring others down with my sadness.",
        "Woke up dreading another day. The thought of facing people exhausts me.",
    ],
    2: [
        "My heart races constantly and I can't control the worrying thoughts.",
        "I'm terrified of everything. Simple tasks feel impossible.",
        "Panic attacks are becoming more frequent. I never know when one will hit.",
        "Can't stop worrying about what might go wrong. My mind won't quiet down.",
        "I avoid social situations because I'm afraid of embarrassing myself.",
        "The anxiety is so overwhelming I can't breathe sometimes.",
        "I've been having trouble sleeping because of constant fear and worry.",
        "My thoughts spiral out of control and I can't make them stop.",
        "I feel like something terrible is always about to happen.",
        "Even small decisions fill me with dread and paralyze me.",
    ],
    3: [
        "Yesterday I felt invincible and spent thousands. Today I can't get out of bed.",
        "My moods swing so wildly. I don't even recognize myself sometimes.",
        "During my highs I need no sleep and feel like I can conquer the world.",
        "The crashes after the manic episodes are devastating and humiliating.",
        "I've ruined relationships because of my unpredictable behavior.",
        "The euphoria feels amazing until it turns into a hurricane of bad decisions.",
        "My energy levels are extreme – either completely depleted or overflowing.",
        "I start dozens of projects but finish none. Then comes the crushing low.",
        "People think I'm dramatic but this cycling between extremes is exhausting.",
        "When the mania hits I don't sleep for days and my thoughts race nonstop.",
    ],
    4: [
        "The nightmares keep coming back. I relive the trauma every night.",
        "Loud noises make me freeze. I'm always on edge, waiting for danger.",
        "I avoid anything that reminds me of what happened. It's destroying my life.",
        "The flashbacks are so vivid I lose track of where and when I am.",
        "I feel emotionally numb, disconnected from everything and everyone.",
        "Trust is impossible after what I went through. I expect to be hurt.",
        "My body reacts to triggers before my mind can catch up.",
        "I can't talk about it without breaking down. The memories are too raw.",
        "I isolate myself because being around people feels unsafe.",
        "The hypervigilance is exhausting. I can't relax or feel safe anywhere.",
    ],
    5: [
        "I have to check the locks 20 times before I can sleep.",
        "If I don't perform my rituals, I'm convinced something terrible will happen.",
        "My intrusive thoughts are shameful and I can't make them stop.",
        "I spend hours on compulsions that I know are irrational but can't resist.",
        "Everything must be perfectly symmetrical or I can't function.",
        "The contamination fears make it impossible to touch everyday surfaces.",
        "I've been late to work again because my morning rituals took too long.",
        "I know the obsessions are irrational but the anxiety forces me to comply.",
        "Counting and arranging things consumes hours of my day.",
        "I'm exhausted from fighting thoughts I didn't choose to have.",
    ],
    6: [
        "Deadlines are piling up and I can't cope. I'm running on empty.",
        "Between work, family, and bills, I feel like I'm drowning.",
        "I've been snapping at people I care about because I'm so overwhelmed.",
        "Constant pressure is giving me headaches and tension in my shoulders.",
        "I don't have time to breathe. Everything needs attention at once.",
        "I feel like I'm failing at everything simultaneously.",
        "The workload is unsustainable but I'm afraid to say no.",
        "My body is showing signs of burnout – fatigue, irritability, brain fog.",
        "I can't switch off even when I have a rare moment of downtime.",
        "The to-do list keeps growing and I can't see the end.",
    ],
}


def augment_text(text: str) -> str:
    """Simple augmentation by paraphrasing sentence structure."""
    augmentations = [
        lambda t: t,
        lambda t: t.replace("I ", "I've been feeling like ") if t.startswith("I ") else t,
        lambda t: t + " It's been really hard.",
        lambda t: "Lately, " + t[0].lower() + t[1:],
        lambda t: t + " I don't know what to do.",
    ]
    return random.choice(augmentations)(text)


def generate_dataset(n_per_class: int = 500) -> pd.DataFrame:
    """Generate augmented dataset with n_per_class samples per label."""
    rows = []
    for label_id, texts in SAMPLES.items():
        base = texts
        while len(base) < n_per_class:
            base = base + [augment_text(t) for t in texts]
        for text in base[:n_per_class]:
            rows.append({
                "text": text,
                "label": label_id,
                "label_name": LABELS[label_id],
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"()-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_and_split(df: pd.DataFrame):
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    return train_df, val_df, test_df


if __name__ == "__main__":
    out = Path("data")
    out.mkdir(exist_ok=True)

    print("Generating dataset...")
    df = generate_dataset(n_per_class=500)
    print(f"Total samples: {len(df)}")
    print(df["label_name"].value_counts())

    train_df, val_df, test_df = preprocess_and_split(df)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    df.to_csv(out / "full_dataset.csv", index=False)

    meta = {"labels": LABELS, "label_colors": LABEL_COLORS, "num_classes": len(LABELS)}
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved splits → data/  train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
