"""
ModelV12 Demo — read a raw EEG CSV and emit per-window emotion probabilities.

Input:  eeg_sample.csv
Output: emotions.csv  (time_sec, dominant_emotion, confidence, artefact, <proba columns>)

Usage
-----
    python run_demo.py
    python run_demo.py --input eeg_sample.csv --output emotions.csv
"""
from __future__ import annotations
import argparse, csv, os, sys

_DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
_RELEASE_DIR = os.path.dirname(_DEMO_DIR)
sys.path.insert(0, os.path.join(_RELEASE_DIR, "interface"))

from realtime_classifier import StreamingEMGClassifier  # noqa: E402


def run(input_path, output_path, window_sec=1.0, step_sec=0.5, sampling_rate=250):
    clf_dir = os.path.join(_RELEASE_DIR, "classifier")
    clf = StreamingEMGClassifier(clf_dir=clf_dir, window_sec=window_sec,
                                 step_sec=step_sec, sampling_rate=sampling_rate)
    results = []
    samples_fed = windows_classified = 0

    print(f"Reading: {input_path}")
    with open(input_path, newline="", encoding="utf-8") as f:
        for raw_line in f:
            parts = raw_line.strip().split(",")
            try:
                sample = [float(v) for v in parts[1:9]]
                if len(sample) < 8:
                    continue
            except ValueError:
                continue
            samples_fed += 1
            proba = clf.push_sample(sample)
            if proba is not None:
                windows_classified += 1
                time_sec = round(samples_fed / sampling_rate, 4)
                dominant = max(proba, key=proba.get)
                confidence = round(proba[dominant], 4)
                row = {"time_sec": time_sec, "dominant_emotion": dominant,
                       "confidence": confidence}
                row.update({k: round(v, 4) for k, v in proba.items()})
                results.append(row)

    if not results:
        print("No windows produced.")
        return

    emotion_cols = [k for k in results[0] if k not in {"time_sec", "dominant_emotion", "confidence"}]
    fieldnames = ["time_sec", "dominant_emotion", "confidence"] + sorted(emotion_cols)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Done. {samples_fed:,} samples | {windows_classified:,} windows → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=os.path.join(os.path.dirname(__file__), "eeg_sample.csv"))
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "emotions.csv"))
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--step",   type=float, default=0.5)
    ap.add_argument("--rate",   type=int,   default=250)
    args = ap.parse_args()
    run(args.input, args.output, args.window, args.step, args.rate)
