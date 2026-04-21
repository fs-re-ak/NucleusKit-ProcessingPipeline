ModelV11 — Facial EMG Emotion Classifier
Release: ModelV11_2026-03-21_d  |  Date: 2026-03-21
RE-AK Trainers
==============================================================

WHAT'S NEW IN V11
-----------------
* Artefact-rejection gate: windows with AVG_RMS above the training-derived
  threshold are returned as Neutral (artefact) before any LDA inference.
  The threshold and calibration metadata are in classifier/artefact_config.json.

QUICK START
-----------
1. Install dependencies:
     pip install -r requirements.txt

2. Run the demo:
     cd demo && python run_demo.py

3. Real-time integration:

     import sys
     sys.path.insert(0, "interface")
     from realtime_classifier import StreamingEMGClassifier

     clf = StreamingEMGClassifier(clf_dir="classifier", window_sec=1.0)
     for raw_frame in your_eeg_source:   # (8,) array in μV, 250 Hz
         result = clf.push_sample(raw_frame)
         if result is not None:
             print(result)

CONTENTS
--------
  classifier/          Trained model artifacts (including artefact_config.json)
  interface/           Python integration modules
  demo/                Sample EEG data + standalone demo script
  docs/                Application note and API reference
  requirements.txt
  README.txt

NOTES
-----
* artefact_threshold is a runtime-configurable attribute:
    clf._clf.artefact_threshold = new_value
* See docs/application_note.html for full details and performance metrics.
