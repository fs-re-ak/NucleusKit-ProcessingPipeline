
import os
import json
import random
import time


def createFrame():
    frame = {
        "status": {
            "is_recording": True,
            "calibration_active": True,
            "headset_connected": True,
            "electrodes_quality": {
                  "tp9": 1.0,
                  "af7": 1.0,
                  "af8": 1.0,
                  "tp10": 1.0
            }
        },
        "metrics": {
            "cognitive": {
                "alpha": 0.049879,
                "engagement": 0.049879,
                "engagement_calib": 0.049879
            },
            "emotions": {
                "anger": 0.009,
                "contempt": 0.008,
                "disgust": 0.005,
                "fear": 0.009,
                "happiness": 0.006,
                "neutral": 0.944,
                "sadness": 0.008,
                "surprise": 0.01
            }
        },
        "timestamp": 1727906403.9483588,
        "type": "Frame",
        "device_id": "vizia-nookii-1",
        "frame_version": 2
    }

    return frame


def assignRandomValues(frame):
    
    
    electrodes = ['tp9', 'af7', 'af8', 'tp10']
    emotions = ['neutral', 'happiness', 'anger', 'contempt', 'surprise', 'disgust', 'sadness', 'fear']
    
    
    # Random booleans
    frame["status"] = {}
    frame["status"]["is_recording"] = random.choice([True, False])
    frame["status"]["calibration_active"] = random.choice([True, False])
    frame["status"]["headset_connected"] = random.choice([True, False])

    # Random electrode qualities (0.0 to 1.0)
    
    frame["status"]["electrodes_quality"] = {}
    for electrode in electrodes:
        frame["status"]["electrodes_quality"][electrode] = round(random.uniform(0.0, 1.0), 3)

    # Random cognitive metrics
    frame["metrics"] = {}
    frame["metrics"]["cognitive"] = {}
    frame["metrics"]["cognitive"]["alpha"] = round(random.uniform(0.0, 1.0), 6)
    frame["metrics"]["cognitive"]["engagement"] = round(random.uniform(0.0, 1.0), 6)
    frame["metrics"]["cognitive"]["engagement_calib"] = round(random.uniform(0.0, 1.0), 6)


    frame["metrics"]["emotions"] = {}
    # Random emotion metrics (sum doesn't have to be 1 unless you want it to)
    for emotion in emotions:
        frame["metrics"]["emotions"][emotion] = round(random.uniform(0.0, 1.0), 3)

    # Update timestamp
    frame["timestamp"] = time.time()


def showFrame(frame):
    os.system('cls' if os.name == 'nt' else 'clear')  # Clears the terminal
    print(json.dumps(frame, indent=4))
    pass







