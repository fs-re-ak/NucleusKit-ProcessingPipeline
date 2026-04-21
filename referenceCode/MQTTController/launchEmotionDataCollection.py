"""
File: launchTrainingExperience.py
Author: Fred Simard @ RE-AK Tech
Date: April 2024
Description: This shows how to read data coming from a Nucleus-Hermes and how to send it control signals.

* Stream refer to the data coming in. The underlying and recommended protocol is OSC. As long as both
devices are on the same network, this should work off the bat.

Emotional and cognitive indexes are assembled in a dictionnary, the frame. The transmission is asynchrone.
The Frame Adapter sends an updated version of the frame 2 times per second.

If the device is inactive, you will either read 0s, if no valid data was ever received or the last valid value
will be repeated.

Messages need to be filtered by device.
(Multidevice can be achieved easily by modifying the code, but is not supported officially, yet)

If you are encountering problems, you should:
    - validate that both devices are on the same network
    - validate that UDP port 10337 is open

* The Control interface use MQTT. You should install a MQTT broker, we recommend Mosquitto. The control interface
doesn't support individual device control yet. You will control all devices on the network at the same time.

The Controller use a brokerPing to send the IP address of the broker to all devices on the network. This is done
using OSC port 10339. Do not interfer with this process or the device won't know where the broker is.

Commands
    - sendStartRecording/sendStopRecording are used to control data recording. The proper operation can be
      confirmed by looking at the green LED on the device.
    - sendTag is used to record events and meta information alongside the data. Those will be found on the dashboard.

"""
from time import sleep
from deviceController.DeviceControllerManager import DeviceControllerManager
from streamInterfaces.FrameManager import FrameManager
from streamInterfaces.adapters.frameUtils import showFrame

class Configuration():
    INPUT_FRAME_ACTIVE = True
    DEVICE_CONTROLLER_ACTIVE = True
    #DEVICE_ID = "/hermes-hacklab-6*" # /<your device id>*
    DEVICE_ID = "/*" # /<your device id>*
    OPERATION_MODE = "CONTROL_DEMO"#"CONTROL_DEMO" or "LIVE_DISPLAY" # 
    SHOW_OSC = False 

def frameHandler(frame):
    """
    Call back used for demo, it simply shows the content of the frame on screen
    :param frame: a dictionnary containing emotional and cognitive metrics
    """
    #showFrame(frame)
    pass
    
# instantiate the FrameManager and attach the callback.
inputFrames = FrameManager('OSC', deviceFilter=Configuration.DEVICE_ID, showOSC=Configuration.SHOW_OSC,stubbed=not Configuration.INPUT_FRAME_ACTIVE)

if not Configuration.SHOW_OSC:
    inputFrames.attachCallback(frameHandler)

# if active, instantiate the controller
if Configuration.OPERATION_MODE=="CONTROL_DEMO":
    deviceController = DeviceControllerManager('MQTT', stubbed=not Configuration.DEVICE_CONTROLLER_ACTIVE)

# basic stream demo, simply shows data on screen
if Configuration.OPERATION_MODE == "LIVE_DISPLAY":

    # runs undefinitely
    while True:
        sleep(1)

# basic control demo, will record a short session and send some tags to show how it's done.
elif Configuration.OPERATION_MODE == "CONTROL_DEMO":


    import random
    from time import sleep

    emotions = [
        "Neutral", "Happy", "Anger", "Surprised",
        "Disgust", "Fear", "Sadness", "Contempt"
    ]

    num_blocks = 3
    trial_counter = 1

    sleep(5)
    print("Start Recording")
    deviceController.sendStartRecording()

    for block in range(1, num_blocks + 1):

        # Randomize emotions for this block
        block_emotions = emotions.copy()
        random.shuffle(block_emotions)

        for emotion in block_emotions:

            # --- STEP 1: INSTRUCTION (3 seconds) ---
            deviceController.sendTag(
                "INSTRUCTION",
                {
                    "trial": trial_counter,
                    "block": block,
                    "emotion": emotion
                }
            )
            print(f"[Block {block}] Trial {trial_counter}: INSTRUCTION → {emotion}")
            sleep(3)

            # --- STEP 2: MAINTAIN EXPRESSION (5 seconds) ---
            deviceController.sendTag(
                "EXPRESSION",
                {
                    "trial": trial_counter,
                    "block": block,
                    "emotion": emotion
                }
            )
            print(f"[Block {block}] Trial {trial_counter}: EXPRESSION → {emotion}")
            sleep(5)

            # --- STEP 3: END TRIAL (1 second) ---
            deviceController.sendTag(
                "END",
                {
                    "trial": trial_counter,
                    "block": block,
                    "emotion": emotion
                }
            )
            print(f"[Block {block}] Trial {trial_counter}: END → {emotion}")
            sleep(1)

            trial_counter += 1

    sleep(1)
    deviceController.sendStopRecording()
    print("Recording Ended")
    sleep(45)