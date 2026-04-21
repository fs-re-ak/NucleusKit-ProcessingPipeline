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
    SHOW_IN_ADAPTER = False 
    
    
MQTTConfigs = {

  "streams": {
    "MQTTFrameStream": {
      "enabled": True,
      "topic": "frame2/"
    }
  },
  "mqtt": {
    "controller": {
      "enabled": True,
      "topic": "TestUnit/control/"
    },
    "broker": {
      "enabled": True,
      "source": "Config",
      "ip": "192.168.50.235",
      "port": 1883
    }
  }

}
    

def frameHandler(frame):
    """
    Call back used for demo, it simply shows the content of the frame on screen
    :param frame: a dictionnary containing emotional and cognitive metrics
    """
    showFrame(frame)
    pass

# instantiate the FrameManager and attach the callback.
inputFrames = FrameManager('MQTT', streamConfig=MQTTConfigs, deviceFilter=Configuration.DEVICE_ID, showFrame=Configuration.SHOW_IN_ADAPTER,stubbed=not Configuration.INPUT_FRAME_ACTIVE)

# if not shown in adapter (debugging purpose), attach a print callback
if not Configuration.SHOW_IN_ADAPTER:
    inputFrames.attachCallback(frameHandler)

# if active, instantiate the controller
if Configuration.OPERATION_MODE=="CONTROL_DEMO":
    deviceController = DeviceControllerManager('MQTT', streamConfig=MQTTConfigs, stubbed=not Configuration.DEVICE_CONTROLLER_ACTIVE)

# basic stream demo, simply shows data on screen
if Configuration.OPERATION_MODE == "LIVE_DISPLAY":

    # runs undefinitely
    while True:
        sleep(1)

# basic control demo, will record a short session and send some tags to show how it's done.
elif Configuration.OPERATION_MODE == "CONTROL_DEMO":

    sleep(5)
    print("Start Recording")
    deviceController.sendStartRecording()

    for i in range(6):
        deviceController.sendTag("TEST",["Value",i])
        sleep(3)

    sleep(1)
    deviceController.sendStopRecording()
    print("Recording Ended")
    sleep(45)
