
import os
import cv2
from nucleuskit_pipeline.logging_utils import printInfo, printWarning


def savePreviewFrame(videoName, outputName, frame=3):
    """
    Extract a preview frame from a video file.
    
    Args:
        videoName: Path to the video file
        outputName: Path where to save the screenshot
        frame: Frame number to extract (default: 3)
    """
    try:
        vid = cv2.VideoCapture(videoName)

        for i in range(frame+1):
            ret, im = vid.read()

        cv2.imwrite(outputName, im)
        vid.release()
    except Exception as e:
        printWarning(f"[metaUtils] Unable to generate screenshot preview: {e}")


def generateScreenshots(basepath):
    """
    Generate preview screenshots from video files.
    
    Args:
        basepath: Path to the recording directory
    """
    printInfo("[metaUtils] Generating screenshots")
    
    screen_video = os.path.sep.join([basepath, 'rawData', "screen_0.mp4"])
    face_video = os.path.sep.join([basepath, 'rawData', "face_0.mp4"])
    
    if os.path.isfile(screen_video):
        savePreviewFrame(screen_video, os.path.sep.join([basepath, 'meta', "screenPreview.jpg"]))
        printInfo("[metaUtils] Screen preview generated")
    else:
        printWarning(f"[metaUtils] Skipping {screen_video}, file not present")

    if os.path.isfile(face_video):
        savePreviewFrame(face_video, os.path.sep.join([basepath, 'meta', "facePreview.jpg"]))
        printInfo("[metaUtils] Face preview generated")
    else:
        printWarning(f"[metaUtils] Skipping {face_video}, file not present")

