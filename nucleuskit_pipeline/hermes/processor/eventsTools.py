"""
Events Tools

Processes and filters raw events data from sessions.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
import sys
from nucleuskit_pipeline.shared.rawEventsUtils import _loadRawEvents, _writeEvents
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError


def eventProcessor(recPath):
    """
    Process raw events and separate into feature events and web events.
    
    Args:
        recPath: Path to the recording directory
    """
    printInfo("[eventsTools] Processing events")
    
    raw_events_path = os.path.sep.join([recPath, "rawData", "rawEvents.csv"])
    rawEvents = _loadRawEvents(raw_events_path)

    if rawEvents is None:
        printWarning("[eventsTools] No raw events found, skipping")
        return None

    featureEvents = []
    webEvents = []

    for i in range(len(rawEvents)):
        if rawEvents[i][1] == 'WEB_SCROLL' or rawEvents[i][1] == 'WEB_TAB_UPDATE' or rawEvents[i][
            1] == 'WEB_INITIAL_SCROLL_POSITION':
            webEvents.append(rawEvents[i])

            if rawEvents[i][1] == 'WEB_TAB_UPDATE' and ("complete" in rawEvents[i][2]):
                featureEvents.append(rawEvents[i])

        elif rawEvents[i][1] == 'NO_KEYPRESS' or rawEvents[i][1] == 'KEYPRESS' or rawEvents[i][1] == 'ANSWER':
            pass
        else:
            featureEvents.append(rawEvents[i])

    if featureEvents is not None:
        _writeEvents(featureEvents.copy(), os.path.sep.join([recPath, "features","processedFeatureEvents.csv"]))

    if webEvents is not None:
        _writeEvents(webEvents.copy(), os.path.sep.join([recPath, "features","processedWebFeatureEvents.csv"]))

    printInfo(f"[eventsTools] Processed {len(featureEvents)} feature events and {len(webEvents)} web events")
