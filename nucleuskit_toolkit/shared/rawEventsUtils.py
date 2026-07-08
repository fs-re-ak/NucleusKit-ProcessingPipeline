

from ast import literal_eval
import csv
import numpy as np
import os

FEATURE_EVENTS_FILTER = ["TRIAL_START", "TRIAL_STOP"]
BEHAVIORAL_EVENTS_FILTER = ["KEY_PRESSED"]


def feedFeatureEvents(dataset, destinationPath):
    rawEvents =  _loadRawEvents(dataset)
    
    if rawEvents is None:
        return None
    
    featureEvents = []
    webEvents = []
    
    for i in range(len(rawEvents)):
        if rawEvents[i][1] == 'WEB_SCROLL' or rawEvents[i][1] == 'WEB_TAB_UPDATE' or rawEvents[i][1] == 'WEB_INITIAL_SCROLL_POSITION' :
            webEvents.append(rawEvents[i])

            if rawEvents[i][1] == 'WEB_TAB_UPDATE' and ("complete" in rawEvents[i][2]):
                featureEvents.append(rawEvents[i])

        elif rawEvents[i][1] == 'NO_KEYPRESS' or rawEvents[i][1] == 'KEYPRESS' or rawEvents[i][1] == 'ANSWER':
            pass
        else:
            featureEvents.append(rawEvents[i])

    if featureEvents is not None:
        _writeEvents(featureEvents.copy(), destinationPath+"processedFeatureEvents.csv")
        
    if webEvents is not None:
        _writeEvents(webEvents.copy(), destinationPath+"processedWebFeatureEvents.csv")

    return rawEvents



def loadRawEvents(recPath, showEvents=False):

    validFilenames = ["rawEvents.csv", "events_0.csv"]
    filename = None

    for tmp in validFilenames:
        if os.path.exists(os.path.join(recPath, "rawData", tmp)):
            filename = tmp
            break


    fileFullpath = os.path.join(recPath,"rawData",filename)

    if not os.path.isfile(fileFullpath):
        print(f"no tmp Events to push {fileFullpath}")
        return None

    events = []
    pcRefTS = None
    rockRefTS = None
    with open(fileFullpath, 'r') as eventsFile:
        reader = csv.reader(eventsFile)

        no_timestamp_header = True

        for row in reader:
            for i in range(len(row)):
                row[i] = row[i].strip("'") # patch for ' inserted in Event types, by Vizia
                row[i] = row[i].lstrip()
                row[i] = row[i].rstrip()

            if len(row) > 3:
                row[2] = ",".join(row[2:len(row)])
                row = row[0:3]

            events.append(row)

            if len(row) >= 3:
                if row[1] == "TIMESTAMP":
                    rockRefTS = float(row[2])
                    pcRefTS = float(row[0])
                elif row[2] == "TIMESTAMP": # patch for badly formatted events
                    rockRefTS = float(row[1])
                    pcRefTS = float(row[0])
                elif no_timestamp_header is True:
                    rockRefTS = float(row[0])
                    pcRefTS = float(row[0])
                no_timestamp_header = False

        _correctEventsTimestamp(rockRefTS, pcRefTS, events)

        if showEvents:
            print(events)
        if len(events)>0:
            if events[0][1] == "TIMESTAMP" or events[0][2] == "TIMESTAMP":
                events.remove(events[0])

    return events




def _loadRawEvents(dataset, showEvents=False):


    if not os.path.isfile(dataset):
        print("no tmp Events to push")
        return None

    events = []
    pcRefTS = None
    rockRefTS = None
    with open(dataset,'r') as eventsFile:
        reader = csv.reader(eventsFile)

        no_timestamp_header = True

        for row in reader:
            for i in range(len(row)):
                row[i] = row[i].strip("'") # patch for ' inserted in Event types, by Vizia
                row[i] = row[i].lstrip()
                row[i] = row[i].rstrip()

            if len(row) > 3:
                row[2] = ",".join(row[2:len(row)])
                row = row[0:3]

            events.append(row)

            if len(row) >= 3:
                if row[1] == "TIMESTAMP":
                    rockRefTS = float(row[2])
                    pcRefTS = float(row[0])
                elif row[2] == "TIMESTAMP": # patch for badly formatted events
                    rockRefTS = float(row[1])
                    pcRefTS = float(row[0])
                elif no_timestamp_header is True:
                    rockRefTS = float(row[0])
                    pcRefTS = float(row[0])
                no_timestamp_header = False

        _correctEventsTimestamp(rockRefTS, pcRefTS, events)

        if showEvents:
            print(events)
        if len(events)>0:
            if events[0][1] == "TIMESTAMP" or events[0][2] == "TIMESTAMP":
                events.remove(events[0])

    return events


def _correctEventsTimestamp(rockRefTS, pcRefTS, events):
    # adjust time base
    for event in events:
        event[0] = float(event[0].strip("'"))
        if rockRefTS is not None:
            event[0] += (rockRefTS - pcRefTS)
            event[0] -= (rockRefTS-1)


def showUniqueEventID(events):
    eventID = []
    for x in events:
        if x[1] not in eventID:
            eventID.append(x[1])
    print(eventID)


def extractMultipleEventIDs(events, eventIDs):
    subset = []
    for event in events:
        if event[1] in eventIDs:
            subset.append(event)

    return subset


def _writeEvents(events, filename):

    with open(filename,'w', newline='') as eventsFile:
        writer = csv.writer(eventsFile)

        for event in events:


            time = event[0]
            name = event[1]
            value = event[2]
            tmp = []
            tmp.append(name)
            tmp.append("PROCESSED")
            tmp.append(str(time*1000))
            tmp.append("")
            if str(value) == '[]':
                tmp.append("")
            else:
                tmp.append(value.replace('"',''))
            writer.writerow(tmp)