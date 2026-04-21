"""
UWB Processor

Processes Ultra-Wideband (UWB) positioning data using multilateration.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

"""
Ultra-Wideband (UWB) Spatial Processing

This module processes UWB positioning data to compute participant locations and zone transitions.
It handles sensor fusion, filtering, and zone detection for indoor positioning systems.

Key Functions:
    - processUWB: Process raw UWB data into position coordinates
    - positionToZone: Map positions to predefined spatial zones

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import csv
import math
import cv2  # may be unused
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError

# Module-level constants
UWB_HDR = "[UWB] "
ZONE_HDR = "[ZONE] "


def round_zero5(num):
    """Round to nearest 0.5"""
    return round(num * 2) / 2


def processUWB(basepath):
    """
    Process UWB positioning data:
        * Compute multilateration
        * Resample UWB position
        * Extract unique antennas
        * Write all information as feature dataframes
    
    Args:
        basepath: Path to the recording directory
    """
    printInfo(f"{UWB_HDR}Processing UWB data")

    # Skip if already done (caching)
    positions_path = os.path.join(basepath, 'results', 'positions.csv')
    if os.path.isfile(positions_path):
        printInfo(f"{UWB_HDR}UWB already processed, using cached results")
        return

    uwbFiles = ["uwb_0.csv", "uwb.tmp"]
    uwbFile = None
    for filename in uwbFiles:
        filepath = os.path.join(basepath, "rawData", filename)
        if os.path.isfile(filepath):
            uwbFile = filename
            break

    if uwbFile is None:
        printWarning(f"{UWB_HDR}No UWB file found, tested files: {uwbFiles}")
        return


    antList = []
    posList = []

    # load raw UWB file
    uwbFile = open(os.path.join(basepath, "rawData", uwbFile), "r")

    # prepare output files
    lines = uwbFile.readlines()

    if not len(lines):
        printWarning(f"{UWB_HDR}UWB doesn't contain valid lines. Skipping")
        return

    refTime = float(lines[0].split(',')[0])

    # remove all spurious lines
    lines = [s for s in lines if 'DIST' in s]

    if not len(lines):
        printWarning(f"{UWB_HDR}UWB doesn't contain valid lines. Skipping")
        return

    printInfo(f"{UWB_HDR}Running 3+ antennas solver")
    _run_firstPass(lines, posList, antList, refTime)
    printInfo(f"{UWB_HDR}Number of None: {len([item for item in posList if item is None])}")

    printInfo(f"{UWB_HDR}Running 2 antennas solver")
    _run_secondPass(lines, posList, refTime)
    printInfo(f"{UWB_HDR}Number of None: {len([item for item in posList if item is None])}")

    printInfo(f"{UWB_HDR}Running 1 antennas solver")
    _run_thirdPass(lines, posList, refTime)
    printInfo(f"{UWB_HDR}Number of None: {len([item for item in posList if item is None])}")

    posList = [item for item in posList if item is not None]

    # wrapping up the results

    if not len(posList):
        printWarning(f"{UWB_HDR}No valid position found, skipping.")
        return

    # organize into a dataframe
    posArray = np.array(posList)
    try:
        posArray[:, 0] /= 1000  # convert to seconds
    except Exception as e:
        printError(f"{UWB_HDR}Error converting timestamps: {e}")
        return
        
    posDf = pd.DataFrame(posArray, columns=["Timestamp", "x", "y","z"])

    # TODO: Implement resample without external dependency
    # posDf = resample(posDf, "500ms", 0.0)

    # write
    positionsDestPath = os.path.sep.join([basepath, "results", "positions.csv"])
    posDf.to_csv(positionsDestPath, index=False, float_format='%.2f', na_rep='NULL')
    
    printInfo(f"{UWB_HDR}UWB processing completed")


    #plt.scatter(posDf.loc[:, "x"], posDf.loc[:, "y"], c=posDf.loc[:, "z"])
    #plt.show()

    # save unique antennas as well
    uniqueAntennas = _extractUniqueAntennas(antList)
    # antFile = open(basepath + "/features/antennas.csv", "w", newline='')
    antFile = open(os.path.sep.join([basepath, "features", "antennas.csv"]), "w", newline='')
    antWriter = csv.writer(antFile)
    antWriter.writerow(["id", "x", "y", "z"])
    antWriter.writerows(uniqueAntennas)
    antFile.close()


def _run_firstPass(lines, posList, antList, refTime):
    # first pass - solve all nb Antennas 3+
    for idx, line in enumerate(lines):

        try:
            valid, ts, pos, antList_tmp = _idealSolver(line, refTime)
        except Exception as e:
            printError(repr(e))
            pos = None

        if pos is not None:
            pos = _bruteForceZ(antList_tmp, pos)
            posList.append(np.r_[[ts], pos])
        else:
            posList.append(None)

        if antList_tmp is not None:
            for x in antList_tmp:
                antList.append(x)


def find_next_non_null_element(lst, start_index):
    for index in range(start_index + 1, len(lst)):
        if lst[index] is not None:
            return lst[index]
    return None

def _run_secondPass(lines, posList, refTime):

    pos = None
    lastNonNullPosition = None
    nextNonNullPosition = None
    closestGuess = None
    # first pass - solve all nb Antennas 3+
    for idx, line in enumerate(lines):

        #try:
        if posList[idx] is None:
            nextNonNullPosition = find_next_non_null_element(posList, idx)
            closestGuess = None

            if nextNonNullPosition is None:
                closestGuess = lastNonNullPosition
            elif lastNonNullPosition is None:
                closestGuess = nextNonNullPosition
            elif nextNonNullPosition is not None and lastNonNullPosition is not None:
                currentTime = float(line.split(',')[0])
                if abs(nextNonNullPosition[0]-currentTime) < abs(lastNonNullPosition[0]-currentTime):
                    closestGuess = nextNonNullPosition
                else:
                    closestGuess = lastNonNullPosition

            if closestGuess is not None:
                valid, ts, pos, antList_tmp = _2AntSolver(line, refTime, closestGuess[1:])

            nextNonNullPosition = None

            if pos is not None:
                pos = _bruteForceZ(antList_tmp, pos)

                if posList[idx] is not None:
                    printError("Error!!!")
                    sys.exit()
                posList[idx] = np.r_[[ts], pos]
        else:
            lastNonNullPosition = posList[idx]

        #except Exception as e:
        #    print(repr(e))
        #    pass


def _run_thirdPass(lines, posList, refTime):

    pos = None
    lastNonNullPosition = None
    nextNonNullPosition = None
    closestGuess = None
    # first pass - solve all nb Antennas 3+
    for idx, line in enumerate(lines):

        #try:
        if posList[idx] is None:
            nextNonNullPosition = find_next_non_null_element(posList, idx)
            closestGuess = None

            if nextNonNullPosition is None:
                closestGuess = lastNonNullPosition
            elif lastNonNullPosition is None:
                closestGuess = nextNonNullPosition
            elif nextNonNullPosition is not None and lastNonNullPosition is not None:
                currentTime = float(line.split(',')[0])
                if abs(nextNonNullPosition[0]-currentTime) < abs(lastNonNullPosition[0]-currentTime):
                    closestGuess = nextNonNullPosition
                else:
                    closestGuess = lastNonNullPosition

            if closestGuess is not None:
                valid, ts, pos, antList_tmp = _1AntSolver(line, refTime, closestGuess[1:])

            nextNonNullPosition = None

            if pos is not None:
                pos = _bruteForceZ(antList_tmp, pos)

                if posList[idx] is not None:
                    printError("Error!!!")
                    sys.exit()
                posList[idx] = np.r_[[ts], pos]
        else:
            lastNonNullPosition = posList[idx]

        #except Exception as e:
        #    print(repr(e))
        #    pass



def _idealSolver(line, refTime):
    line = line.replace('"', '')
    line = line.replace(' ', '')

    line = line.split(',')

    nbAntennas = int(line[2])

    antPos = []
    antDist = []
    antList = []
    for i in range(nbAntennas):
        antNumber = line[3 + i * 6]
        antId = line[3 + i * 6 + 1]
        antPos_tmp = [float(line[3 + i * 6 + 2]), float(line[3 + i * 6 + 3]), float(line[3 + i * 6 + 4])]
        antPos.append(antPos_tmp)
        antDist.append(float(line[3 + i * 6 + 5]))
        antList.append([antId] + antPos_tmp)

    antPos = list(np.array(antPos))

    # check if can be solved using standard solver
    if nbAntennas >= 3:
        return True, float(line[0]) - refTime, _uwb_solve(antDist, antPos), antList
    else:
        return False, float(line[0]) - refTime, None, None


def _bruteForceZ(antList_tmp, pos):

    #printWarning("[UWBProcess] Warning: hardcoding Z coordinate - Logisco")

    antNames = [x[0] for x in antList_tmp]

    FIRST_FLOOR = ['DCB0', 'CF01', '9200', '4A82', '4107', '5338', 'D92B', '8902', '028C', 'D29D', '4709', '5D17', '00A0']
    SECOND_FLOOR = ['4496', 'CF1C', '0CA1', 'D816', '021A', 'D71A', 'D806', '012A', '0111', '8124', '0181', '0B2E', '91A9', 'D285', 'CFB0']

    if any(x in FIRST_FLOOR for x in antNames):
        pos[2] = 2.4
    elif any(x in SECOND_FLOOR for x in antNames):
        pos[2] = 70

    return pos


def loadPositions(basepath):
    posFile = open(basepath + "/pos.csv", "r")
    lines = posFile.readlines()

    allPos = []
    timestamps = []

    for line in lines:
        line = line.split(',')
        if "None" not in line[1]:
            pos = np.fromstring(line[1].replace("[", "").replace("]", ""), dtype=float, sep=' ')
            allPos.append(pos)

            timestamps.append(float(line[0]))

    allPos = np.array(allPos)
    return np.array(timestamps), np.array(allPos)

"""
# ref: https://github.com/glucee/Multilateration
def _uwb_solve(distances_to_station, stations_coordinates):
    def error(x, c, r):
        return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])

    l = len(stations_coordinates)
    S = sum(distances_to_station)
    # compute weight vector for initial guess
    W = [((l - 1) * S) / (S - w) for w in distances_to_station]
    # get initial guess of point location
    x0 = sum([W[i] * stations_coordinates[i] for i in range(l)])
    # optimize distance from signal origin to border of spheres
    return minimize(error, x0, args=(stations_coordinates, distances_to_station), method='Nelder-Mead').x
"""

def _2AntSolver(line, refTime,bestGuess):
    line = line.replace('"', '')
    line = line.replace(' ', '')
    line = line.split(',')

    nbAntennas = int(line[2])

    antPos = []
    antDist = []
    antList = []

    for i in range(nbAntennas):
        antNumber = line[3+i*6]
        antId = line[3+i*6 + 1]
        antPos_tmp = [float(line[3+i*6 + 2]), float(line[3+i*6 + 3]), float(line[3+i*6 + 4])]
        antPos.append(antPos_tmp)
        antDist.append(float(line[3+i*6 + 5]))
        antList.append([antId] + antPos_tmp)

    antPos = list(np.array(antPos))

    # check if can be solved using standard solver

    if nbAntennas >= 3:
        printError("Error, there shouldn't be 3+ antennas at this step...")
        sys.exit()
    elif nbAntennas==2:
        small_number = 1e-10
        antDist = [small_number if item == 0 else item for item in antDist]

        return True, float(line[0]) - refTime, _uwb_solve(antDist, antPos,x0=bestGuess), antList
    else:
        return False, float(line[0])-refTime, None, None



def _1AntSolver(line, refTime, bestGuess):
    line = line.replace('"', '')
    line = line.replace(' ', '')
    line = line.split(',')

    nbAntennas = int(line[2])

    antPos = []
    antDist = []
    antList = []

    for i in range(nbAntennas):
        antNumber = line[3+i*6]
        antId = line[3+i*6 + 1]
        antPos_tmp = [float(line[3+i*6 + 2]), float(line[3+i*6 + 3]), float(line[3+i*6 + 4])]
        antPos.append(antPos_tmp)
        antDist.append(float(line[3+i*6 + 5]))
        antList.append([antId] + antPos_tmp)

    antPos = list(np.array(antPos))

    # check if can be solved using standard solver

    if nbAntennas >= 2:
        printError("Error, there shouldn't be 2+ antennas at this step...")
        sys.exit()
    elif nbAntennas==1:
        return True, float(line[0]) - refTime, find_closest_point_on_circle(antDist[0], antPos[0][:2], bestGuess), antList
    else:
        return False, float(line[0])-refTime, None, None



def _extractUniqueAntennas(antList):
    uniques = []
    for x in antList:
        if x not in uniques:
            uniques.append(x)
    return uniques


def positionToZone(recPath, resources_path="./resources"):
    """
    Assign a zone tag to every position in position.csv based on ROI definitions.
    
    Args:
        recPath: Path to the recording directory
        resources_path: Path to resources directory containing roi.csv
    """
    printInfo(f"{ZONE_HDR}Assigning zones to positions")

    positionFilePath = os.path.sep.join([recPath, "results", "positions.csv"])
    roiFilePath = os.path.join(resources_path, "roi.csv")
    outPath = os.path.sep.join([recPath, "results", "zones.csv"])

    # Skip if already done (caching)
    if os.path.isfile(outPath):
        printInfo(f"{ZONE_HDR}Zone analysis already done, using cached results")
        return

    if not os.path.isfile(positionFilePath):
        printWarning(f"{ZONE_HDR}Missing 'positions.csv'. Please run UWB analysis beforehand.")
        return

    if os.path.isfile(roiFilePath):
        roi = pd.read_csv(roiFilePath)
    else:
        printWarning(f"{ZONE_HDR}ROI file not found: {roiFilePath}")
        return

    printInfo(f"{ZONE_HDR}Generating zone tags...")
    pos = pd.read_csv(positionFilePath)
    zoneDF = pd.DataFrame(data=np.c_[pos["Timestamp"].copy(), pos.shape[0] * [None]], columns=["Timestamp", "ROI"])

    for i in range(roi.shape[0]):
        x1, y1 = roi.loc[i, "x"], roi.loc[i, "y"]
        x2, y2 = x1 + roi.loc[i, "w"], y1 + roi.loc[i, "h"]
        maskX = [x1 <= p <= x2 for p in pos["x"]]
        maskY = [y1 <= p <= y2 for p in pos["y"]]
        zoneDF.loc[maskX and maskY, "ROI"] = roi.loc[i, "label"]

    zoneDF.to_csv(outPath, index=False)
    printInfo(f"{ZONE_HDR}Zone tagging completed")


def drawZoneRectangle(img, roi, offsets=(530, 370), scale=10, fontsize=4, roi_color='k', origin_color='r', origin='lower'):
    """
    Draw the zones defined by "roi" as rectangle on top of image "img".

    Args:
        img: 2d array representing an image
        roi: pandas DataFrame with columns=["label", "x", "y", "w", "h"] in meters
        offsets: translation of the reference point a.k.a. initiator (x, y) in pixels from the top-left corner
        scale: number of pixels per meter
        fontsize: size of the police when writing the zone labels
        roi_color: color of the region of interest's rectangle (always not filled)
        origin_color: color of the origin (reference) point (always filled)
        origin: "upper" or "lower", define the the y-axis origin left corner

    FOR DEV RUN:
        import matplotlib.image as mpimg

        # Campus Party
        img = mpimg.imread("D:\\REAK\\processes\\projects\\SPCM\\resources\\CampusParty\\CampusParty_Plan_Clean.png")
        roi = "D:\\REAK\\processes\\projects\\SPCM\\resources\\CampusParty_roi.csv"
        scale, offsets, origin, roi_color, origin_color, fontsize = 10, [530, 370], "lower", 'k', 'r', 4

        # Grand Salon Homme
        img = mpimg.imread("D:\\REAK\\processes\\projects\\SPCM\\resources\\GSH_Map.png")
        roi = "D:\\REAK\\processes\\projects\\SPCM\\resources\\GrandSalonHomme_roi.csv"
        scale, offsets, origin, roi_color, origin_color, fontsize = 1, [0, 0], "upper", 'k', 'r', 4
    """

    if type(roi) == str:
        outDir = os.path.dirname(roi)
        if os.path.isfile(roi):
            roi = pd.read_csv(roi)
        else:
            printWarning("Unrecognized roi.csv file: {roi}")
            return
    else:
        outDir = "." + os.path.sep

    if origin == "upper":
        img = np.flip(img, axis=0)
    fig = plt.imshow(img)

    # Draw origin marker as a 1x1 "meter" square
    fig.axes.add_patch(mpatches.Rectangle(offsets, scale, scale, edgecolor=origin_color, facecolor=origin_color))

    # Draw zones + label in their center
    textOffsets = (offsets[0] + fontsize, offsets[1] + fontsize)
    for i in range(roi.shape[0]):
        fig.axes.add_patch(mpatches.Rectangle(roi.loc[i, ["x", "y"]] * scale + offsets,
                                              roi.loc[i, "w"] * scale,
                                              roi.loc[i, "h"] * scale,
                                              edgecolor=roi_color, facecolor="none"))
        fig.axes.text((roi.loc[i, "x"] + int(roi.loc[i, "w"]/2)) * scale + textOffsets[0],
                      (roi.loc[i, "y"] + int(roi.loc[i, "h"]/2)) * scale + textOffsets[1],
                      str(roi.loc[i, "label"]).replace(" ", "\n"), ha='center', va='center',
                      fontsize=fontsize, color=roi_color)

    if origin == "upper":
        plt.gca().invert_yaxis()
    plt.axis("off")

    outputFileName = os.path.join(outDir, "background_ROI_ColLABorathon.png")
    plt.savefig(fname=outputFileName, bbox_inches='tight', pad_inches=0)

    # Show images overlaid by the regions of interest
    #plt.show(origin=origin)
    plt.show()


def GeneratePolygoneROI(img, roi, offsets=(530, 370), scale=10, fontsize=4, roi_color='k', origin_color='r', origin='lower'):
    """
    Generate a mask on the background image img that represent the ROIs defined in roi.

    Args:
        img: 2d array representing an image
        roi: pandas DataFrame with columns=["label", "x", "y"] in meters from initiator (origin). Label are repeated as
              many times as there is data point to define it.
        offsets: translation of the reference point a.k.a. initiator (x, y) in pixels from the top-left corner
        scale: number of pixels per meter
        fontsize: size of the police when writing the zone labels
        roi_color: color of the region of interest's rectangle (always not filled)
        origin_color: color of the origin (reference) point (always filled)
        origin: "upper" or "lower", define the the y-axis origin left corner

    FOR DEV RUN:
        import matplotlib.image as mpimg

        # Campus Party
        img = mpimg.imread("D:\\REAK\\processes\\projects\\SPCM\\resources\\CampusParty\\CampusParty_Plan_Clean.png")
        roi = "D:\\REAK\\processes\\projects\\SPCM\\resources\\CampusParty_roi.csv"
        scale, offsets, origin, roi_color, origin_color, fontsize = 10, [530, 370], "lower", 'k', 'r', 4

        # Grand Salon Homme
        img = mpimg.imread("D:\\REAK\\processes\\projects\\SPCM\\resources\\GSH_Map.png")
        roi = "D:\\REAK\\processes\\projects\\SPCM\\resources\\GrandSalonHomme_roi.csv"
        scale, offsets, origin, roi_color, origin_color, fontsize = 1, [0, 0], "upper", 'k', 'r', 4
    """

    if type(roi) == str:
        outDir = os.path.dirname(roi)
        if os.path.isfile(roi):
            roi = pd.read_csv(roi)
        else:
            printWarning(f"Unrecognized roi.csv file: {roi}")
            return
    else:
        outDir = "." + os.path.filesep

    # if origin == "upper":
    #     img = np.flip(img, axis=0)
    # fig = plt.imshow(img)

    myROIs = roi["label"].unique()
    for myROI in myROIs:
        myROIcoord = roi.loc[roi["label"] == myROI, ["x", "y"]].values
        mask = np.zeros(img.shape)  # (height, width)
        cv2.fillPoly(mask, [myROIcoord], 1)
        # TODO: Debug this show + save image
        cv2.imshow("mask", mask)
        cv2.imwrite(os.path.join(outDir, f"ROI_mask_{myROI}.png"), mask)

    # Define the ROIs. The order of the point is important as it will  be followed when drawing the polygon's contour
    myROI = [(1186, 548), (1151, 524), (1271, 540), (1233, 553)]  # (x, y)   --> Sun painting
    myROI = [(1186, 548), (1151, 524),  (1207, 629), (1208, 588)]  # (x, y)  --> Woman painting
    myROI = [(1233, 553), (1208, 588), (1207, 629), (1271, 540)]  # (x, y)   --> Sisters painting

    mask = np.ones(img.shape)  # (height, width)
    cv2.fillPoly(mask, [myROIcoord], 0)

    cv2.imshow('image', mask)
    cv2.waitKey(0)

    # Continue to adapt using cv2 below
    if origin == "upper":
        plt.gca().invert_yaxis()
    plt.axis("off")

    plt.savefig(fname=os.path.sep.join([outDir, "background_ROI_ColLABorathon.png"]), bbox_inches='tight', pad_inches=0)

    # Show images overlaid by the regions of interest
    plt.show(origin=origin)


# ref: https://github.com/glucee/Multilateration
def _uwb_solve(distances_to_station, stations_coordinates,x0=None):
    def error(x, c, r):
        return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])

    l = len(stations_coordinates)
    S = sum(distances_to_station)
    # compute weight vector for initial guess
    W = [((l - 1) * S) / (S - w) for w in distances_to_station]

    if x0 is None:
        # get initial guess of point location
        x0 = sum([W[i] * stations_coordinates[i] for i in range(l)])

    # optimize distance from signal origin to border of spheres
    return minimize(error, x0, args=(stations_coordinates, distances_to_station), method='Nelder-Mead').x

def find_closest_point_on_circle(radius, center, position):
    # Calculate the vector from the circle's center to the position
    delta_x = position[0] - center[0]
    delta_y = position[1] - center[1]

    # Calculate the distance from the center to the position
    distance = math.sqrt(delta_x**2 + delta_y**2)

    # Calculate the scaling factor to move from the center to the circle's edge
    scaling_factor = radius / distance

    # Calculate the coordinates of the closest point on the circle
    closest_x = center[0] + delta_x * scaling_factor
    closest_y = center[1] + delta_y * scaling_factor

    return [closest_x, closest_y, position[2]]
