import os

from Python.WLED_Discovery import discover_wled_devices

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import cv2.aruco as aruco
# Choose your dictionary — 4x4 with 50 unique markers is good for quick testing
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create the detector using the newer Detector class
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

import mediapipe as mp
import numpy as np
from openCV_methods import *
from pyzbar.pyzbar import decode
import pandas as pd

frame = cv2.imread('aruco2.png')
bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
cv2.imshow('window', frame)

#qrPoints = []
#qrNames = []
#cv2.QRCodeDetector.decodeMulti(bgr_frame, qrPoints, qrNames)

corners, ids, rejected = detector.detectMarkers(frame)
if ids is not None:
    for i, marker_id in enumerate(ids.flatten()):
        decodedData = str(marker_id)  # Use ArUco ID as the unique identifier
        corner_pts = corners[i][0]
        centerX = int(np.mean(corner_pts[:, 0]))
        centerY = int(np.mean(corner_pts[:, 1]))
        cv2.circle(frame, (centerX, centerY), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, decodedData, ((centerX + 20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)

hsv = np.uint8([[[150, 255, 255]]])
print(hsv)
# qrDetector = cv2.QRCodeDetector()
# retval, qrNames, qrPoints, _ = qrDetector.detectAndDecodeMulti(frame)
# scannedCodes = decode(frame)
# print(scannedCodes)
# for indiv in scannedCodes:
#     print(average_of_points(*indiv.polygon))
#
# for eachCode in qrPoints:
#     xAvg = 0
#     yAvg = 0
#     for eachPoint in eachCode:
#         xAvg += eachPoint[0]
#         yAvg += eachPoint[1]
#     xAvg /= len(eachCode)
#     yAvg /= len(eachCode)
#
#     rValue = 0
#     gValue = 0
#     bValue = 0
#     rValue, gValue, bValue = position_to_rgb(xAvg, 870)
#
#     cv2.circle(frame, (int(xAvg), int(yAvg)), 10, (bValue, gValue, rValue), cv2.FILLED)
#
# for each in np.arange(0, 870, 30):
#     xAvg = each
#     yAvg = 150
#     rValue, gValue, bValue = position_to_rgb(xAvg, 870)
#     cv2.circle(frame, (xAvg, yAvg), 10, (bValue, gValue, rValue), cv2.FILLED)
#
# for each in qrPoints[2]:
#     cv2.circle(frame, (int(each[0]), int(each[1])), 10, (0, 0, 255), cv2.FILLED)

cv2.imshow('window', frame)

#print(qrPoints)
#print(qrNames)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break