import os

from Python.WLED_Discovery import discover_wled_devices

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
from openCV_methods import *
import pandas as pd

frame = cv2.imread('qrtest2.png')
bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
cv2.imshow('window', frame)

#qrPoints = []
#qrNames = []
#cv2.QRCodeDetector.decodeMulti(bgr_frame, qrPoints, qrNames)
qrDetector = cv2.QRCodeDetector()
retval, qrNames, qrPoints, _ = qrDetector.detectAndDecodeMulti(frame)
for eachCode in qrPoints:
    xAvg = 0
    yAvg = 0
    for eachPoint in eachCode:
        xAvg += eachPoint[0]
        yAvg += eachPoint[1]
    xAvg /= len(eachCode)
    yAvg /= len(eachCode)

    rValue = 0
    gValue = 0
    bValue = 0
    rValue, gValue, bValue = position_to_rgb(xAvg, 870)

    cv2.circle(frame, (int(xAvg), int(yAvg)), 10, (bValue, gValue, rValue), cv2.FILLED)

for each in np.arange(0, 870, 30):
    xAvg = each
    yAvg = 150
    rValue, gValue, bValue = position_to_rgb(xAvg, 870)
    cv2.circle(frame, (xAvg, yAvg), 10, (bValue, gValue, rValue), cv2.FILLED)

for each in qrPoints[2]:
    cv2.circle(frame, (int(each[0]), int(each[1])), 10, (0, 0, 255), cv2.FILLED)

cv2.imshow('window', frame)

#print(qrPoints)
#print(qrNames)
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break