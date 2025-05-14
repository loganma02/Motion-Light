import os
from pickle import FRAME

from absl.testing.flagsaver import restore_flag_values

from Python.WLED_Discovery import discover_wled_devices, register_aruco_for_wled

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import mediapipe as mp
import numpy as np
from pyzbar.pyzbar import decode
from openCV_methods import *
import pandas as pd

from pykinect2024 import PyKinectRuntime
from pykinect2024 import PyKinect2024

FRAME_WIDTH = 1920 # Kinect v2 color feed resolution
FRAME_HEIGHT = 1080

InfraredPoints = np.float32([[221, 138], [218, 300], [86, 301], [91, 133]])
ColorPoints = np.float32([[784, 339], [761, 814], [371, 806], [396, 313]])
perspectiveTransform = cv2.getPerspectiveTransform(ColorPoints, InfraredPoints)

COLOR_INTRINSICS = np.array([[1081.37, 0.0, 959.5],
                             [0.0, 1081.37, 539.5],
                             [0.0, 0.0, 1.0]])
ROLLING_AVERAGE_FRAMES = 5
ENCODING = 'utf-8' #Encoding the qr codes are in
WLED_PORT = 21324
VIDEO_MODE = 'color' # Currently support the color and infrared feeds from kinect
DISPLAY_DEPTH = True
FILENAME = 'DEVICES.pkl'
LOAD = True

if LOAD:
    DEVICES = pd.read_pickle(FILENAME)
else:
    DEVICES = pd.DataFrame(columns=['name', 'ip', 'numLED', 'markerID'])

import cv2.aruco as aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

#Video mode setup
if (VIDEO_MODE == 'color'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)
elif (VIDEO_MODE == 'infrared'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
    FRAME_WIDTH = 512 #Kinect v2 Infrared camera resolution
    FRAME_HEIGHT = 424
else:
    raise ValueError('Check video mode, should be color or infrared.')
kinectInfra = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
kinectDepth = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Depth)

while True:
    if (VIDEO_MODE == 'infrared'):
        frame = kinect.get_last_infrared_frame()
        frame = frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        frame = (frame / 256).astype(np.uint8)
    elif (VIDEO_MODE == 'color'):
        frame = kinect.get_last_color_frame()
        #frameInfra = kinectInfra.get_last_infrared_frame()
        frame = frame.reshape((1080, 1920, 4))
    frame = cv2.flip(frame, 1)

    frameDepth = kinectDepth.get_last_depth_frame()
    frameDepth = frameDepth.reshape((424, 512))
    frameDepth = cv2.flip(frameDepth, 1)

    # Saving this for taking more pictures to get perspective transform
    # frameInfra = kinectInfra.get_last_infrared_frame()
    # frameInfra = frameInfra.reshape((424, 512))
    # frameInfra = cv2.flip(frameInfra, 1)
    # frameDepth = cv2.resize(frameDepth, (FRAME_HEIGHT, FRAME_WIDTH))

    corners, ids, rejected = detector.detectMarkers(frame)
    depthX = 0 #TODO: make this support multiple codes, may just need to process depth frame after all
    depthY = 0
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            decodedData = str(marker_id)  # Use ArUco ID as the unique identifier
            corner_pts = corners[i][0]
            centerX = int(np.mean(corner_pts[:, 0]))
            centerY = int(np.mean(corner_pts[:, 1]))

            point = np.array([[[centerX, centerY]]], dtype=np.float32)
            transformedPoint = cv2.perspectiveTransform(point, perspectiveTransform)[0][0]

            depthX = min(int(transformedPoint[0]), 511)
            depthY = min(int(transformedPoint[1]), 423)

            depthValue = frameDepth[depthY, depthX]
            if depthValue > 500 and depthValue < 1500:
                intensity = abs((depthValue*(-1.0/1000.0)) + 1.5)
            else:
                intensity = 1.0

            cameraFramePoints = depth_to_camera_space(centerX, centerY, depthValue, COLOR_INTRINSICS)

            if decodedData in DEVICES['markerID'].values: # Code exists in dataframe, do the coloring process
                thisRow = DEVICES.loc[DEVICES['markerID'] == decodedData]
                thisNumLed = thisRow.iloc[0]['numLED']
                thisIP = thisRow.iloc[0]['ip']

                rValue, gValue, bValue = position_to_rgb(centerX, FRAME_WIDTH)
                cv2.circle(frame, (centerX, centerY), 10, (bValue, gValue, rValue), cv2.FILLED) # If found in devices put circle on center with transmit color
                cv2.putText(frame, f"{decodedData}, int:{intensity}, depth: {depthValue}", ((centerX+20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2) # Name on code with matching color
                cv2.putText(frame, f"R: {rValue}, G:{gValue}, B:{bValue}", ((centerX+20), (centerY+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2) # Name on code with matching color
                # cv2.putText(frame, f"R: {rValue}, G:{gValue}, B:{bValue}", ((centerX + 20), (centerY + 20)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue),
                #             2)  # Name on code with matching color
                cv2.putText(frame, f"X: {cameraFramePoints[0]}, Y:{cameraFramePoints[1]}, Z:{cameraFramePoints[2]}", ((centerX+20), (centerY+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2)

                led_data = bytearray(thisNumLed * 3)  # Initialize all LEDs to off
                for i in range(thisNumLed):
                    led_data[i * 3] = int(rValue * intensity)
                    led_data[i * 3 + 1] = int(gValue * intensity)
                    led_data[i * 3 + 2] = int(bValue * intensity)
                send_wled_udp(led_data, thisIP, WLED_PORT, 'DRGB')
            else:
                cv2.circle(frame, (centerX, centerY), 10, (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, f"{decodedData}, int:{intensity}, depth: {depthValue}", ((centerX + 20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)
                cv2.putText(frame, f"X: {cameraFramePoints[0]}, Y:{cameraFramePoints[1]}, Z:{cameraFramePoints[2]}",
                            ((centerX + 20), (centerY + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            2)


    # scannedCodes = decode(frame)
    # if scannedCodes is not None:
    #     for eachCode in scannedCodes: # Go through each QR code found
    #         decodedData = eachCode.data.decode(ENCODING)
    #         centerX, centerY = average_of_points(*eachCode.polygon)
    #         if decodedData in DEVICES['name'].values: # Code exists in dataframe, do the coloring process
    #             thisRow = DEVICES.loc[DEVICES['name'] == decodedData]
    #             thisNumLed = thisRow.iloc[0]['numLED']
    #             thisIP = thisRow.iloc[0]['ip']
    #
    #             rValue, gValue, bValue = position_to_rgb(centerX, FRAME_WIDTH)
    #             cv2.circle(frame, (centerX, centerY), 10, (bValue, gValue, rValue), cv2.FILLED) # If found in devices put circle on center with transmit color
    #             cv2.putText(frame, decodedData, ((centerX+20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2) # Name on code with matching color
    #
    #             led_data = bytearray(thisNumLed * 3)  # Initialize all LEDs to off
    #
    #             for i in range(thisNumLed):
    #                 # Set the LED's RGB values
    #                 led_data[i * 3] = rValue
    #                 led_data[i * 3 + 1] = gValue
    #                 led_data[i * 3 + 2] = bValue
    #             send_wled_udp(led_data, thisIP, WLED_PORT, 'DRGB')
    #         else:
    #             cv2.circle(frame, (centerX, centerY), 10, (0, 0, 0), cv2.FILLED)
    #             cv2.putText(frame, decodedData, ((centerX + 20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                         (0, 0, 0), 2)  # Name on code with matching color

    if (VIDEO_MODE == 'color'):
        #frameDepthCopy = cv2.copyMakeBorder(frameDepth, 0, 0, 224, 224, cv2.BORDER_CONSTANT)
        frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
    elif (VIDEO_MODE == 'infrared'):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)

    if DISPLAY_DEPTH: # Save the frameDepth processing if I dont want to display it
        frameDepth = cv2.normalize(frameDepth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        frameDepth = cv2.cvtColor(frameDepth, cv2.COLOR_GRAY2RGBA)
        frameDepth = cv2.circle(frameDepth, (depthX, depthY), 10, (255, 255, 255), cv2.FILLED)
        cv2.imshow('Motion Light Depth', frameDepth)

    cv2.imshow('Motion Light Main', frame)


    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Saving device list as {FILENAME}")
        DEVICES.to_pickle(FILENAME)
        print("Quitting...")
        break

    # Listen for refresh when r is pressed
    if cv2.waitKey(1) & 0xFF == ord('r'):
        print('Refreshing Devices...')
        DEVICES = discover_wled_devices()
        DEVICES = register_aruco_for_wled(DEVICES)
        print('Done Refreshing!')
        print()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print('Saving images')
        cv2.imwrite('infrared1.png', frameInfra)
        cv2.imwrite('color1.png', frame)
        print()

    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(DEVICES.head())

# cap.release()
cv2.destroyAllWindows()

