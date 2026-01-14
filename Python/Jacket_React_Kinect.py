import os

#from Python.WLED_Discovery import discover_wled_devices, register_aruco_for_wled
from CustomMethods import *
from CustomMethods.openCV_methods import depth_to_camera_space

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import pandas as pd
import ctypes
import pickle

from pykinect2024 import PyKinectRuntime
from pykinect2024 import PyKinect2024
from pykinect2024.PyKinect2024 import *

FRAME_WIDTH = 1920 # Kinect v2 color feed resolution
FRAME_HEIGHT = 1080

imageNum = 0

InfraredPoints = np.float32([[221, 138], [218, 300], [86, 301], [91, 133]])
ColorPoints = np.float32([[784, 339], [761, 814], [371, 806], [396, 313]])
perspectiveTransform = cv2.getPerspectiveTransform(ColorPoints, InfraredPoints)

COLOR_INTRINSICS = np.array([[1081.37, 0.0, 959.5],
                             [0.0, 1081.37, 539.5],
                             [0.0, 0.0, 1.0]])
ROLLING_AVERAGE_FRAMES = 5
ENCODING = 'utf-8' #Encoding the qr codes are in
WLED_PORT = 21324
VIDEO_MODE = 'infrared' # Currently support the color and infrared feeds from kinect
DISPLAY_DEPTH = True
FILENAME = 'DEVICES.pkl'
LOAD = True
ABSOLUTE_POSITION = True
ABSOLUTE_POSITION_SCALE = 1

if LOAD:
    DEVICES = pd.read_pickle(FILENAME)
else:
    DEVICES = pd.DataFrame(columns=['name', 'ip', 'numLED', 'markerID'])

import cv2.aruco as aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)


def get_3D_coordinates(filename, show_each_frame=False):
    """saves the 3D positions of a list of 2D pixel positions in the colour image. Correspodning depth data stored in DEPTH.filename.pickle"""

    # Define a list of 2D coordinates you want to locate
    colour_image_pixels_to_locate_list = [[880, 555], [1440, 200]]

    # Start a kinect (NEED TO CONNECT A KINECT or run a recording in kinect studio to make this command work,  even though we are reading saved depth values)
    #kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

    # Do a bunch of defines required for matching the colour coordinates to their depth later
    color2depth_points_type = _DepthSpacePoint * np.int32(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(_DepthSpacePoint))
    S = 1080 * 1920
    TYPE_CameraSpacePointArray = PyKinect2024._CameraSpacePoint * S
    csps1 = TYPE_CameraSpacePointArray()

    # load your saved depth data
    depthdatafile = open("DEPTH." + filename + ".pickle", "rb")

    # make list to store the 3D positions in
    pixel_positions_3D_list = []

    # Iterate over each saved frame of depth data
    depth_file_not_finished = True
    while depth_file_not_finished == True:
        try:
            depthframe = pickle.load(
                depthdatafile)  # each call loads a sucessive frame from a pickle file, so we need to do this once per frame

            three_D_pixel_positions_in_frame = []  # list to store the 3D pixel positions from one frame

            # Defines to allow colour pixel mapping to 3D coords to work correctly
            ctypes_depth_frame = np.ctypeslib.as_ctypes(depthframe.flatten())
            L = depthframe.size
            kinect._mapper.MapColorFrameToCameraSpace(L, ctypes_depth_frame, S, csps1)

            # Carry out certain actions if you want an image of where all the tracked points are in the depth data (makes program 20x slower)
            if show_each_frame == True:
                # Note the method on the line below, for finding the corrsponding depth pixel of a single tracked pixel in the colour image, is NOT what I am using to find the 3D position of a colour pixel
                kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), ctypes_depth_frame,
                                                         ctypes.c_uint(1920 * 1080), color2depth_points)

                cut_down_depth_frame = depthframe.astype(np.uint8)
                cut_down_depth_frame = np.reshape(cut_down_depth_frame, (424, 512))

            # Iterate over the lists of pixel positions in the 2D colour image to locate
            for pixel in colour_image_pixels_to_locate_list:

                # find x and y in pixel position in the 2D colour image
                x = pixel[0]
                y = pixel[1]

                # Find 3D position of each pixel (relative to camera) using Colour_to_camera method, all measurements (x, y and z) in m
                x_3D = csps1[y * 1920 + x].x
                y_3D = csps1[y * 1920 + x].y
                z_3D = csps1[y * 1920 + x].z
                pixel_position_3D = [x_3D, y_3D, z_3D]

                # if show_each_frame flag set,  display the depth data and corresponding points you are reading
                if show_each_frame == True:

                    try:

                        # method below finds 2D depth pixel that corresponds to a 2D colour pixel, for use in the pop up images, to show you what points you are tracking. While it could be used to find 3D joint positions, IT IS NOT THE METHOD I USE OR RECOMMEND FOR FINDING 3D JOINT POSITIONS, as it gives you x and y in pixels not m (z is in mm)
                        read_pos = x + y * 1920 - 1
                        depth_image_corresponding_x = int(color2depth_points[read_pos].x)
                        depth_image_corresponding_y = int(color2depth_points[read_pos].y)

                        # plot a circle at the pixel in the depth frame that matches the corresponding pixel in the image frame
                        cv2.circle(cut_down_depth_frame, (depth_image_corresponding_x, depth_image_corresponding_y), 5,
                                   (255, 0, 255), -1)

                        # note that the value below is NOT used in this code, included just for reference
                        corresponding_depth = depthframe[
                            ((depth_image_corresponding_y * 512) + depth_image_corresponding_x)]

                    except OverflowError:
                        # the SDK returns infinity for the depth of some positions, so we need to handle that
                        # I choose to not find the corresponding pixel in the depth image, and so dont plot a circle there, in this case
                        pass

                # Display annotated depth image if flag is set
                if show_each_frame == True:
                    cv2.imshow('KINECT Video Stream', cut_down_depth_frame)

                    # code to close window if escape is pressed, doesnt do anything in this program (as we keep calling for data to be displayed in the window) but included for reference
                    key = cv2.waitKey(1)
                    if key == 27:
                        pass

                # add 3D positions found in this frame to an intermediate list
                three_D_pixel_positions_in_frame.append(pixel_position_3D)

            # add per frame lists of 3D position into a results list
            pixel_positions_3D_list.append(three_D_pixel_positions_in_frame)

        # close loop at end of file
        except EOFError:
            cv2.destroyAllWindows()
            depth_file_not_finished = False

    # return 3D joint position lists
    return pixel_positions_3D_list

#Video mode setup
if (VIDEO_MODE == 'color'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)
elif (VIDEO_MODE == 'infrared'):
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
    FRAME_WIDTH = 512 #Kinect v2 Infrared camera resolution
    FRAME_HEIGHT = 424
    print(kinect._mapper.GetDepthCameraIntrinsics().FocalLengthY)
else:
    raise ValueError('Check video mode, should be color or infrared.')
kinectInfra = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Infrared)
kinectDepth = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Depth)

while True:
    print(kinect._mapper.GetDepthCameraIntrinsics().PrincipalPointX)
    if (VIDEO_MODE == 'infrared'):
        frame = kinect.get_last_infrared_frame()
        frame = frame.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        frame = (frame / 256).astype(np.uint8)
    elif (VIDEO_MODE == 'color'):
        frame = kinect.get_last_color_frame()
        frameInfra = kinectInfra.get_last_infrared_frame()
        frame = frame.reshape((1080, 1920, 4))
    frame = cv2.flip(frame, 1)

    frameDepth = kinectDepth.get_last_depth_frame()
    frameDepth = frameDepth.reshape((424, 512))
    frameDepth = cv2.flip(frameDepth, 1)

    # Saving this for taking more pictures to get perspective transform
    frameInfra = kinectInfra.get_last_infrared_frame()
    frameInfra = frameInfra.reshape((424, 512))
    frameInfra = cv2.flip(frameInfra, 1)
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

                if ABSOLUTE_POSITION:
                    rValue, gValue, bValue = position_to_rgb(normalize_clamped(cameraFramePoints[0], ABSOLUTE_POSITION_SCALE, FRAME_WIDTH), FRAME_WIDTH)
                else:
                    rValue, gValue, bValue = position_to_rgb(centerX, FRAME_WIDTH)

                cv2.circle(frame, (centerX, centerY), 10, (bValue, gValue, rValue), cv2.FILLED) # If found in devices put circle on center with transmit color
                cv2.putText(frame, f"{decodedData}, int:{intensity}, depth: {depthValue}", ((centerX+20), centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2) # Name on code with matching color
                cv2.putText(frame, f"R: {rValue}, G:{gValue}, B:{bValue}", ((centerX+20), (centerY+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2) # Name on code with matching color
                cv2.putText(frame, f"X: {cameraFramePoints[0]}, Y:{cameraFramePoints[1]}, Z:{cameraFramePoints[2]}", ((centerX+20), (centerY+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (bValue, gValue, rValue), 2)

                led_data = single_color_bytearray(int(rValue * intensity), int(gValue * intensity), int(bValue * intensity), thisNumLed)  # Initialize all LEDs to off
                # for i in range(thisNumLed):
                #     led_data[i * 3] = int(rValue * intensity)
                #     led_data[i * 3 + 1] = int(gValue * intensity)
                #     led_data[i * 3 + 2] = int(bValue * intensity)
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
        kinect.close()
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
        print(f'Saving images {imageNum}')
        cv2.imwrite(f'infrared{imageNum}.png', frameInfra)
        cv2.imwrite(f'color{imageNum}.png', frame)
        imageNum += 1
        print()

    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(DEVICES.head())
        #print(get_3D_coordinates('test', False))
        ctypes.c_float(1.9)
        testPoint = PyKinect2024._DepthSpacePoint()
        testPoint.x = ctypes.c_float(100.0)
        testPoint.y = ctypes.c_float(100.0)
        newPoint = kinect._mapper.MapDepthPointToColorSpace(testPoint, 1000)
        print(newPoint.x, newPoint.y)

        color2depth_points_type = _DepthSpacePoint * np.int32(1920 * 1080)
        color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(_DepthSpacePoint))
        ctypes_depth_frame = np.ctypeslib.as_ctypes(frameDepth.flatten())
        kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), ctypes_depth_frame,
                                                 ctypes.c_uint(1920 * 1080), color2depth_points)

        print(color2depth_points)


# cap.release()
cv2.destroyAllWindows()

