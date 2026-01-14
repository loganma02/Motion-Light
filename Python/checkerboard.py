import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)  # corners (columns, rows) inside pattern
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points in real-world space (e.g., square size = 25mm)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= 25  # size in mm

objpoints = []  # 3D world points
imgpoints_color = []  # 2D points in color image
imgpoints_infrared = []  # 2D points in depth image

image_pairs = []

color_files = sorted(glob.glob("color*.png"))
infrared_files = sorted(glob.glob("infrared*.png"))

print(color_files)
print(infrared_files)

for cfile, dfile in zip(color_files, infrared_files):
    color_img = cv2.imread(cfile)
    infrared_img = cv2.imread(dfile)  # shape: (424, 512), dtype=uint16

    # Optional: normalize depth for visualization
    # norm_depth = cv2.convertScaleAbs(depth_img, alpha=255.0/np.max(depth_img))

    image_pairs.append((color_img, infrared_img))

for color_img, infrared_img in image_pairs:
    gray_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray_infrared = cv2.cvtColor(infrared_img, cv2.COLOR_BGR2GRAY)  # If depth is 16-bit

    ret_c, corners_c = cv2.findChessboardCorners(gray_color, CHECKERBOARD, None)
    ret_d, corners_d = cv2.findChessboardCorners(gray_infrared, CHECKERBOARD, None)

    if ret_c and ret_d:
        print("entered if")
        objpoints.append(objp)

        corners_c = cv2.cornerSubPix(gray_color, corners_c, (11, 11), (-1, -1), criteria)
        corners_d = cv2.cornerSubPix(gray_infrared, corners_d, (11, 11), (-1, -1), criteria)

        imgpoints_color.append(corners_c)
        imgpoints_infrared.append(corners_d)


# ret, mtx_color, dist_color, mtx_depth, dist_depth, R, T, _, _ = cv2.stereoCalibrate(
#     objpoints,
#     imgpoints_color,
#     imgpoints_infrared,
#     None, None,  # Or use known intrinsics
#     None, None,
#     (500, 500))
#
# print(ret)

# # Get image sizes BEFORE the calibration loops
# # Note: shape is (rows, cols), but imageSize needs (width, height), hence [::-1]
# color_img_size = cv2.imread(color_files[0]).shape[1::-1]
# infrared_img_size = cv2.imread(infrared_files[0], cv2.IMREAD_UNCHANGED).shape[1::-1]
#
# print(f"Color image size: {color_img_size}")
# print(f"Infrared image size: {infrared_img_size}")
#
# # --- After your point collection loop ---
#
# # Calibrate the Color Camera with ITS OWN size
# ret_c, mtx_c, dist_c, rvecs_c, tvecs_c = cv2.calibrateCamera(
#     objpoints, imgpoints_color, color_img_size, None, None
# )
#
# # Calibrate the Infrared Camera with ITS OWN size
# ret_ir, mtx_ir, dist_ir, rvecs_ir, tvecs_ir = cv2.calibrateCamera(
#     objpoints, imgpoints_infrared, infrared_img_size, None, None
# )
#
# stereocalibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
#
# # Call stereoCalibrate. The different resolutions are handled by the unique camera matrices.
# # We pass the color camera's size as the reference.
# retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
#     objpoints,
#     imgpoints_color,
#     imgpoints_infrared,
#     mtx_c,
#     dist_c,
#     mtx_ir,
#     dist_ir,
#     color_img_size, # Use one of the sizes as a reference
#     criteria=stereocalibration_criteria,
#     flags=cv2.CALIB_FIX_INTRINSIC
# )
#
# print("Stereo Calibration Complete!")
# print("Rotation matrix (R):\n", R)
# print("Translation vector (T):\n", T)
# color_new = np.empty(0)
# for each in imgpoints_color:
#     color_new = np.append(color_new, each, axis=0)
#
# infrared_new = np.empty(0)
# for each in imgpoints_infrared:
#     infrared_new = np.append(infrared_new, each, axis=0)

colornp = np.array(imgpoints_color)
reshaped_color = colornp.reshape(-1, 2)

infrarednp = np.array(imgpoints_infrared)
reshaped_infrared = infrarednp.reshape(-1, 2)

#print(imgpoints_color[0])
H, status = cv2.findHomography(reshaped_color, reshaped_infrared, method=0) #, 5.0
#print(status)
#print(H)
print(cv2.perspectiveTransform(np.array([[(263, 87)]], dtype=np.float32), H))
