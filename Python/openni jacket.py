from openni import openni2
import sys

# Initialize OpenNI2
try:
    openni2.initialize()  # can also be given a path to the OpenNI2 drivers
except openni2.OpenNIError as e:
    print(f"Failed to initialize OpenNI: {e}")
    sys.exit(1)

print(openni2.get_version())
dev = openni2.Device.open_any()
# Check for connected devices
#devices = openni2.enumerateDevices()
if len(devices) == 0:
    print("No device found, exiting.")
    openni2.shutdown()
    sys.exit()

print(f"Found devices: {devices}")

# Open the first available device
device = None
try:
    device = openni2.Device.open_any()
    print(f"Opened device: {device.get_device_info()}")

    # --- This is the key check ---
    is_supported = device.is_image_registration_mode_supported(
        openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
    )

    # Report the result
    if is_supported:
        print("\n✅ Success! This device supports image registration (Depth to Color).")
    else:
        print("\n❌ This device does not support image registration (Depth to Color).")

finally:
    # Clean up
    if device:
        device.close()
    openni2.shutdown()
    print("\nOpenNI shut down.")