# Motion Light

<p>This project allows users to interact with an addressable LED strip using their hands, which reacts to hand position along the strip
and changes intensity as the hand gets closer.</p>

[![Motion Light in use](Images/Motion%20Light%20Main.jpg)](https://youtu.be/SaaqHdl8EyQ)
*Click the image to see it in action!*


## Webcam Version
The webcam version should be relatively straightforward, and run on any PC that has access to a webcam (built-in or plugged in). 
Simply run the [Motion_Light_Webcam](Python/Motion_Light_Webcam.py) file, modify some of the fields as commented, and then run the program.

## Kinect Version

To buy a kinect I recommend searching thrift stores as many people have given them away, I found my kinect v2 at a goodwill for less than ten dollars (a kinect v1 may work, but is untested).
To get it connected to my computer I followed [this guide](https://www.youtube.com/watch?v=l0rWWT24TNE&t=379s) to mod it to use a standard USB type B cable and
a 12v power supply. For an easier approach a converter such as [this](https://www.amazon.com/Xbox-Kinect-Adapter-One-Windows-10/dp/B01GVE4YB4?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A3LUVL9PHBMIC1&gQT=2)
can be found online for anywhere from \$20 to \$50. 

![Power supply mod](Images/Kinect%20Mod%20Soldering.jpg)
*Soldering connections on the kinect*

![Plastic shell mod](Images/Kinect%20Mod%20Plastic.jpg)
*Backside of the rear plastic*

![Rear connecter mod](Images/Kinect%20Mod%20Rear.jpg)
*Rear connectors after mod, USB type B on top and 12v barrel jack on bottom*

To get your computer to recognize the kinect, download the SDK [here](https://www.microsoft.com/en-us/download/details.aspx?id=44561) which includes drivers and testing software.
You can verify it is set up correctly by running the Color Basics sample in the SDK Browser, and you should see the video feed from the kinect.

The [Motion_Light_Kinect](Python/Motion_Light_Kinect.py) file offers two video modes, I find that the infrared mode works best for most use cases due to is responsiveness and stability compared to color mode. 
The color mode may work better in cases where granularity is more important.</p>

![Kinect infrared mode](Images/Kinect%20Infrared%20Mode.png)
*Kinect version running in infrared mode*

## Hardware Setup

I used an ESP8266 dev board running [WLED](https://kno.wled.ge/basics/getting-started) connected to a WS2812B LED strip, and supplied both with 5V. 
Since the communication happens over wifi there is no need for extra cables connecting the computer to the ESP. I used pin D1 on the board,
which is GPIO pin 5 in the WLED software. It is best practice to put a logic level shifter between the ESP and the LED strip (since the ESP
used 3.3v and the LED strip 5v), but since the cable I used was so short I did not have communication issues.

![Motion Light Hardware](Images/Motion%20Light%20Hardware.jpg)
*Hardware setup on breadboard*

![Entire setup](Images/Motion%20Light%20Setup.jpg)
*The entire Motion Light system setup*

## Next Steps

I am quite happy with the progress so far, moving forward I hope to delve into more of the kinect sensors (such as depth) to make even better
interactions possible.

