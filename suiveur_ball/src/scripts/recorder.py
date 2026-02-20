#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
# Create a VideoCapture object

class Recorder:

    def camera_callback(self, img):
        rospy.loginfo("Image reçu")
        image = self.bridge.compressed_imgmsg_to_cv2(img, desired_encoding='bgr8')
        image = cv2.resize(image, [128, 128], cv2.INTER_AREA)
        self.out.write(image)

    def __init__(self):
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.camera_callback)
        self.out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (128, 128))
    def stop(self):
        print("shutdown time!")
        self.out.release()

if __name__ == '__main__':
    rospy.init_node('recorder_node')
    recorder = Recorder()
    rospy.on_shutdown(recorder.stop)
    rospy.spin()