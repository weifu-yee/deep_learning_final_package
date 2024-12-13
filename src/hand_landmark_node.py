#!/usr/bin/env python3

import rospy
import cv2
import mediapipe as mp
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np

class HandLandmarkNode:
    def __init__(self):
        # Initialize the node
        rospy.init_node("hand_landmark_detection_node")

        # Setup MediaPipe Hand
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # ROS Image and bridge setup
        self.bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)

        # Publisher for annotated output
        self.detection_pub = rospy.Publisher("/camera/color/image_handlandmark/compressed", CompressedImage, queue_size=1)

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # horizontal flip
        frame = cv2.flip(frame, 1)

        # Process image for hand landmarks
        self.detect_hand_landmarks(frame)

    def detect_hand_landmarks(self, frame):
        # Convert image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe hands
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Convert annotated frame to ROS message and publish
        self.publish_detection(frame)

    def publish_detection(self, annotated_image):
        # Convert the image to CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', annotated_image)[1]).tobytes()

        # Publish the compressed annotated image
        self.detection_pub.publish(msg)


if __name__ == "__main__":
    node = HandLandmarkNode()
    rospy.spin()
