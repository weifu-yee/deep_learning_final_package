#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import cv2
import mediapipe as mp
import numpy as np  # Ensure numpy is imported

class HandTrajectoryNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('hand_trajectory_node', anonymous=True)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize trajectory points list
        self.path_points = []

        # Initialize the trajectory points to be published
        self.path_points_pub = []

        # Initialize the timestamp for the last detected left hand
        self.last_left_hand_time = rospy.Time.now()
        self.last_right_hand_time = rospy.Time.now()

        # Subscriber to the compressed image_handlandmark topic
        self.image_sub = rospy.Subscriber(
            # '/camera/color/image_handlandmark/compressed',
            '/camera/color/image_raw/compressed',
            CompressedImage,
            self.image_callback
        )

        # Publisher for the processed image (uncompressed)
        self.image_pub = rospy.Publisher(
            '/camera/color/image_handlandmark_processed',
            Image,
            queue_size=10
        )

        # Publisher for the trajectory image with white background and black trajectory
        self.trajectory_pub = rospy.Publisher(
            '/camera/color/image_handlandmark_trajectory',
            Image,
            queue_size=10
        )

        rospy.loginfo("Hand Trajectory Node Initialized.")
        rospy.loginfo("Subscribed to: /camera/color/image_handlandmark/compressed")
        rospy.loginfo("Publishing to: /camera/color/image_handlandmark_processed")
    
    def publish_detection(self, annotated_image):
        try:
            # Convert OpenCV image back to ROS Image message
            processed_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            # Publish the processed image
            self.image_pub.publish(processed_image_msg)
            rospy.loginfo_throttle(5, "Published processed image.")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
            
    def publish_trajectory(self, trajectory_image):
        if trajectory_image is None:
            return
        try:
            # Convert OpenCV image back to ROS Image message
            processed_image_msg = self.bridge.cv2_to_imgmsg(trajectory_image, encoding='bgr8')
            # Publish the processed image
            self.trajectory_pub.publish(processed_image_msg)
            rospy.loginfo_throttle(5, "Published processed image.")
        except CvBridgeError as e:
            return

    def image_callback(self, msg):
        rospy.loginfo_throttle(5, "Image received.")

        try:
            # Convert ROS CompressedImage message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # horizontal flip
            cv_image = cv2.flip(cv_image, 1)

            if cv_image is None:
                rospy.logerr("Failed to decode compressed image.")
                return
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        except Exception as e:
            rospy.logerr(f"Unexpected error during image decoding: {e}")
            return

        # Get image dimensions
        image_height, image_width, _ = cv_image.shape

        # Convert the image from BGR to RGB as MediaPipe uses RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = self.hands.process(rgb_image)

        # Flags to check for left and right hands
        left_hand_present = False
        right_hand_present = False
        right_index_fingertip = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get the label of the hand (Left or Right)
                hand_label = handedness.classification[0].label

                # Draw hand landmarks on the image
                self.mp_drawing.draw_landmarks(cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if hand_label == 'Left':
                    left_hand_present = True
                    # Update the timestamp since left hand is detected
                    self.last_left_hand_time = rospy.Time.now()
                elif hand_label == 'Right':
                    right_hand_present = True
                    # Get index finger tip landmark (id 8)
                    index_finger_tip = hand_landmarks.landmark[8]
                    x, y = index_finger_tip.x, index_finger_tip.y
                    # Convert normalized coordinates to pixel coordinates
                    pixel_x, pixel_y = int(x * image_width), int(y * image_height)
                    right_index_fingertip = (pixel_x, pixel_y)
                    # Update the timestamp since right hand is detected
                    self.last_right_hand_time = rospy.Time.now()


        # Calculate time difference since last hand detection
        current_time = rospy.Time.now()
        time_diff_left = current_time - self.last_left_hand_time
        time_diff_right = current_time - self.last_right_hand_time

        # action based on hand presence
        break_line = (2*image_width, 2*image_height)
        if time_diff_right.to_sec() > 0.1:
            self.path_points.clear()
            rospy.loginfo_throttle(5, "Cleared trajectory points due to absence of right hand.")
            self.path_points_pub = self.path_points

        elif time_diff_left.to_sec() > 0.1:
            rospy.loginfo_throttle(5, "Paused tracking due to presence of left hand.")
            self.path_points.append(break_line)

        else:
            if right_hand_present:
                self.path_points.append(right_index_fingertip)
            # Optionally, limit the number of points to prevent memory issues
            if len(self.path_points) > 200:
                self.path_points.pop(0)

        # Draw the trajectory on the image
        for i in range(1, len(self.path_points)):
            if self.path_points[i] == break_line or self.path_points[i - 1] == break_line:
                continue
            cv2.line(cv_image, self.path_points[i - 1], self.path_points[i], (255, 0, 0), 2)
        
        trajectory_image = np.ones_like(cv_image) * 255
        for i in range(1, len(self.path_points_pub)):
            if self.path_points_pub[i] == break_line or self.path_points_pub[i - 1] == break_line:
                continue
            cv2.line(trajectory_image, self.path_points_pub[i - 1], self.path_points_pub[i], (255, 0, 0), 2)
        

        # Publish the processed image
        self.publish_detection(cv_image)
        self.publish_trajectory(trajectory_image)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = HandTrajectoryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
