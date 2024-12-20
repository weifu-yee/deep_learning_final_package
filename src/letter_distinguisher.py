#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import String

# Initialize the CvBridge
bridge = CvBridge()

# PyTorch model architecture as provided:
class CNNModel(nn.Module):
    def __init__(self, num_classes=26):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.4)

        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)
        self.bn7 = nn.BatchNorm2d(128)

        self.drop3 = nn.Dropout(0.4)
        # Final output is (128, 1, 1), hence:
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # -> (32, 26, 26)
        x = F.relu(self.bn2(self.conv2(x)))   # -> (32, 24, 24)
        x = F.relu(self.bn3(self.conv3(x)))   # -> (32, 12, 12)
        x = self.drop1(x)
        x = F.relu(self.bn4(self.conv4(x)))   # -> (64, 10, 10)
        x = F.relu(self.bn5(self.conv5(x)))   # -> (64, 8, 8)
        x = F.relu(self.bn6(self.conv6(x)))   # -> (64, 4, 4)
        x = self.drop2(x)
        x = F.relu(self.bn7(self.conv7(x)))   # -> (128, 1, 1)
        x = x.view(x.size(0), -1)             # -> (batch_size, 128)
        x = self.drop3(x)
        x = self.fc(x)                        # -> (batch_size, num_classes)
        return x

def decode_label(label_index):
    # 0 -> A, 1 -> B, ... 25 -> Z
    return chr(label_index + 65)

def preprocess_and_crop_image_from_ros_msg(ros_img_msg, margin=5):
    # Convert the ROS Image message to a numpy array (grayscale)
    try:
        img = bridge.imgmsg_to_cv2(ros_img_msg, "mono8")
    except Exception as e:
        rospy.logerr("Error converting ROS Image message to CV2: %s", e)
        return None

    # Convert to "white on black" by inverting pixels where letters are present
    binary = img < 255
    img = np.where(binary, 255 - img, 0).astype(np.uint8)

    coords = np.argwhere(binary)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(img.shape[0], y_max + margin)
    x_max = min(img.shape[1], x_max + margin)

    # Crop the image based on computed coordinates
    cropped_img = PILImage.fromarray(img).crop((x_min, y_min, x_max, y_max))
    resized_img = cropped_img.resize((28, 28))

    processed_img = np.array(resized_img).astype(np.float32) / 255.0

    # Display the processed image (optional)
    # plt.imshow(processed_img, cmap="gray")
    # plt.axis("off")
    # plt.title("Cropped and Resized Image")
    # plt.show()

    return processed_img

# Create model and load weights
model_path = "/home/hrc/Deep_learning_d415/catkin_ws/src/final/model/model_colab_char.pth"  # Path to your saved PyTorch model weights
model = CNNModel(num_classes=26)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# def trajectory_callback(msg):
#     # This is where you can process the incoming message
#     rospy.loginfo("Received hand trajectory message")
    
#     processed_img = preprocess_and_crop_image_from_ros_msg(msg)

#     if processed_img is not None:
#         # Do something with the processed image (e.g., further processing or inference)
#         rospy.loginfo("Image processed successfully")
#     # arr = preprocess_and_crop_image(img_path)

#     # PyTorch input: channels_first
#     torch_input = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 28, 28)
#     with torch.no_grad():
#         torch_pred = model(torch_input)
#     torch_label = torch.argmax(torch_pred, dim=1).item()

#     # # print(f"Image: {img_path}")
#     print(f"  PyTorch Prediction: {decode_label(torch_label)} (index: {torch_label})")
#     # print("-" * 50)

def trajectory_callback(msg):
    rospy.loginfo("Received hand trajectory message")
    
    processed_img = preprocess_and_crop_image_from_ros_msg(msg)

    if processed_img is not None:
        rospy.loginfo("Image processed successfully")
        # PyTorch input: channels_first
        torch_input = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 28, 28)
        with torch.no_grad():
            torch_pred = model(torch_input)
        torch_label = torch.argmax(torch_pred, dim=1).item()

        # Decode label and publish it
        letter = decode_label(torch_label)
        rospy.loginfo(f"Predicted letter: {letter}")
        letter_pub.publish(letter)

def listener():
    global letter_pub

    # Initialize the node
    rospy.init_node('letter_distinguisher', anonymous=True)

    # Create a publisher for the /hand_trajectory/letter topic
    letter_pub = rospy.Publisher('/hand_trajectory/letter', String, queue_size=10)

    # Subscribe to the /hand_trajectory/trajectory topic
    rospy.Subscriber("/hand_trajectory/trajectory", Image, trajectory_callback)

    # Spin to keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass