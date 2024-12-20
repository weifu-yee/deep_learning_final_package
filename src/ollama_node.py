#!/usr/bin/env python3

import rospy
import ollama
from std_msgs.msg import String

# Define the system message
# SYSTEM_MESSAGE = """
# this is a Capital Alphabet learning system, the input will be an alphabet, and please generate one volcabulary that start with this Letter.
# """

# # # Example feedback
# EXAMPLE_USER_MESSAGE = "This is the student’s attempt to write the letter 'E'. Please generate one simple volcabulary for children that start with this letter."
# EXAMPLE_ASSISTANT_MESSAGE = (
#     "Eagle (noun.)"
# )
# Define the system message
SYSTEM_MESSAGE = """
This is a Capital Alphabet learning system, the input will be an alphabet letter, and please generate a sentence with volcabulary that start with this Letter.
"""

# # Example feedback
EXAMPLE_USER_MESSAGE = "This is the student’s attempt to write the letter 'E'. Please generate one simple sentence with volcabulary for children that start with this letter."
EXAMPLE_ASSISTANT_MESSAGE = (
    "I saw an Eagle flying in the sky."
)

# Global variable to store the last received letter
last_letter = None

def letter_callback(msg):
    global last_letter

    # The letter received from the topic
    letter = msg.data

    # Check if the received letter is the same as the last one
    if letter == last_letter:
        # rospy.loginfo("Received the same letter as last time. Skipping processing.")
        return  # Skip processing

    # Update the last received letter
    last_letter = letter

    rospy.loginfo(f"Received letter: {letter}")

    # Generate feedback using Ollama
    response = ollama.chat(
        model='llava-phi3',
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": EXAMPLE_USER_MESSAGE},
            {"role": "assistant", "content": EXAMPLE_ASSISTANT_MESSAGE},
            #{"role": "user", "content": f"This is the student’s attempt to write the letter '{letter}'.Please generate one simple volcabulary for children that start with this letter."}
            {"role": "user", "content": f"This is the student’s attempt to write the letter '{letter}'.Please generate one simple sentence with volcabulary for children that start with this letter."}
        ]
    )

    # Extract and log the feedback
    feedback = response['message']['content']
    rospy.loginfo(f"Generated feedback: {feedback}")

    # Publish the feedback to the /feedback topic
    feedback_pub.publish(feedback)

# Main function
if __name__ == '__main__':
    try:
        # Initialize the ROS node
        rospy.init_node('hand_ollama_node')

        # Subscriber to the /student_letter topic
        rospy.Subscriber('/hand_trajectory/letter', String, letter_callback)

        # Publisher to the /feedback topic
        feedback_pub = rospy.Publisher('/hand_trajectory/feedback', String, queue_size=10)

        rospy.loginfo("Ollama ROS node is running and waiting for letters...")

        # Keep the node running
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Ollama ROS node terminated.")
