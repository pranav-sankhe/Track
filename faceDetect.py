import face_recognition
import os
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import argparse
import time
import imutils
import matplotlib.pyplot as plt

sess = tf.Session() 
my_head_pose_estimator = CnnHeadPoseEstimator(sess) 
roll_list = []
pitch_list =[]
yaw_list = []

my_head_pose_estimator.load_roll_variables(os.path.realpath("./weights/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("./weights/pitch/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_yaw_variables(os.path.realpath("./weights/yaw/cnn_cccdd_30k"))

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	cap = cv2.VideoCapture(0)                      #index value needs to be changed accordingly
	time.sleep(0.25)

else:
    cap = cv2.VideoCapture(args["video"])


def plot_data():
	
	plt.plot(roll_list,color='b', linewidth=1)		
	plt.plot(pitch_list,color='r', linewidth=1)		
	plt.plot(yaw_list,color='g', linewidth=1)		
	plt.show()

while(1):
	
	ret, image = cap.read()
	cv2.imshow("video",image)
	# image = face_recognition.load_image_file("test.jpg")
	face_locations = face_recognition.face_locations(image)

	if len(face_locations) == 0:
		print "face not detected"
	
	if(len(face_locations) > 0):

		x= face_locations[0][0]
		w= face_locations[0][1]
		h= face_locations[0][2]
		y= face_locations[0][3]
		df = image[x:x+w, y:y+h]
		height= 400
		width = 400
		df = cv2.resize(df,(width, height), interpolation = cv2.INTER_CUBIC)
		print "face detected"
		cv2.imshow("face",df)
		
		roll = my_head_pose_estimator.return_roll(df)  # Evaluate the roll angle using a CNN
		pitch = my_head_pose_estimator.return_pitch(df) # Evaluate the pitch angle using a CNN
		yaw = my_head_pose_estimator.return_yaw(df)  # Evaluate the yaw angle using a CNN
		
		roll_list.append(roll[0,0,0])
		pitch_list.append(pitch[0,0,0])
		yaw_list.append(yaw[0,0,0])

		print("Estimated [roll, pitch, yaw] ..... [" + str(roll[0,0,0]) + "," + str(pitch[0,0,0]) + "," + str(yaw[0,0,0])  + "]")


		
	if cv2.waitKey(1) & 0xFF == ord('q'):
		plot_data()
		break

# When everything done, release the capture
cv2.destroyAllWindows()