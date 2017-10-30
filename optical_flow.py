from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import scipy 
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()             #detect face using Dlib's classifier                          
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")    #detect landmarks 

if args.get("video", None) is None:
    cap = cv2.VideoCapture(0)
    time.sleep(0.25)
 
else:
    cap = cv2.VideoCapture(args["video"])

shape_list = [] 
shape = []
no_of_frames = 0
count = 0 
patchSize = 10            #patch size over which the pixel values of facial landmarks are averaged out 

def  movements(shapes):
#--------------------define lists to store values of movements of individual expressions-------------------
	shape_diff_list = []
	diff = [] 
	dLeye = []
	dReye = []
	dVnose = [] 
	dHnose = []
	dReyebrows= [] 
	dLeyebrows = []
	dLip = []
	dOutline = []

	for i in range(len(shape_list)-1):
		prev_x =  np.array(shape_list[i][:,0])
		prev_y =  np.array(shape_list[i][:,1])
		next_x =  np.array(shape_list[i+1][:,0])
		next_y =  np.array(shape_list[i+1][:,1])
		d = np.sqrt(np.square(next_x - prev_x) + np.square(next_y - prev_y))		
		
		dLeye.append(np.sum(d[36:42]))
		dReye.append(np.sum(d[42:48]))
		dVnose.append(np.sum(d[27:31]))
		dHnose.append(np.sum(d[31:36]))
		dLeyebrows.append(np.sum(d[17:22]))
		dReyebrows.append(np.sum(d[22:27]))
		dLip.append(np.sum(d[48:68]))
		dOutline.append(np.sum(d[0:17]))		
		d_sum  = np.sum(d)
		diff.append(d_sum)
	
	dLip = scipy.signal.medfilt(dLip)				# applying median filter to remove noise and preserve edges 
	plt.plot(dLip,color='b', linewidth=1)		
	plt.show()
		
while (cap.isOpened()):
	(grabbed, frame) = cap.read()
	if grabbed == False:							#solved the error of failure of camera
		break

	frame = imutils.resize(frame, width=400)
	no_of_frames = no_of_frames + 1
	print "no_of_frames : " , no_of_frames				

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)					# contrast stretching using histogram equalization 	
	
	rects = detector(gray, 0)						#detect face 
	for rect in rects:
		shape = predictor(gray, rect)				#detect facial landmarks
		shape = face_utils.shape_to_np(shape)

	if len(shape) == 0: 							#check whether face is detected or not to avoid empty lists in the final collection
		print "face not detected. Exiting.... "
		break

	shape_list.append(shape)     
	
	for (x, y) in shape:							# draw circles at the detected landmark points	
		cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	cv2.imshow("Frame", frame)
	
	key = cv2.waitKey(1) & 0xFF
	 
	if key == ord("q"):							#press 'q' to extract movement features						
		print "frames_captured : " , len(shape_list)
		face_landmarks = []
		for i in range(len(shape_list)%patchSize):	 #average to filter out noise 											
			sum_of_patch = np.zeros(shape_list[0].shape)
			for j in range(patchSize):
				if i*patchSize + j < len(shape_list):
					sum_of_patch = np.add(sum_of_patch,shape_list[ i*patchSize + j ])
			sum_of_patch = sum_of_patch/patchSize
			face_landmarks.append(sum_of_patch)	
		movements(face_landmarks)
		break

cap.release()
cv2.destroyAllWindows()