import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt*


cap = cv2.VideoCapture('test2.mp4')
count = 0 
fFeatures_list = []

def movements(flist):
	nose_bridge = []
	nose_bridgeM = []
	left_eye = []
	left_eyeM = [] 
	nose_tip= []
	nose_tipM = []
	chin = []
	chinM = [] 
	right_eye = [] 
	right_eyeM = [] 
	left_eyebrow = [] 
	left_eyebrowM = [] 
	bottom_lip = []
	bottom_lipM = []
	right_eyebrow = []
	right_eyebrowM = []
	top_lip = []
	top_lipM = [] 

	for i in range(len(flist)):

		nose_bridge.append(flist[i][0]) 
		left_eye.append(flist[i][1])
		nose_tip.append(flist[i][2])
		chin.append(flist[i][3])
		right_eye.append(flist[i][4])
		left_eyebrow.append(flist[i][5])
		bottom_lip.append(flist[i][6])
		right_eyebrow.append(flist[i][7])
		top_lip.append(flist[i][8])
		

	nose_bridgeM  = frameDiff(nose_bridge , "nose_bridge") 
	left_eyeM     = frameDiff(left_eye,"left_eye" ) 
	nose_tipM     = frameDiff(nose_tip,"nose_tip") 
	right_eyeM    = frameDiff(right_eye,"right_eye") 
	chinM         = frameDiff(chin,"chin") 
	left_eyebrowM = frameDiff(left_eyebrow,"left_eyebrow") 
	bottom_lipM   = frameDiff(bottom_lip,"bottom_lip") 
	right_eyebrow = frameDiff(right_eyebrow,"right_eyebrow") 
	top_lipM      = frameDiff(top_lip,"top_lip")

	plt.plot(np.add(left_eyeM,right_eyeM),color='b', linewidth=1)		
	plt.show()


def frameDiff(apList,string):
	diff = []
	print "calculating frame wise difference for " + string
	for i in range(len(apList) - 1):
		
		prevapList = np.array(apList[i])
		prev_x = np.array(prevapList[:][:,0])
		prev_y = np.array(prevapList[:][:,1])
		
		nextapList = np.array(apList[i+1])
		next_x = np.array(nextapList[:][:,0])
		next_y = np.array(nextapList[:][:,1])	
		
		d = np.sqrt(np.square(next_x - prev_x) + np.square(next_y - prev_y))
		sUm = np.sum(d)			
		diff.append(sUm)
	return diff


while(cap.isOpened()):

	ret, frame = cap.read()
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
	# frame[:,:,0] = cv2.equalizeHist(frame[:,:,0])
	# frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
	
	face_landmarks_list = face_recognition.face_landmarks(frame)
	cv2.imshow("frame",frame)

	if len(face_landmarks_list) > 0:
		count = count + 1 
		print 'count of faces ',count 
		print "landmarks detected. appending list to 'fFeatures_list' "
		fFeatures_list.append(face_landmarks_list[0].values())

	if cv2.waitKey(1) & 0xFF == ord('q'):
		if len(fFeatures_list) == 0:
			break
		movements(fFeatures_list)
		break

cap.release()
cv2.destroyAllWindows()

