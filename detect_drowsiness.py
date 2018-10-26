from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
#from playsound import playsound
import argparse
import imutils
import time
import dlib 
import cv2
import urllib


def sound_alarm(path):
	
	playsound.playsound(path)
	#playsound("home/ghosthapper/Desktop/Desktop/python/new/alert.mp3")


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	
	A = dist.euclidean(eye[1], eye[5]) # vertical eye landmarks (x, y)-coordinates
	B = dist.euclidean(eye[2], eye[4])

	
	
	C = dist.euclidean(eye[0], eye[3]) # compute the euclidean distance between the horizontal

	
	ear = (A + B) / (2.0 * C) # compute the eye aspect ratio

	
	return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="home/ghosthapper/Desktop/Desktop/python/new/shape_predictor_68_face_landmarks.dat")
ap.add_argument("-a", "--alarm", type=str, default=0,
	help="home/ghosthapper/Desktop/Desktop/python/new/alert.mp3")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="http://192.168.1.100:4747/")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

COUNTER = 0
ALARM_ON = False



print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector() # initialize dlib's face detector (HOG-based) and then create
predictor = dlib.shape_predictor(args["shape_predictor"]) # the facial landmark predictor

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(0.5)


while True:
	

	frame = vs.read()
	frame = imutils.resize(frame, width=860)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # it, and convert it to grayscale

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	for rect in rects:
		 

		 
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array
 
		 
		 
		leftEye = shape[lStart:lEnd] # extract the left and right eye coordinates, then use the
		rightEye = shape[rStart:rEnd] # coordinates to compute the eye aspect ratio for both eyes
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
 
		 # average 
		ear = (leftEAR + rightEAR) / 2.0

	     # compute the convex hull for the left and right eye, then
		 # visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


	     
		 
		if ear < EYE_AR_THRESH:
			COUNTER += 1
 
			 
			 
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
			    
				if not ALARM_ON:
					ALARM_ON = True
 
					
					
					
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()
 
				 
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
		 
		 
	        else:
		    	COUNTER = 0
			ALARM_ON = False
	
	    		
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 	 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
  
	if key == ord("q"):
		break
 
cv2.destroyAllWindows()
vs.stop()
