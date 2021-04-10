import cv2
import numpy as np
import imutils
import imutils.contours
import dlib
from imutils import face_utils
from imutils.video import VideoStream
import argparse
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[0], eye[3])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[1], eye[5])

    ear = (B + C) / (2.0 * A)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help="path to shape predictor")
ap.add_argument('-v', '--video', help="[optional] path to video file")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()

else:
    vs = cv2.VideoCapture(args["video"])

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 2
COUNTER = 0
TOTAL = 0

(rl, rr) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(ll, lr) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[ll:lr]
        rightEye = shape[rl:rr]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0,255,0), 2)
        cv2.drawContours(frame, [rightHull], -1, (0,255,0), 2)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
        
        else:
            if COUNTER > EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            
            COUNTER = 0
        
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 3)
        cv2.putText(frame, "EAR: {}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 3)
    
    cv2.imshow("Image", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
