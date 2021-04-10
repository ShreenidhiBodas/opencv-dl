import dlib
import imutils
import cv2
import numpy as np
import argparse
from imutils.video import VideoStream
from imutils import face_utils

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required=True, help="papth to pretrained shape predictor")
ap.add_argument('-v', '--video', help="[optional] path to video file")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Face #{}".format(i+1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        for (x,y) in shape:
            cv2.circle(frame, (x,y), 2, (0,0,255),-1)
    
    cv2.imshow("Image", frame)
        
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

if not args.get("video", False):
    vs.stop()

else:
    vs.release()

cv2.destroyAllWindows()
