import cv2
from ultralytics import YOLO
import numpy as np
import time
from centroidtracker import CentroidTracker
from collections import defaultdict

MIN_MOVEMENT_THRESH = 10
ALERT_SECONDS = 10

coco_model = YOLO('yolov8x.pt')
tracker = CentroidTracker(maxDisappeared=50, maxDistance = 90)
centroid_dict = defaultdict(list)
object_id_list = []
person = [0]
cap = cv2.VideoCapture('./testun.mp4')
prev_box = None
start_time = None
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vps = cap.get(cv2.CAP_PROP_FPS)
widht = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./final1.mp4', fourcc, vps, (widht, height))
elapsed = 1
bags = [24,26,28]
ret = True
frame_nmr = -1
while ret:
    frame_nmr =+ 1
    ret,frame = cap.read()
    detections = coco_model(frame)[0]
    detections_ = []
    rects = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        box=[x1,y1,x2,y2]
        if int(class_id) in bags:
            detections_.append([x1, y1, x1+x2, y1+y2])
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),5)
            if elapsed>30:
                val = 0
                val = int(elapsed)-30
                cv2.putText(frame, str(int(val)), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif int(class_id) in person:
            rects.append(box)

            objects = tracker.update(rects)
            for (objectId, bbox) in objects.items():
                x3, y3, x4, y4 = bbox
                x3 = int(x3)
                y3 = int(y3)
                x4 = int(x4)
                y4 = int(y4)
                cx = int((x3 + x4) / 2.0)
                cy = int((y3 + y4) / 2.0)
                cv2.circle(frame, (cx, cy),4,(0,255,0),-1)

                centroid_dict[objectId].append((cx,cy))
                if objectId not in object_id_list:
                    object_id_list.append(objectId)
                    start_pt = (cx,cy)
                    end_pt = (cx,cy)
                    cv2.line(frame, start_pt, end_pt,(0,255,0),2)
                else:
                    l = len(centroid_dict[objectId])
                    for pt in range(len(centroid_dict[objectId])):
                        if not pt + 1 == l:
                            start_pt = (centroid_dict[objectId][pt][0],centroid_dict[objectId][pt][1])
                            end_pt = (centroid_dict[objectId][pt+1][0],centroid_dict[objectId][pt+1][1])
                            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

                cv2.rectangle(frame, (x3, y3),(x4, y4),(0,0,255),2)
                text = "ID: {}".format(objectId)
                cv2.putText(frame,text, (x3,y3-5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,255),1)

    if prev_box is not None:
        movement = abs(box[0] - prev_box[0]) + abs(box[1] - prev_box[1])
        if movement < MIN_MOVEMENT_THRESH:
            if start_time is None:
                start_time = time.time()

            elapsed = time.time() - start_time
            elapsed = elapsed/10
            if elapsed >= ALERT_SECONDS:
                print("Object stationary for too long!")


    prev_box = box

    out.write(frame)


cap.release()
out.release()
