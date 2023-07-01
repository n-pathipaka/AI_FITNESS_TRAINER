import cv2
import json
import PoseModule as pm
import numpy as np
from time import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# Radius of circle
radius = 20
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 1


def extractMotionSequence(vid_path,motion_path, debug=False):
    cap = cv2.VideoCapture(vid_path)
    detector = pm.poseDetector()

    fps = cap.get(5)
    print('Frames per second : ', fps,'FPS')
    frame_count = cap.get(7)
    print('Frame count : ', frame_count)

    poses = {}
    starttime = time()
    for i in tqdm(range(int(frame_count)), desc="Processing..."):
        ret, img = cap.read()
        if i%5==0:
            if ret == True:
                img = detector.findPose(img, False)
                if img is not None:
                    lmList = np.array(detector.findPosition(img, False))
                    if len(lmList) != 33:
                        print('wrong length')
                    if debug == True:
                        for p in lmList:
                            cv2.circle(img,(int(p[1]),int(p[2])),
                            radius,color,thickness)
                            cv2.putText(img, str(int(p[0])), (int(p[1]),int(p[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (246,255,12), 1)
                        cv2.imshow('image',img)
                        #cv2.waitKey(1)
                        cv2.imwrite('debug/mp'+str(i)+'.png',img)

                    for pnt in lmList:
                        coor_time = [int(pnt[1]),int(pnt[2]),i]
                        if int(pnt[0]) in poses:
                            poses[int(pnt[0])].append(coor_time)
                        else:
                            poses[int(pnt[0])] = [coor_time]
                    #frame_num = np.reshape(frame_num,(33,1))
                    #lmList = np.hstack([lmList,frame_num])
                    #poses.append(lmList)

    endtime = time()
    print(endtime,starttime)
    print('Time taken: %f seconds',endtime-starttime)
    json_poses = json.dumps(poses)
    f = open("mpposes.json","w")
    f.write(json_poses)
    f.close()
    return poses

def main():
    extractMotionSequence("videos/barbellExpert.mp4",'motions/barbell_expert.npy')
    f = open('mpposes.json')
    data = json.load(f)
    for n in data:
        print(len(data[n]))

main()
