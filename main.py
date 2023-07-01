import numpy as np
import math
import json
from FindRepeatingMotion import FindRepeatingMotion
import cv2

lk_motions = np.load('motions/barbell_lk.npy',allow_pickle=True)
print('here')
print(lk_motions.shape)
lkdata = {}
for i in range(len(lk_motions)):
    for j in range(len(lk_motions[0])):
        if i == 0:
            lkdata[str(j)] = [lk_motions[i,j].tolist()]
        else:
            lkdata[str(j)].append(lk_motions[i,j].tolist())
print('lk_data',lkdata['0'])


f = open('mpposes.json')
mpdata = json.load(f)
#mp_times = FindRepeatingMotion(mpdata)
lk_times = FindRepeatingMotion(lkdata)
vid_path = "videos/barbellExpert.mp4"
#display times
cap = cv2.VideoCapture(vid_path)

fps = cap.get(5)
print('Frames per second : ', fps,'FPS')
frame_count = cap.get(7)
print('Frame count : ', frame_count)

workout = [lk_times[0],lk_times[2]]
for i in range(int(frame_count)):
    ret, img = cap.read()
    if i >= workout[0]:
        cv2.imshow('Workout',img)
        cv2.waitKey(1)
        if i == workout[1]:
            break
