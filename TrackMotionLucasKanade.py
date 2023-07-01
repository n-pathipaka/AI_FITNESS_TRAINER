import cv2
import json
from tqdm import tqdm
import numpy as np
from skimage import color as clr

feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))

color = np.random.randint(0, 255, (100, 3))


def track_motion(vid_path):
    cap = cv2.VideoCapture(vid_path)

    fps = cap.get(5)
    print('Frames per second : ', fps,'FPS')
    frame_count = cap.get(7)
    print('Frame count : ', frame_count)

    # Take first frame and find corners in it
    p0 = None
    while (p0 is None):
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                                     **feature_params)
    #print(p0)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    motion_sequence = []
    for i in tqdm(range(int(frame_count)-1), desc="Processing..."):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame,
                                  cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        #print(p0)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                               frame_gray,
                                               p0, None,
                                               **lk_params)

        for j in range(len(p1)):
            if st[j][0] == 0:
                p1[j][0] = p0[j][0]

        if i%5==0:
            p0 = np.reshape(p0,(len(p0),2))
            frame_num = np.ones(len(p0))*i
            frame_num = np.reshape(frame_num,(len(p0),1))
            #print(p0.shape,frame_num.shape)
            #print(p0)
            p0 = np.hstack([p0,frame_num])
            print(p0)
            motion_sequence.append(p0)

        # Updating Previous frame and points
        old_gray = frame_gray.copy()
        p0 = p1.reshape(-1, 1, 2)

    return motion_sequence
def main():
    barbell_sequence_lk = track_motion("videos/barbellExpert.mp4")
    #print(barbell_sequence_lk)
    np.save('motions/barbell_lk.npy',np.array(barbell_sequence_lk))
    #lk_motions = np.load('motions/barbell_lk.npy',allow_pickle=True)
    #print(lk_motions)
'''
    cap = cv2.VideoCapture("videos/barbellExpert.mp4")
    frame_count = cap.get(7)
    print('Frame count : ', frame_count)
    lk_motions = np.load('motions/barbell_lk.npy',allow_pickle=True)
    # Radius of circle
    radius = 20

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    for i in range(int(frame_count)-1):
        ret, frame = cap.read()
        for j in range(len(lk_motions[0])):
            cv2.circle(frame,(int(lk_motions[0][j][0]),int(lk_motions[0][j][1])),
            radius,color,thickness)
        cv2.imshow('points',frame)
        cv2.waitKey(10)
'''
main()
