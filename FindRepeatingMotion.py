import numpy as np
import math
import json

#for each point
#find local max and mins
#calc diff between local max and mins, and timeframe diffs

#Things to calculate
#number of frames with motion
#degree of displacement

def FindRepeatingMotion(data):
    #STEP 1 : FIND points with most movement
    movement_rank = []
    size = len(data['0'])
    for n in range(33):
        x = np.array([p[0] for p in data[str(n)]])
        y = np.array([p[1] for p in data[str(n)]])
        x_movement = np.linalg.norm(x[:size-1]-x[1:])
        y_movement = np.linalg.norm(y[:size-1]-y[1:])
        movement = x_movement+y_movement
        #print(movement)
        #choose points with which movement
        movement_rank.append([n,movement])


    movement_rank.sort(reverse=True, key=lambda x:x[1])
    print(movement_rank)
    dynamic_pnts = [movement_rank[i][0] for i in range(5)]
    print(dynamic_pnts)

    #STEP2: Calculate movement vectors at each timestep
    motion_tf = []
    for i in range(1):
        #take difference of each point at each time step as a vector
        entries = np.array(data[str(dynamic_pnts[i])])
        timeseries = entries[1:size, 0:2] - entries[0:size-1,0:2]
        print(timeseries.shape)
        curr_direction = ''
        direction_change = 0

    #STEP3: Calculate direction change
        dir_change = [0]
        for j in range(len(timeseries)-1):
            direction = np.dot(timeseries[j],timeseries[j+1])/(np.linalg.norm(timeseries[j])*np.linalg.norm(timeseries[j+1]))
            #print(direction)
            if direction <-0.7:
                change = 1
            else:
                change = 0
            dir_change.append(change)
        #print(dir_change)
        motion_tf.append(dir_change)

    #STEP4: Calculate velocity
        velocity = []
        low_velo = []
        for k in range(len(timeseries)):
            velo = np.linalg.norm(timeseries[k])/(data[str(dynamic_pnts[i])][k+1][2]-data[str(dynamic_pnts[i])][k][2])
            velocity.append(velo)
            if velo <=1:
                low_v = 1
            else:
                low_v = 0
            low_velo.append(low_v)
        #print(low_velo)
        motion_tf.append(low_velo)


    #STEP5: Extract timesteps that are likely key frames in workout
    motion_tf = np.array(motion_tf)
    motion_sum = np.sum(motion_tf, axis=0)
    #Cleaning: get rid of values less than 3
    for k in range(len(motion_sum)):
        if motion_sum[k]>0 and motion_sum[k]<1:
            motion_sum[k] = 0
    #Smoothing: set values to a positive number if the numbers before and after are positive
    old_motion = motion_sum.copy()
    for k in range(1,len(motion_sum)-1):
        if old_motion[k] == 0:
            if old_motion[k-1]>0 and old_motion[k+1]>0:
                motion_sum[k] = 1
    print(motion_sum)
    #identify stretches of positive values
    stretches = []
    stretches_start = -1
    stretch_length = 0
    for k in range(len(motion_sum)):
        if motion_sum[k]>0:
            stretch_length+=1
            if stretches_start == -1:
                stretches_start = k
        else:
            if stretch_length > 0:
                stretches.append([stretches_start,stretch_length])
                stretch_length = 0
                stretches_start = -1
    print(stretches)
    #Calculate central values
    central_vals = []
    for k in range(len(stretches)):
        c = int(stretches[k][0]+stretches[k][1]/2)
        central_vals.append(c)
    print(central_vals)
    #Set other values to 0
    for k in range(len(motion_sum)):
        if k not in central_vals:
            motion_sum[k] = 0
    print(motion_sum)
    #create sparse array of key frames [time,time]
    sparse_frames = []
    for k in range(len(motion_sum)):
        if motion_sum[k]>0:
            print(data['0'][k])
            sparse_frames.append(data['0'][k][2])
    print(sparse_frames)
    return sparse_frames

def main():
    f = open('mpposes.json')
    mpdata = json.load(f)
    FindRepeatingMotion(mpdata)

main()
