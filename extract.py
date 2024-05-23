from mvextractor.videocap import VideoCap
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib import animation

import scipy
import torch

import time
import os
import gc

bins_x = [i * 16 for i in range(14)]
bins_y = [i * 16 for i in range(14)]

def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame

def identify_block(motion_vector, frame_shape):
    # from copy import deepcopy
    mv_x_origin = motion_vector[:,3]
    mv_y_origin = motion_vector[:,4]

    mv_x_origin = torch.where((mv_x_origin < 0) | (mv_y_origin < 0), motion_vector[:,5], motion_vector[:,3])
    mv_y_origin = torch.where((mv_x_origin < 0) | (mv_y_origin < 0), motion_vector[:,6], motion_vector[:,4])

    H, xedges, yedges, binnumber = scipy.stats.binned_statistic_2d(
    mv_x_origin.cpu().numpy(), mv_y_origin.cpu().numpy(), None, 'count', expand_binnumbers=True, bins=[bins_x, bins_y])

    return torch.from_numpy(binnumber).cuda()

def export_motion_vectors(motion_vectors):
    mv_array = torch.zeros((224,224,6)).cuda()
    #pack motion vectors. Not space efficient (yet), but video convolution reduces the data to Tx14x14x6
    #mv
    if motion_vectors.shape[0] == 0:
        return mv_array
    block_idx = identify_block(motion_vectors, None)
    num_mvs = motion_vectors.shape[0]
    i = 0
    for mv in torch.split(motion_vectors, num_mvs):
        id_x = block_idx[0,i] - 1
        id_y = block_idx[1,i] - 1
        mv_array[16*id_x:16*(id_x+1), 16*id_y:16*(id_y+1), :] = mv[0, 3:9].reshape(-1)
        i += 1

    return mv_array

def decompress_intra(path, limit=16):
     with torch.no_grad():
        cap = VideoCap()
        success = cap.open(path)
        if not success:
            return None

        frames = None
        frame_ct = 0
        while success:
            success, frame, motion_vectors, frame_type, timestamp = cap.read()
            if not success or frame_ct > 224:
                break

            if frame_type == 'I':
                frame = torch.from_numpy(frame)
                frame = torch.dstack((frame[:, :, 2], frame[:, :, 1], frame[:, :, 0])).reshape(224, 224, 3)
                if frames is None:
                    frames = frame.reshape(1, 224, 224, 3)
                else:
                    frames = torch.cat((frames, frame.reshape(1,224,224,3)))
                frame_ct += 1
        frames = frames[:limit, :, :, :]
        #print(frames.shape[0])
        while frames.shape[0] < limit:
            frames = torch.cat((frames, torch.zeros(1,224,224,3)))
        
        frames = frames.permute(0, 3, 1, 2)
        return frames

def decompress(path):
    with torch.no_grad():
        cap = VideoCap()
        success = cap.open(path)
        if not success:
            return

        mvs = None
        init_frame = None
        last_frame = None
        frame_ct = 0
        while success:
            success, frame, motion_vectors, frame_type, timestamp = cap.read()
            if not success or frame_ct > 224:
                break
            if frame_type == 'P':
                mv_arr = export_motion_vectors(torch.from_numpy(motion_vectors).cuda())
                mv_arr = mv_arr.reshape(1, mv_arr.shape[0], mv_arr.shape[1], mv_arr.shape[2])
                if mvs is None:
                    mvs = mv_arr
                else:
                    mvs = torch.cat((mvs, mv_arr), 0)
            elif frame_type == "I":
                frame = np.dstack((frame[:, :, 2], frame[:, :, 1], frame[:, :, 0])).reshape(224, 224, 3)
                if init_frame is None:
                    init_frame = torch.from_numpy(frame).cuda()
                last_frame = torch.from_numpy(frame).cuda()
                #print(last_frame.device)
            
            frame_ct += 1
        cap.release()
        init_frame = init_frame.repeat(1, 1, 2).reshape(1, 224, 224, 6)
        last_frame = last_frame.repeat(1, 1, 2).reshape(1, 224, 224, 6)
        mvs = mvs[:224:16, :, :, :]
        mvs = torch.cat((init_frame, mvs, last_frame))


        while mvs.shape[0] < 16:
            mvs = torch.cat((mvs, torch.zeros((1, 224, 224, 6)).cuda()))
        
        mvs = mvs.permute(0, 3, 1, 2)
        
        if(mvs.shape[0] != 16 or mvs.shape[1] != 6):
            print("ERROR", mvs.shape)
            return torch.zeros(16, 6, 224, 224).cuda()

        np.savez_compressed(path[:-4], a=mvs.cpu())
        #ani = animation.ArtistAnimation(fig, frames, interval=50, repeat_delay=1000)
        
        #plt.show()
        #gc.collect()  
        return mvs
    
def decompress_fast(path):
    npy = path[:-4] + ".npz"

    if os.path.exists(npy):
        return torch.from_numpy(np.load(npy)['a'])
    return None

def main():
    start = time.time()
    folder = "fmt_videos_test"
    classes = os.listdir(folder)

    num_vids = 0
    for cls in classes:
        clsdir = os.path.join(folder, cls)

        vids = os.listdir(clsdir)

        for vid in vids:
            vpath = os.path.join(clsdir, vid)
            if not os.path.exists(vpath[:-4] + ".npz"):
                decompress(vpath)
                num_vids += 1
                print(num_vids)
                #x = np.load(vpath[:-4] + ".npz")

    print("End", time.time() - start)
    print("Vids processed: ", num_vids)

if __name__ == "__main__":
    main()