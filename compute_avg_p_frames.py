from mvextractor.videocap import VideoCap
import os

import matplotlib.pyplot as plt

def main():
    folder = "test_vids"
    classes = os.listdir(folder)

    p_frames = 0
    num_vids = 0
    min_p_frames = 10000
    max_p_frames = 0
    frame_hist = []
    i_frame_hist = []
    for cls in classes:
        clsdir = os.path.join(folder, cls)

        vids = os.listdir(clsdir)

        for vid in vids:
            vpath = os.path.join(clsdir, vid)
            cap = VideoCap()
            success = cap.open(vpath)

            if success:
                num_vids += 1
            vid_p_ct = 0
            vid_i_ct = 0
            while success:
                success, frame, motion_vectors, frame_type, timestamp = cap.read()

                if frame_type == 'P':
                    vid_p_ct += 1
                elif frame_type == 'I':
                    vid_i_ct += 1

            frame_hist.append(vid_p_ct)
            i_frame_hist.append(vid_i_ct)
            if vid_p_ct > max_p_frames:
                max_p_frames = vid_p_ct

            if vid_p_ct < min_p_frames:
                min_p_frames = vid_p_ct

    print("Avg P frames", p_frames / num_vids)
    print("Min", min_p_frames)
    print("Max", max_p_frames)

    fig, ax = plt.subplots(1,2)
    ax[0].hist(frame_hist, density=True)
    ax[1].hist(i_frame_hist, density=True)
    plt.show()

if __name__ == "__main__":
    main()