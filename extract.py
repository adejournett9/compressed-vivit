from mvextractor.videocap import VideoCap
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib import animation

def draw_motion_vectors(frame, motion_vectors):
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            cv2.arrowedLine(frame, start_pt, end_pt, (0, 0, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame

def main():
    cap = VideoCap()
    success = cap.open("save2.mp4")

    frames = []
    
    fig, ax = plt.subplots(1,1)
    while success:
        success, frame, motion_vectors, frame_type, timestamp = cap.read()

        if not success or len(frames) == 5000:
            break
        frame = draw_motion_vectors(frame, motion_vectors)
        im1 = ax.imshow(frame)

        frames.append([im1])
        print("-----")
        print(frame.shape)
        print(motion_vectors.shape, motion_vectors[:5])
        print(frame_type)
        print(timestamp)
        print("-----")

    ani = animation.ArtistAnimation(fig, frames, interval=50, repeat_delay=1000)
    
    plt.show()


if __name__ == "__main__":
    main()