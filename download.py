'''
Code to download Kinetics400 is modified from 
https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/pytorchvideo_tutorial.ipynb
'''
from datetime import timedelta
import json
import os
import subprocess

import youtube_dl
from youtube_dl.utils import (DownloadError, ExtractorError)

def download_video(url, start, dur, output):
    output_tmp = os.path.join("/tmp",os.path.basename(output))
    try:
        print(url)
    # From https://stackoverflow.com/questions/57131049/is-it-possible-to-download-a-specific-part-of-a-file
        with youtube_dl.YoutubeDL({'format': 'best'}) as ydl:
            result = ydl.extract_info(url, download=False)
            video = result['entries'][0] if 'entries' in result else result

        url = video['url']
        if start < 5:
            offset = start
        else:
            offset = 5
        start -= offset
        offset_dur = dur + offset
        start_str = str(timedelta(seconds=start))
        dur_str = str(timedelta(seconds=offset_dur))

        cmd = ['ffmpeg', '-i', url, '-ss', start_str, '-t', dur_str, '-c:v',
                'copy', '-c:a', 'copy', output_tmp]
        subprocess.call(cmd)

        start_str_2 = str(timedelta(seconds=offset))
        dur_str_2 = str(timedelta(seconds=dur))

        cmd = ['ffmpeg', '-i', output_tmp, '-ss', start_str_2, '-t', dur_str_2, output]
        subprocess.call(cmd)
        return True

    except (DownloadError, ExtractorError) as e:
        print("Failed to download %s" % output)
        return False

def main():
    with open("./train.json", "r") as f:
        train_data = json.load(f)

    #rint(len(train_data), list(train_data.items())[6670])

    #exit(1)
    target_classes = [
    'springboard diving',
    'surfing water',
    'swimming backstroke',
    'swimming breast stroke',
    'swimming butterfly stroke',
    ]
    data_dir = "./train_vids"

    classes_count = {c:0 for c in target_classes}
    ct = 0
    for fn, data in train_data.items():
        print("{:.2f}".format(ct / len(train_data))) 
        label = data["annotations"]["label"]
        segment = data["annotations"]["segment"]
        url = data["url"]
        dur = data["duration"]
        #if label in classes_count and classes_count[label] < max_samples:
        c_dir = os.path.join(data_dir, label)
        if not os.path.exists(c_dir):
            os.makedirs(c_dir)



        start = segment[0]
        output = os.path.join(c_dir, "%s_%s.mp4" % (label.replace(" ","_"), fn))

        result = True
        if not os.path.exists(output) and not os.path.exists(os.path.join("/tmp",os.path.basename(output))):
            result = download_video(url, start, dur, output)
        if result:
            if label not in classes_count:
                classes_count[label] = 0
            classes_count[label] += 1

print("Finished downloading videos!")


if __name__ == "__main__":
    main()




