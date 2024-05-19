import os

def format_vid(vid_name, resolution, framerate, indir, outdir):
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  print("ffmpeg -i \"{}\" -s {}x{} -aspect 1:1 -r 15 -vcodec mpeg4 \"{}\"".format(indir +vid_name, resolution, resolution, outdir + vid_name))
  os.system("ffmpeg -i \"{}\" -s {}x{} -aspect 1:1 -r 15 -vcodec mpeg4 \"{}\"".format(indir + vid_name, resolution, resolution, outdir + vid_name))


def main():
  vid_dir = "test_vids"

  for cls in os.listdir(vid_dir):
    videos = os.listdir(os.path.join(vid_dir, cls))

    for vid in videos:
       format_vid(vid, 224, -1, vid_dir + '/' + cls + '/', 'fmt_videos_test/'+ cls + '/')


if __name__ == "__main__":
  main()


