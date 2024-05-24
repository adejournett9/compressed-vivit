#Video Compression To Scale Vision Transformers


##Required Packages
1. torch
2. torchvision
3. torchsummary
4. eigen
5. matplotlib
6. cv2
7. scipy


##Formatting the Kinetics400 dataset
1. To download the Kinetics400 dataset, run download.py. This will take several days depending on internet speed.
2. To perform the preprocessing listed in the paper, run reformat_video.py on the appropriate directory
3. The raw data is now prepared

##Fast Decompression
1. To compress the videos in the test and training set, run extract.py
2. The above will generate NPZ files which can then be loaded on the fly by the LazyLoaderDataset during training. This only applies for Motion Augmented Frame Bindings in the paper. Intra-coded only can be extracted very quickly using ffmpeg. 

##Training the model
There are 3 training scripts availible with the pattern main_*.py 

##Evaluating the model
After training, the model can be evaluated using eval_model.py, by swapping in the associated checkpoint *.pt file. This is how Table 1 was generated in the paper. 