from torchvision.io import read_video
import os
import csv

from torch.utils.data import Dataset
from sklearn import preprocessing
import torch
import numpy as np

import random

from extract import decompress, decompress_fast, decompress_intra

from torchvision.transforms import v2

class LazyLoaderDataset(Dataset):
    def __init__(self, path_list, labels, batch_size, num_frames, use_motion=False, mode=1):
        self.paths = path_list
        self.labels = labels
        self.batch_size = batch_size
        self.path_idx = [i for i in range(len(self.paths))]
        self.num_frames = num_frames
        self.use_motion = use_motion

        '''
        Modes:
        0: unmodified tubelet embedding (raw video)
        1: motion augmented frame bindings 
        2: intra-coded only
        '''
        self.mode = mode

        print(int(round(len(self.paths) / 64.0) * 16))
        #self.tensors = (yield self.__getitem__(1), self.labels)

        self.transforms = v2.Compose([
            v2.RandomCrop((224,224)),
            v2.RandomHorizontalFlip(0.5),
            v2.ScaleJitter((224,224), (0.9, 1.33)),
            v2.ColorJitter(),
            v2.Resize((224,224))
        ])

    def __len__(self):
        #4 views if unmodified, else 1 view
        return 4 * len(self.paths) if self.mode == 0 else len(self.paths)
    
    def get_video(self, path):
        with torch.no_grad():
            vid = None
            if self.use_motion == True:
                if self.mode == 1:
                    vid = decompress_fast(path)
                elif self.mode == 2:
                    vid = decompress_intra(path, self.batch_size)
            else:
                vid = read_video(path, output_format="TCHW", pts_unit='sec')
                vid = vid[0]
                start_frame = 0 if vid.shape[0] <= self.num_frames+1 else random.randint(0, (vid.shape[0] - 1) - self.num_frames)
                vid = vid[start_frame:start_frame + self.num_frames,:,:,:] / 255.0 #normalize
                vid = None if vid.shape[-1] != 224 else vid
        
            return vid
    
    def __getitem__(self, idx):
        with torch.no_grad():
            path_idx = random.choice(self.path_idx)#, k=1)
            path = self.paths[path_idx]

            vid = self.get_video(path)

            while vid is None:
                path_idx = random.choice(self.path_idx)#, k=1)
                path = self.paths[path_idx]
                vid = self.get_video(path)
            
            vid = vid / 255.0
            #if vid is 

            # if vid.shape[0] < self.num_frames:
            #     os.system("rm \"{}\"".format(path))
            #     print("ER", vid.shape)
            #     path_idx = random.choice(self.path_idx)#, k=1)
            #     path = self.paths[path_idx]

            #     vid = read_video(path, output_format="TCHW", pts_unit='sec')
            #     vid = vid[0]
            #     vid = vid[:self.num_frames,:,:,:] / 255.0 #normalize

            label = self.labels[path_idx]
            #print("Bef trans", vid.shape)
            trans = self.transforms(vid)

            #print("After trans", trans.shape, vid.shape)
            return trans, label
    

def get_label(csv_reader, cls, tag):
    for row in csv_reader:
        if row[0] == cls and row[1] == tag:
            return row
    return None


def load_dataset(vid_dir, label_file, batch_size, num_frames, use_compressed_data, mode):
    classes = os.listdir(vid_dir)
    reader = csv.reader(open(label_file, 'r'), delimiter=',')

    labels = []
    paths = []
    for cls in classes:
        folder = os.path.join(vid_dir, cls)

        vids = os.listdir(folder)

        #get a list of valid paths and labels
        ct = 0
        for vid in vids:
            tag = vid.split('_')[1].split('.')[0]
            label = get_label(reader, cls, tag)
            
            full_path = os.path.join(folder, vid)

            paths.append(full_path)
            labels.append(label)
            
            ct += 1
    
    le = preprocessing.LabelEncoder()
    numeric_labels = le.fit_transform(labels)
    labels = torch.tensor(numeric_labels)

    return LazyLoaderDataset(paths, labels, batch_size, num_frames, use_compressed_data, mode)


