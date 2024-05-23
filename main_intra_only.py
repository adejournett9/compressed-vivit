from create_dataset import load_dataset
from vivit import VideoTransformer
from correct import VisionTransformer
import torch

from torchsummary import summary

from train import train

import warnings

'''
Train ViViT using the motion augmented frame binding technique
'''

def main():
    warnings.filterwarnings('ignore')
    num_frames = 16
    num_heads = 12
    hidden_dim = 768
    num_layers = 12
    patch_size = 16
    num_channels = 6
    #training params form table 7 of ViViT 
    train_opts = {'lr': 0.1, 'weight_decay': 0.001, 'num_epochs': 5, 'step_size':3, 'gamma':0.1, 'batch_size': 16, 'momentum': 0.9, 'use_compressed_data' : True}


    train_ds = load_dataset("fmt_videos_train", "train.csv", train_opts['batch_size'], num_frames, train_opts['use_compressed_data'], 2)
    test_ds = load_dataset("fmt_videos_test", "test.csv",  train_opts['batch_size'], num_frames, train_opts['use_compressed_data'], 2)
    model = VideoTransformer((224, 224, num_frames), patch_size, hidden_dim, num_heads, num_layers, 0.1, num_channels, train_opts['batch_size'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    summary(model, (16, num_channels, 224, 224))
    model = model.to(device)
    train(model, device, train_ds, test_ds, train_opts, exp_dir="vivit_compress_npz")

if __name__ == "__main__":
    main()