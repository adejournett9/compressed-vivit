from create_dataset import load_dataset
from vivit import VideoTransformer
from correct import VisionTransformer
import torch

from torchsummary import summary

from train import train

def main():
    num_frames = 16
    num_heads = 16
    hidden_dim = 1024
    num_layers = 12
    #training params form table 7 of ViViT 
    train_opts = {'lr': 0.1, 'weight_decay': 0.001, 'num_epochs': 30, 'step_size':3, 'gamma':0.1, 'batch_size': 32, 'momentum': 0.9}


    train_ds = load_dataset("fmt_videos_train", "train.csv", train_opts['batch_size'], num_frames)
    test_ds = load_dataset("fmt_videos_test", "test.csv",  train_opts['batch_size'], num_frames)
    model = VideoTransformer((224, 224, num_frames), num_heads, hidden_dim, num_heads, num_layers)
    #model = VisionTransformer(224, 16, 3, 400, 768, 12, 12, 3, True, 0.1, 0.1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    summary(model, (16, 3, 224, 224))

    train(model, device, train_ds, test_ds, train_opts, exp_dir="vivit_unmodified")

if __name__ == "__main__":
    main()