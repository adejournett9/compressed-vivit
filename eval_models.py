import torch
from train import eval
from create_dataset import load_dataset
from vivit import VideoTransformer

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    print("Device:", device)

    val_ds = load_dataset("fmt_videos_test", "test.csv",  16, 16, False, 0)
    unmodified = torch.load("vivit_unmodified/checkpoint_1.pt")
    v_loss, v_acc = eval(unmodified, device, val_ds, 16)


    print("Unmodified Acc: {}%".format(v_acc))

if __name__ == "__main__":
    main()