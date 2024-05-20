import torchvision
from torchsummary import summary

from vivit import VideoTransformer


def main():
    # model = torchvision.models.vit_l_16()
    # summary(model, (3, 224, 224))

    model = VideoTransformer((224,224,16), 16, 768, 12, 12, 0.1, 6)
    summary(model, (16, 6, 224, 224))

    # model = torchvision.models.video.mvit_v1_b()
    # summary(model, (3,16, 224, 224))

if __name__ == "__main__":
    main()