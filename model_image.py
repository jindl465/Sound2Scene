import torch
from torch import nn
import torch.nn.functional as F
from models import resnet_image

class A2INet(nn.Module):

    def __init__(self,args):
        super(A2INet, self).__init__()
        self.imgnet = Resnet(args)

    def forward(self, image):
        x, img, inter = self.imgnet(image)

        # return aud, x, inter
        return img, x

def Resnet(opt):

    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

    if opt.model_depth == 10:
        model = resnet_image.resnet10(
            num_classes=opt.n_classes)
    elif opt.model_depth == 18:
        model = resnet_image.resnet18(
            num_classes=opt.n_classes,
            pool=opt.pool)
    elif opt.model_depth == 34:
        model = resnet_image.resnet34(
            num_classes=opt.n_classes,
            pool=opt.pool)
    elif opt.model_depth == 50:
        model = resnet_image.resnet50(
            num_classes=opt.n_classes,
            pool=opt.pool)
    elif opt.model_depth == 101:
        model = resnet_image.resnet101(
            num_classes=opt.n_classes)
    elif opt.model_depth == 152:
        model = resnet_image.resnet152(
            num_classes=opt.n_classes)
    elif opt.model_depth == 200:
        model = resnet_image.resnet200(
            num_classes=opt.n_classes)
    return model

if __name__=='__main__':
    model = A2INet()