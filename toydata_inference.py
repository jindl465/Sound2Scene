import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from model_image import A2INet
from torchvision.utils import save_image,make_grid
from inference.generate_images import get_model
from scipy import signal
import soundfile as sf
import librosa
from pydub import AudioSegment
from torchvision.transforms.functional import to_pil_image
import pdb
import shutil
from PIL import Image as Image_PIL


def get_arguments():
    parser = argparse.ArgumentParser()
    #Definine Sound2Scene model (audio encoder)
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=2048,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--checkpoint_vggish', dest='checkpoint',help='Path of checkpoint file for load model', default="./samples/output/test.pth")

    #Defining image encoder and image decoder
    parser.add_argument('--root_path', default="./checkpoints")
    parser.add_argument('--model', default="icgan")
    parser.add_argument('--model_backbone', default="biggan")
    parser.add_argument('--resolution', default=128)
    parser.add_argument("--z_var", type=float, default=1, help="Noise variance: %(default)s)")
    parser.add_argument(
        "--trained_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the model has been trained on.",
    )

    #Defining directories
    parser.add_argument("--dataset", type=str, default="vgg", choices=["vgg", "vegas"],help="Dataset in which the model has been trained on.", )
    parser.add_argument("--img_path", type=str, default="./samples/testimages")
    parser.add_argument("--out_path", type=str, default="./samples/output")



    return parser.parse_args()

# def load_data(args, batch_size=None):
#     if batch_size==None:
#         batch_size=args.batch_size
#
#     if args.dataset == 'vgg':
#         aud_path = os.path.join(args.vgg_dir, "audios")
#         vid_path = os.path.join(args.vgg_dir, "frames_10fps")
#         train_dataset = GetVGGSound(args.data_txt, args.annotation, aud_path, vid_path, vid_path)
#         test_dataset = GetVGGSound(args.data_t_txt, args.annotation, aud_path, vid_path, vid_path)
#
#     elif args.dataset == "vegas":
#         train_dataset = GetAudioVideoDataset(args.data_txt, args.aud_path, args.emb_path, args.img_path)
#         test_dataset = GetAudioVideoDataset(args.data_t_txt, args.aud_t_path, args.emb_t_path, args.img_t_path)
#
#     #train_loader=None
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     return train_loader, test_loader

def makeAudio(args,generator,feature_extractor, emb):
    # define noise_vectors
    z = torch.empty(8,generator.dim_z).normal_(mean=0, std=args.z_var)

    #normalize audio embedding
    emb /= torch.linalg.norm(emb, dim=-1, keepdims=True)
    emb_ = torch.tile(emb, (8,1))

    gen_img = generator(z.cuda(), None, emb_)
    gen_img = torch.clamp(gen_img,-1., 1.)

    output = make_grid(gen_img, normalize=True, scale_each=True, nrow=8)

    return output

def gen_name(tuple):
    name = ''
    for i in tuple:
        name = name+i+"_"
    return name


def preprocess_img_feature(img_path):
    pil_image = Image_PIL.open(img_path).convert('RGB')
    norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    feature_transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor(),
         transforms.Normalize(norm_mean, norm_std)])
    tensor_image = feature_transform(pil_image)
    tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
    return tensor_image

def generate_audios(args, model, generator,feature_extractor,device):
    img_paths = args.img_path
    save_path = args.out_path
    os.makedirs(save_path, exist_ok=True)

    for image in os.listdir(img_paths):
        if image != ".DS_Store":
            image_path = os.path.join(img_paths, image)
            #audio = "./samples/inference/chainsaw.wav"

            image_data = preprocess_img_feature(image_path)
            image_data = Variable(image_data.squeeze(1)).to(device)
            img, emb = model(image_data)
            print(emb)
            # output = showImage(args, generator, feature_extractor, emb)

            # save_name = audio.split("/")[-1].split(".")[0]
            #
            # save_final = os.path.join(save_path, save_name+".png")
            #
            # save_image(output.cpu(), save_final)



def main():
    import random
    random_seed=1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load sound2scene model
    checkpoint = torch.load(args.checkpoint,map_location=device)
    model = A2INet(args).to(device)
    model.load_state_dict(checkpoint)
    model.eval()


    #load image decoder
    suffix = (
        "_nofeataug"
        if args.resolution == 256
        and args.trained_dataset == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        args.model,
        args.model_backbone,
        args.trained_dataset,
        args.resolution,
        suffix,
    )
    generator, feature_extractor = get_model(
        exp_name, args.root_path, args.model_backbone, device=device
    )

    generate_audios(args, model, generator, feature_extractor, device)

if __name__=='__main__':
    main()




