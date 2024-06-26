import os,glob,json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional
from PIL import Image
import random
import data_utils.utils as data_utils
from PIL import Image as Image_PIL
import json
import soundfile as sf
from scipy import signal
import librosa

class GetVGGSound(Dataset):
    def __init__(self, data_path):
        self.img_path = os.path.join(data_path, "image")
        self.aud_path = os.path.join(data_path, "audio")

        self.load_files()
        # initialize audio transform
        self._init_atransform()

        # preprocess image for visualization
        norm_mean = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        norm_std = torch.Tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.image_transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor(),
             transforms.Normalize(norm_mean, norm_std)])  # only resize

        # preprocess image for feature extractor
        norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.feature_transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(),
             transforms.Normalize(norm_mean, norm_std)])  # only resize

    def load_files(self):
        self.img_lists = []
        self.aud_lists = []
        self.index_lists = []
        # self.onehot_lists = []
        # self.onehot_dict = {}
        k = 2
        index = 0
        aud_list = os.listdir(self.aud_path)
        for dirs in os.listdir(self.img_path):
            if dirs + ".wav" in aud_list:
                # if dirs not in self.onehot_dict.keys():
                #     self.onehot_dict[dirs] = index
                #     index += 1

                for i in range(k):
                    # self.onehot_lists.append(self.onehot_dict[dirs])
                    self.index_lists.append(dirs)
                    self.img_lists.append(os.path.join(self.img_path, dirs + "/" + dirs + "_index_"+str(5 + i))+".jpg")
                    self.aud_lists.append(os.path.join(self.aud_path, dirs + ".wav"))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def preprocess_img(self, img_path):
        pil_image = Image_PIL.open(img_path).convert('RGB')
        tensor_image = self.image_transform(pil_image)
        tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 128, mode="bicubic",align_corners=True)
        tensor_image=torch.clamp(tensor_image, -1., 1.)
        return tensor_image

    def preprocess_img_feature(self, img_path):
        pil_image = Image_PIL.open(img_path).convert('RGB')
        tensor_image = self.feature_transform(pil_image)
        tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
        return tensor_image

    def __len__(self):
        return len(self.index_lists)

    def __getitem__(self, idx):
        index = self.index_lists[idx]
        # onehot_index = self.onehot_lists[idx]
        frame_length = 0.025
        frame_stride = 0.010

        # Audio processing
        wav_file = self.aud_lists[idx]
        samples, samplerate = sf.read(wav_file)
        # repeat in case audio is too short
        resamples = np.tile(samples,10)[:160000] #10sec
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        # frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512,noverlap=353)
        input_nfft = int(round(samplerate*frame_length))
        input_stride = int(round(samplerate*frame_stride))
        mel_spectrogram = librosa.feature.melspectrogram(y=resamples, sr=samplerate, n_mels=40, n_fft=input_nfft, hop_length=input_stride)
        spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        # spectrogram = np.log(spectrogram+ 1e-7)
        # mean = np.mean(spectrogram)
        # std = np.std(spectrogram)
        # spectrogram = np.divide(spectrogram-mean,std+1e-9)

        # Image processing
        img_file = self.img_lists[idx]

        img = self.preprocess_img(img_file) #for visualization
        emb = self.preprocess_img_feature(img_file) #for feeding to image encoder

        return index, spectrogram, emb, img
