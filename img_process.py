# 모듈 설치: !pip install moviepy
import os
import librosa
import soundfile
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoFileClip


path = os.getcwd()
dir_path = "./toy/image_raw"
new_path = os.path.join(path, "toy/image")
dir_list = os.listdir(dir_path)

for file in dir_list:
    # filename = file.replace('mp4', 'wav')
    src = dir_path + "/" + file
    # src = os.path.join(dir_path, file)
    # dst = os.path.join(new_path, filename)
    # extract_audio_from_video(src, dst)
    downsampling(src)




