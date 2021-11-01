from torchvision.io import read_video
import os
import numpy as np
import torch
import av
from timesformer.datasets.decoder import decode

# filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\avi_videos", "1025_MTI_HAP_XX.avi")
# container = av.open(filename)
# print(container)
# exit(1)

def load_np(filename):
    with open(os.path.join(r'C:\Users\Gebruiker\Documents\GitHub\Research Internship extra\numpy_videos', filename), 'rb') as r:
        video_arr = np.load(r)
    r.close()
    print(video_arr.shape)
    return video_arr

def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def convert_video(filename):
    filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\avi_videos", filename)
    video_container = av.open(filename)
    frames = decode(video_container, 32, 8, -1, 1, )
    video_torch = tensor_normalize(frames, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    video_torch = video_torch.permute(3, 0, 1, 2)
    video_torch = torch.index_select(video_torch, 1, torch.linspace(0, video_torch.shape[1] - 1, 8).long(), )
    print(video_torch)
    print(video_torch.shape)

convert_video("1025_MTI_HAP_XX.avi")
exit(1)

video_torch = torch.from_numpy(load_np("1025_MTI_HAP_XX.npy"))
print(video_torch.shape)
video_torch = tensor_normalize(video_torch, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
video_torch = video_torch.permute(3, 0, 1, 2)
print(video_torch.shape)
# video_torch = video_torch.unsqueeze(0)
video_torch = torch.index_select(video_torch, 1, torch.linspace(0, video_torch.shape[1] - 1, 8).long(),)
print(video_torch.shape)

# filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\avi_videos", "1025_MTI_HAP_XX.avi")
# video = read_video(filename)[0].int().permute(3, 0, 1, 2)
# print(video)
# print(video.shape)
# print(video.eq(video))
# video = video[0].reshape((1, 3, 8, 224, 224))
# print(video)
# print(video.shape)
