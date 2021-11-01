import torch
from timesformer.models.vit import TimeSformer
from timesformer.datasets.decoder import decode
import os
import numpy as np
import pickle
import av


def load_np(filename):
    with open(os.path.join(r'C:\Users\Gebruiker\Documents\GitHub\Research Internship extra', filename), 'rb') as r:
        video_arr = np.load(r)
    r.close()
    return video_arr

def load_video(filename):
    filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship", filename)
    video_container = av.open(filename)
    return video_container

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


def timesformer_pred(filename, load_numpy=True):
    if load_numpy:
        video_arr = load_np(filename)
        video_torch = torch.from_numpy(video_arr)
    else:
        video_container = load_video(filename)
        video_torch = decode(video_container, 32, 8, -1, 1, )
    video_torch = tensor_normalize(video_torch, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
    video_torch = video_torch.permute(3, 0, 1, 2)
    video_torch = torch.index_select(video_torch, 1, torch.linspace(0, video_torch.shape[1] - 1, 8).long(), )
    video_torch = video_torch.unsqueeze(0)
    pred = model(video_torch,)
    print(pred)
    print_top_classes(pred)
    return video_torch


def print_top_classes(predictions, **kwargs):
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    print("prob ", prob)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    print("LOLOL", class_indices)
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(labels[cls_idx])
        if len(labels[cls_idx]) > max_str_len:
            max_str_len = len(labels[cls_idx])

    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, labels[cls_idx])
        output_string += ' ' * (max_str_len - len(labels[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)

def topks_correct(preds, labels, ks):
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # print(preds)
    # print(top_max_k_inds)
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # print(rep_max_k_labels)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # print(top_max_k_correct[0][258], top_max_k_correct[0][259])
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


with open(r"C:\Users\Gebruiker\Documents\GitHub\TimeSformer\Results\results_output", 'rb') as f:
    x = pickle.load(f)

f.close()
# print(x)
# print(x[0].shape, x[1].shape)
# print(x[0][258], x[1][258])

total_correct = topks_correct(x[0], x[1], [1])
# print(total_correct)
# print(total_correct[0]/x[1].shape[0] * 100)


model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time',  pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth')
labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}

# filename = os.path.join('avi_videos', '1025_MTI_HAP_XX.avi')
# true_label = 0
#
# timesformer_pred(filename, load_numpy=False)
# print("True label is", labels[true_label])
# print(x[0][258], x[1][258])


filename = os.path.join('avi_custom', 'happy.avi')
true_label = 0

timesformer_pred(filename, load_numpy=False)
print("True label is", labels[true_label])


filename = os.path.join('avi_custom', 'anger.avi')
true_label = 1

timesformer_pred(filename, load_numpy=False)
print("True label is", labels[true_label])



exit(1)

filename = os.path.join('avi_videos', '1003_TAI_SAD_XX.avi')
true_label = 1

timesformer_pred(filename, load_numpy=False)
print("True label is", labels[true_label])
print(x[0][259], x[1][259])