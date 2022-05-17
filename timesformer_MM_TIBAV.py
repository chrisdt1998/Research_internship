"""
This file contains the adaption of the MM TIBAV visual technique on the TimeSformer.

This code was written and designed by Christopher du Toit.
"""

import torch
from timesformer.models.vit_new import TimeSformer
import os
import numpy as np
import cv2
from timesformer.datasets.decoder import decode
import av


def np_to_video(visual_arr, video_path, video_name):
    height, width = 224, 224

    video = cv2.VideoWriter(video_path + video_name + ".avi", 0, 1, (width, height))

    for image in visual_arr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


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
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
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
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    top_max_k_inds = top_max_k_inds.t()
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(model, original_image, class_index=None, inc_temporal=False):
    transformer_attribution = generate_relevance(model, original_image, index=class_index,
                                                 inc_temporal=inc_temporal).detach()
    transformer_attribution = transformer_attribution.reshape(1, 8, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(8, 224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    vis_list = []

    for i in range(8):
        image = original_image.squeeze()[:,i,:,:]
        image_transformer_attribution = image.permute(1, 2, 0).data.numpy()
        image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
                    image_transformer_attribution.max() - image_transformer_attribution.min())
        vis = show_cam_on_image(image_transformer_attribution, transformer_attribution[i])
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        vis_list.append(vis)

    return vis_list


# rule 5 from paper
def avg_heads(cam, grad):
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=1)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    if cam_ss.type() != R_ss.type():
        R_ss = R_ss.float()
        cam_ss = cam_ss.float()
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    num_tokens_temporal = model.blocks[0].temporal_attn.get_attention_map().shape[-1]
    R = []

    for _ in range(num_tokens_temporal):
        R.append(np.eye(num_tokens, num_tokens))
    R = np.stack(R)
    R = torch.from_numpy(R).cuda()

    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())

    return R[:, 0, 1:]


model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time',
                    pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth')
labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}

files = ['1025_MTI_HAP_XX.avi', '1003_TAI_SAD_XX.avi', '1052_TIE_ANG_XX.avi',
         '1041_TSI_FEA_XX.avi', '1014_TSI_DIS_XX.avi', '1014_IWW_NEU_XX.avi']
