import torch
from timesformer.models.vit import TimeSformer
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\Gebruiker\Documents\GitHub\Transformer-MM-Explainability')

from baselines.ViT.ViT_new import vit_base_patch16_224 as vit

def load_np(filename):
    with open(os.path.join(r'C:\Users\Gebruiker\Documents\GitHub\Research Internship extra\numpy_videos', filename), 'rb') as r:
        video_arr = np.load(r)
    r.close()
    return video_arr

def timesformer_pred(filename):
    video_arr = load_np(filename)
    video_arr = video_arr.reshape((1, 3, 8, 224, 224))
    video_torch = torch.from_numpy(video_arr)
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


def video_to_image(filename, num_frames=8):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize, ])

    image_list = []
    filename = filename.replace(".npy", ".jpg")
    for i in range(num_frames):
        file_path = os.path.join(r'C:\Users\Gebruiker\Documents\GitHub\Research Internship extra\jpg_files', 'image' + str(i) + '_' + filename.lower())
        print("Opened image at ", file_path)
        img = Image.open(file_path)
        img = transform(img)
        image_list.append(img)
    return image_list


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(original_image, class_index=None):
    transformer_attribution = generate_relevance(model, original_image.unsqueeze(0).cuda(), index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
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
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda())
    return R[0, 1:]


model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time',  pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth').cuda()
labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}

filename = '1047_IOM_SAD_XX.npy'

video_torch = timesformer_pred(filename)
images = video_to_image(filename, 8)
for image in images:
    print(np.array(image).shape)
images = np.stack(images)
print(images.shape)
video_arr = images.reshape((3, 8, 224, 224))
print(video_arr.shape)
video_torch = torch.from_numpy(video_arr)
print(video_torch.shape)
output = generate_visualization(video_torch)
plt.show(output)
plt.waitforbuttonpress()
exit(1)

video_arr = load_np(filename)
print(video_arr.shape)
video_arr = video_arr.reshape((3, 8, 224, 224))
print(video_arr.shape)
video_torch = torch.from_numpy(video_arr)
print(video_torch)
print(video_torch.shape)
print(video_torch.unsqueeze(0).cuda().shape)
output = generate_visualization(video_torch)
plt.show(output)
plt.waitforbuttonpress()