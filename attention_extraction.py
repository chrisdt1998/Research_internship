import csv
import torch
from timesformer.models.vit_new import TimeSformer
import os
import numpy as np
import cv2
from timesformer.datasets.decoder import decode
import av
import pickle
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

class AttnExtract():
    def __init__(self, src, model):
        self.height = 224
        self.width = 224
        self.src = src
        self.model = model

    def load_video(self, filename):
        # filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship", filename)
        filename = os.path.join(self.src, filename)
        video_container = av.open(filename)
        return video_container

    def tensor_normalize(self, tensor, mean, std):
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

    def timesformer_pred(self, filename):
        video_container = self.load_video(filename)
        video_torch = decode(video_container, 32, 8, -1, 1, )
        video_torch = self.tensor_normalize(video_torch, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        video_torch = video_torch.permute(3, 0, 1, 2)
        video_torch = torch.index_select(video_torch, 1, torch.linspace(0, video_torch.shape[1] - 1, 8).long(), )
        video_torch = video_torch.unsqueeze(0)
        return video_torch

    def print_top_classes(self, predictions, **kwargs):
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

    def topks_correct(self, preds, labels, ks):
        # Find the top max_k predictions for each sample
        _top_max_k_vals, top_max_k_inds = torch.topk(
            preds, max(ks), dim=1, largest=True, sorted=True
        )
        top_max_k_inds = top_max_k_inds.t()
        rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
        top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
        topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
        return topks_correct

    def generate_visualization(self, original_image, class_index=None, inc_temporal=False):
        transformer_attribution = self.generate_relevance(original_image, index=class_index, inc_temporal=inc_temporal).detach()
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
            vis = self.show_cam_on_image(image_transformer_attribution, transformer_attribution[i])
            vis = np.uint8(255 * vis)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            vis_list.append(vis)
        return vis_list

    # rule 5 from paper
    def avg_heads(self, cam, grad):
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=1)
        return cam

    # rule 6 from paper
    def apply_self_attention_rules(self, R_ss, cam_ss):
        if cam_ss.type() != R_ss.type():
            R_ss = R_ss.float()
            cam_ss = cam_ss.float()
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def generate_relevance(self, input, index=None, inc_temporal=False):
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # Not sure what this one_hot is for.
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        num_tokens = self.model.blocks[0].attn.get_attention_map().shape[-1]
        num_tokens_temporal = self.model.blocks[0].temporal_attn.get_attention_map().shape[-1]
        R = []
        for _ in range(num_tokens_temporal):
            R.append(np.eye(num_tokens, num_tokens))
        R = np.stack(R)
        R = torch.from_numpy(R).cuda()
        for blk in self.model.blocks:
            grad = blk.attn.get_attn_gradients()
            cam = blk.attn.get_attention_map()
            cam = self.avg_heads(cam, grad)
            R += self.apply_self_attention_rules(R.cuda(), cam.cuda())
        if inc_temporal:
            R_t1 = np.ones((8, 196))
            R_t1 = torch.from_numpy(R_t1).cuda()
            R_t2 = np.ones((8, 196))
            R_t2 = torch.from_numpy(R_t2).cuda()
            for blk in self.model.blocks:
                grad = blk.temporal_attn.get_attn_gradients()
                cam = blk.temporal_attn.get_attention_map()
                cam = self.avg_heads(cam, grad)
                cam = cam.permute(2, 1, 0)
                cam_1 = cam[0,:,:] # time 1
                cam_2 = cam[:,0,:] # time 2
                R_t1 += R_t1.cuda() * cam_1.cuda()
                R_t2 += R_t2.cuda() * cam_2.cuda()
            R_ones = np.ones((8, 196))
            R_ones = torch.from_numpy(R_ones).cuda()
            return [R[:, 0, 1:].detach().cpu().numpy(), (R_t1 - R_ones).detach().cpu().numpy(), (R_t2 - R_ones).detach().cpu().numpy()], torch.argmax(output)

        return R[:, 0, 1:]

    def extract_TIBAV(self, test_file):
        correct_output, correct_labels, wrong_output, wrong_labels = ([] for i in range(4))
        with open(test_file) as r:
            csv_file = csv.reader(r, delimiter=' ')
            for index, row in enumerate(csv_file):
                file = row[0]
                filename = os.path.join('avi_videos', file)

                video_torch = self.timesformer_pred(filename)

                transformer_attribution, pred = self.generate_relevance(video_torch.cuda(), index=None,
                                                                   inc_temporal=True)
                if pred.item() != int(row[1]):
                    wrong_labels.append(pred.item())
                    print(f"{index}: File: {file}, Output is incorrect: True: {int(row[1])}, Predicted: {pred.item()}. Total incorrect = {len(wrong_labels)}.")
                    wrong_output.append(transformer_attribution)
                else:
                    correct_labels.append(pred.item())
                    print(f"{index}: File: {file}, Output is Correct: True: {int(row[1])}, Predicted: {pred.item()}. Total correct = {len(wrong_labels)}.")
                    correct_output.append(transformer_attribution)
        r.close()
        wrong_output = np.stack(wrong_output, axis=0)
        correct_output = np.stack(correct_output, axis=0)
        print(f"FINAL: Correct = {len(correct_labels)}, Incorrect: =  {len(wrong_labels)}.")
        with open("wrong_output", "wb") as f:
            pickle.dump(wrong_output, f)

        with open("correct_output", "wb") as f:
            pickle.dump(correct_output, f)

        with open("correct_labels", "wb") as f:
            pickle.dump(np.array(correct_labels), f)

        with open("wrong_labels", "wb") as f:
            pickle.dump(np.array(wrong_labels), f)


    def gradCAM(self, filename):
        def reshape_transform(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        target_layers = [model.blocks[-1].norm1]
        input_tensor =  self.timesformer_pred(filename)
        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=args.use_cuda, reshape_transform=reshape_transform)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=None)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    def extract_gradCAM(self, test_file):
        correct_output, correct_labels, wrong_output, wrong_labels = ([] for i in range(4))
        target_layers = [model.layer4[-1]]
        with GradCAM(model=self.model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
            with open(test_file) as r:
                csv_file = csv.reader(r, delimiter=' ')
                for index, row in enumerate(csv_file):
                    file = row[0]
                    filename = os.path.join('avi_videos', file)

                    video_torch = self.timesformer_pred(filename)


                    # if pred.item() != int(row[1]):
                    #     wrong_labels.append(pred.item())
                    #     print(f"{index}: File: {file}, Output is incorrect: True: {int(row[1])}, Predicted: {pred.item()}. Total incorrect = {len(wrong_labels)}.")
                    #     wrong_output.append(transformer_attribution)
                    # else:
                    #     correct_labels.append(pred.item())
                    #     print(f"{index}: File: {file}, Output is Correct: True: {int(row[1])}, Predicted: {pred.item()}. Total correct = {len(wrong_labels)}.")
                    #     correct_output.append(transformer_attribution)
            r.close()
        wrong_output = np.stack(wrong_output, axis=0)
        correct_output = np.stack(correct_output, axis=0)
        print(f"FINAL: Correct = {len(correct_labels)}, Incorrect: =  {len(wrong_labels)}.")
        with open("wrong_output", "wb") as f:
            pickle.dump(wrong_output, f)

        with open("correct_output", "wb") as f:
            pickle.dump(correct_output, f)

        with open("correct_labels", "wb") as f:
            pickle.dump(np.array(correct_labels), f)

        with open("wrong_labels", "wb") as f:
            pickle.dump(np.array(wrong_labels), f)

model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time', pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth')
labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}

tibav_visual = AttnExtract(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D", model.model.cuda())
tibav_visual.extract_TIBAV("CREMA_D/avi_videos/test.csv")