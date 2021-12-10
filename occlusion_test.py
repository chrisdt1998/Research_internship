import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from emotion_trends import Trends
from attention_extraction import AttnExtract
from timesformer.models.vit_new import TimeSformer

class OcclusionTest():
    def __init__(self, src, model, visual_type='TIBAV', data_type='correct'):
        self.DetectTrends = Trends(visual_type, data_type)
        self.AttnExtract = AttnExtract(src, model)
        self.timesformer_pred = self.AttnExtract.timesformer_pred
        self.interpolate_attn = self.AttnExtract.interpolate_attn
        self.emotion_dict = self.DetectTrends.emotion_dict
        self.labels = self.DetectTrends.labels
        self.num_frames = 8
        self.probability_dict = {}
        self.model = model

    def occlude_gaussian(self, num_patches, is_max=True):
        for emotion in self.emotion_dict:
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            average = self.interpolate_attn(torch.from_numpy(average))
            average = average.reshape(8, 224*224)
            total = np.sum(average, axis=1)
            binary_arr = []
            for i in range(self.num_frames):
                frame_prob = average[i] / total[i]
                print(frame_prob.max(), frame_prob.min())
                if not is_max:
                    frame_prob = (1 - frame_prob)/(224*224 - 1)
                    print(frame_prob.max(), frame_prob.min())
                binary_arr.append(np.random.choice(frame_prob.shape[0], num_patches*16*16, p=frame_prob))
            binary_arr = np.stack(binary_arr)
            self.probability_dict[self.labels[emotion]] = binary_arr

    def occlude_max_min(self, num_patches, is_max=True):
        for emotion in self.emotion_dict:
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            average = self.interpolate_attn(torch.from_numpy(average))
            average = average.reshape(8, 224 * 224)
            binary_arr = []
            for i in range(self.num_frames):
                if is_max:
                    binary_arr.append(np.argpartition(average[i], -num_patches * 16 * 16)[-num_patches * 16 * 16:])
                else:
                    binary_arr.append(np.argpartition(average[i], num_patches * 16 * 16)[:num_patches * 16 * 16])
            binary_arr = np.stack(binary_arr)
            print(binary_arr.shape)
            self.probability_dict[self.labels[emotion]] = binary_arr


    def occlude_features(self, face_parts):
        for emotion in self.emotion_dict:
            face_parts_arr = np.array([])
            if 'right eye' in face_parts:
                start_row = 3
                start_col = 9
                end_row = 5
                end_col = 12
                for j in range((end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            if 'left eye' in face_parts:
                start_row = 3
                start_col = 3
                end_row = 5
                end_col = 6
                for j in range((end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            if 'nose' in face_parts:
                start_row = 5
                start_col = 6
                end_row = 8
                end_col = 8
                for j in range((end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            if 'mouth' in face_parts:
                start_row = 9
                start_col = 3
                end_row = 11
                end_col = 12
                for j in range((end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            # if 'forehead' in face_parts:
            binary_arr = []
            for i in range(self.num_frames):
                binary_arr.append(face_parts_arr)
            binary_arr = np.stack(binary_arr)
            self.probability_dict[self.labels[emotion]] = binary_arr


    def apply_occlusion(self, filename, emotion):
        video_torch = self.timesformer_pred(filename).squeeze(0)
        video_torch = np.transpose(video_torch, (1, 2, 3, 0))
        orig_image = self.timesformer_pred(filename)
        for i in range(8):
            video_torch[i] = (video_torch[i] - video_torch[i].min()) / (video_torch[i].max() - video_torch[i].min())
        for j, frame in enumerate(self.probability_dict[emotion]):
            for i in frame:
                i = int(i)
                video_torch[j, int(i/224), i%224, :] = 0
        self.visualize_occlusion(video_torch, orig_image, emotion)
        video_torch = np.transpose(video_torch, (3, 0, 1, 2)).unsqueeze(0)
        output = self.model(orig_image.cuda(), register_hook=False)
        print(f"Original image prediction is: {self.labels[torch.argmax(output).item()]}")
        # print(torch.argmax(output))
        output = self.model(video_torch.cuda(), register_hook=False)
        print(f"Occluded image prediction is: {self.labels[torch.argmax(output).item()]}")
        # print(torch.argmax(output))
        #


    def visualize_occlusion(self, occ_image, orig_image, emotion):
        for i, frame in enumerate(occ_image):
            # frame = (frame - frame.min()) / (frame.max() - frame.min())
            plt.imshow(frame, label=str(i))
            # plt.title(self.labels[emotion])
            plt.show()


if __name__ == '__main__':
    model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time', pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth')
    labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}
    filename = os.path.join('avi_videos', '1025_MTI_HAP_XX.avi')
    filename_arr = ['1025_MTI_HAP_XX.avi', '1003_TAI_SAD_XX.avi', '1052_TIE_ANG_XX.avi', '1053_DFA_FEA_XX.avi', '1014_TSI_DIS_XX.avi', '1014_IWW_NEU_XX.avi']

    src = r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D"
    Occlusion = OcclusionTest(src, model.model.cuda())
    # Occlusion.occlude_gaussian(60, is_max=True)
    Occlusion.occlude_max_min(40, is_max=True)
    # Occlusion.occlude_features(['right eye', 'left eye', 'nose', 'mouth'])
    Occlusion.apply_occlusion(filename, 'happy')
