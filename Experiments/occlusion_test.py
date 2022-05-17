"""
This file contains code for occlusion testing for the different visual types.

This code was written and designed by Christopher du Toit.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import av
import cv2
import csv
import shutil

from emotion_trends import Trends
from attention_extraction import AttnExtract
from timesformer.datasets.decoder import decode

class OcclusionTest():
    def __init__(self, src, model, visual_type='TIBAV', data_type='correct'):
        if visual_type is not None:
            self.visual_type = visual_type
            self.DetectTrends = Trends(visual_type, data_type)
            self.AttnExtract = AttnExtract(src, model)
            self.timesformer_pred = self.AttnExtract.timesformer_pred
            self.interpolate_attn = self.AttnExtract.interpolate_attn
            self.emotion_dict = self.DetectTrends.emotion_dict
        self.labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}
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
                    binary_arr.append(np.argpartition(average[i], -num_patches)[-num_patches:])
                else:
                    binary_arr.append(np.argpartition(average[i], num_patches)[:num_patches])
            binary_arr = np.stack(binary_arr)
            self.probability_dict[self.labels[emotion]] = binary_arr


    def occlude_features(self, face_parts):
        for emotion in [0, 1, 2, 3, 4, 5]:
            face_parts_arr = np.array([])
            if 'Eyes' in face_parts:
                start_row = 2.5
                start_col = 1.5
                end_row = 5.5
                end_col = 12
                for j in range(int((end_row - start_row + 1))*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            if 'Nose' in face_parts:
                start_row = 5
                start_col = 4.5
                end_row = 8
                end_col = 8.5
                for j in range(int(end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            if 'Mouth' in face_parts:
                start_row = 8.5
                start_col = 2
                end_row = 12
                end_col = 12
                for j in range(int(end_row - start_row + 1)*16):
                    face_parts_arr = np.concatenate([face_parts_arr, np.arange(start_row*16*16*14 + j*14*16 + start_col * 16, start_row*16*16*14 + j*14*16 + (end_col + 1) * 16)])
            # if 'forehead' in face_parts:
            binary_arr = []
            for i in range(self.num_frames):
                binary_arr.append(face_parts_arr)
            binary_arr = np.stack(binary_arr)
            self.probability_dict[self.labels[emotion]] = binary_arr

    def occlude_frames(self, num_frames, is_max=True):
        for emotion in self.emotion_dict:
            average = np.mean(np.stack(self.emotion_dict[emotion]), axis=0)
            average = self.interpolate_attn(torch.from_numpy(average))
            average = average.reshape(8, 224 * 224)
            frames_sum = np.sum(average, axis=1)
            binary_arr = []
            if is_max:
                binary_arr.append(np.argpartition(frames_sum, -num_frames)[-num_frames:])
            else:
                binary_arr.append(np.argpartition(frames_sum, num_frames)[:num_frames])
            binary_arr = np.stack(binary_arr).reshape(num_frames)
            self.probability_dict[self.labels[emotion]] = binary_arr

    def load_video(self, filename):
        filename = os.path.join(self.AttnExtract.src, filename)
        # filename = self.AttnExtract.src + ''
        video_container = av.open(filename)
        video_container = decode(video_container, 32, 8, -1, 1, )
        return video_container

    def img_to_video(self, images, video_name, folder_name):
        width = 224
        height = 224
        path = self.AttnExtract.src + '/Occlusion/Face_parts/' + folder_name + '/' + video_name
        # video = cv2.VideoWriter(video_path + video_name + ".avi", 0, 1, (width, height))
        video = cv2.VideoWriter(path, 0, 1, (width, height))

        for image in images:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        cv2.destroyAllWindows()
        video.release()

    def apply_occlusion(self, folder_name):
        with open("CREMA_D/avi_videos/test.csv") as r:
            csv_file = csv.reader(r, delimiter=' ')
            for index, row in enumerate(csv_file):
                file = row[0]
                emotion = self.labels[int(row[1])]
                filename = os.path.join('avi_videos', file)
                video_torch = self.load_video(filename)
                for j, frame in enumerate(self.probability_dict[emotion]):
                    for i in frame:
                        i = int(i)
                        video_torch[j, int(i/224), i%224, :] = 0
                self.img_to_video(video_torch.numpy(), file, folder_name)

    def apply_occlusions_frame(self, num_frames, folder_name, max_or_min, percentage):
        with open("CREMA_D/avi_videos/test.csv") as r:
            csv_file = csv.reader(r, delimiter=' ')
            for index, row in enumerate(csv_file):
                file = row[0]
                emotion = self.labels[int(row[1])]
                filename = os.path.join('Occlusion', self.visual_type, max_or_min, '0', percentage, file)
                # filename = 'Occlusion/' + self.visual_type + '/' + max_or_min + '/' + percentage + '/' + file
                video_torch = self.load_video(filename)
                for j, frame in enumerate(self.probability_dict[emotion]):
                    video_torch[frame] = torch.zeros((224, 224, 3))
                self.img_to_video(video_torch.numpy(), file, percentage, folder_name)

    def visualize_occlusion(self, occ_image, orig_image, emotion):
        for i, frame in enumerate(occ_image):
            # frame = (frame - frame.min()) / (frame.max() - frame.min())
            plt.imshow(frame, label=str(i))
            # plt.title(self.labels[emotion])
            plt.show()

    def split_test(self):
        with open("CREMA_D/avi_videos/test.csv") as r:
            csv_file = csv.reader(r, delimiter=' ')
            for index, row in enumerate(csv_file):
                file = row[0]
                file_src = os.path.join(self.AttnExtract.src, "avi_videos", file)
                file_dst = os.path.join(self.AttnExtract.src, 'Occlusion', self.visual_type, 'MMAX', '0', '0', file + '.avi')
                shutil.copyfile(file_src, file_dst)


if __name__ == '__main__':
    # model = TimeSformer(img_size=224, num_classes=6, num_frames=8, attention_type='divided_space_time', pretrained_model=r'C:\Users\Gebruiker\Documents\GitHub\TimeSformer\checkpoints\checkpoint_epoch_00015.pyth')
    labels = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'fear', 4: 'disgust', 5: 'neutral'}
    filename = os.path.join('avi_videos', '1025_MTI_HAP_XX.avi')
    filename_arr = ['1025_MTI_HAP_XX.avi', '1003_TAI_SAD_XX.avi', '1052_TIE_ANG_XX.avi', '1053_DFA_FEA_XX.avi', '1014_TSI_DIS_XX.avi', '1014_IWW_NEU_XX.avi']

    src = r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D"
    Occlusion = OcclusionTest(src, None, visual_type='TIBAV')
    # total_pixels = 224 * 224
    # pixel_arr = [0.1 * total_pixels, 0.2 * total_pixels, 0.3 * total_pixels, 0.4 * total_pixels, 0.5 * total_pixels]
    # for i, num_pixels in enumerate(pixel_arr):
    #     print(f"Top {int(((i * 0.1) + 0.1) * 100)}%")
    #     Occlusion.occlude_max_min(int(num_pixels), is_max=True)
    #     Occlusion.apply_occlusion(str(int(((i * 0.1) + 0.1) * 100)), 'MMAX/0')
    #
    # for i, num_pixels in enumerate(pixel_arr):
    #     print(f"Bottom {int(((i * 0.1) + 0.1) * 100)}%")
    #     Occlusion.occlude_max_min(int(num_pixels), is_max=False)
    #     Occlusion.apply_occlusion(str(int(((i * 0.1) + 0.1) * 100)), 'MMIN/0')
    #
    # exit(1)

    # Occlusion.split_test()
    #
    # exit(1)

    # is_max = True
    # for i in range(1,5):
    #     for percent in [0, 10, 20, 30, 40, 50]:
    #         print(f"Num frames = {i}, percentage = {percent}, is_max = {is_max}")
    #         Occlusion.occlude_frames(num_frames=i, is_max=is_max)
    #         Occlusion.apply_occlusions_frame(num_frames=str(i), folder_name='MMAX/' + str(i), max_or_min='MMAX', percentage=str(percent))
    #
    # is_max = False
    # for i in range(1,5):
    #     for percent in [0, 10, 20, 30, 40, 50]:
    #         print(f"Num frames = {i}, percentage = {percent}, is_max = {is_max}")
    #         Occlusion.occlude_frames(num_frames=i, is_max=is_max)
    #         Occlusion.apply_occlusions_frame(num_frames=str(i), folder_name='MMIN/' + str(i), max_or_min='MMIN', percentage=str(percent))

    facial_parts = [
        ['Eyes'],
        ['Nose'],
        ['Mouth'],
        ['Eyes', 'Nose'],
        ['Eyes', 'Mouth'],
        ['Nose', 'Mouth'],
        ['Eyes', 'Mouth', 'Nose'],
    ]

    # facial_parts = [['Mouth']]

    for occ in facial_parts:
        print(occ)
        Occlusion.occlude_features(occ)
        filename = occ[0]
        for i in range(1, len(occ)):
            filename += '_' + occ[i]
        Occlusion.apply_occlusion(filename)


