"""
This file contains useful code to preprocess the dataset CREMA-D for the TimeSformer.

This code was written and designed by Christopher du Toit.
"""

import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


import sys
sys.path.insert(1, r'C:\Users\Gebruiker\Documents\GitHub')

from facenet_pytorch import MTCNN, InceptionResnetV1

class Preprocess:
    def __init__(self, folder):
        self.failed_videos = []
        self.num_frames = 8
        self.folder = folder

    def check_file_exists(self, filename, folder, file_ending, target_file_ending):
        filename = filename.replace(target_file_ending, file_ending)
        filepath = os.path.join(folder, filename)
        return os.path.isfile(filepath)

    def crop_frame(self, image, mtcnn):
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        img_cropped = mtcnn(image)
        if img_cropped is None:
            return None
        img_arr = img_cropped.numpy()
        img_arr = np.transpose(img_arr, (1, 2, 0))
        norm_image = cv2.normalize(img_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return norm_image

    def video_to_frames(self, video):
        vidcap = cv2.VideoCapture(video)
        if not vidcap.isOpened():
            print("Error: Could not open file: %s" % (video))
            return None
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(length / self.num_frames)
        success, image = vidcap.read()
        # print(success)
        count = 0
        frames = []
        while success:
            if count % frame_step == 0 and count != 0:
                frames.append(image)

            success, image = vidcap.read()
            count += 1
        return np.array(frames, dtype=object)

    def crop_videos(self, src, dst, number_of_frames=8):
        # If required, create a face detection pipeline using MTCNN:
        image_size = 224
        mtcnn = MTCNN(image_size=image_size, margin=0)
        for root, dirs, filenames in os.walk(src, topdown=False):
            for filename in tqdm(filenames):
                if self.check_file_exists(filename, folder="numpy_videos", file_ending=".npy", target_file_ending=".mp4") is False:
                    print("flv opened: ", filename)
                    video = os.path.join(root, filename)
                    frames = self.video_to_frames(video)
                    # Here the frame will be cropped to the face. However, if there is no face in the frame, then it
                    # will simply be a duplicate of the frame before or after such that 8 frames is maintained.
                    if frames is not None:
                        cropped_frames = []
                        cropped_frame = []
                        use_next_frame = False
                        for frame in frames:
                            previous_frame = cropped_frame
                            cropped_frame = self.crop_frame(frame, mtcnn)
                            if cropped_frame is None:
                                if previous_frame is None or len(previous_frame) == 0:
                                    use_next_frame = True
                                else:
                                    cropped_frames.append(previous_frame)
                                    self.failed_videos.append(filename)
                            else:
                                if use_next_frame:
                                    use_next_frame = False
                                    cropped_frames.append(cropped_frame)
                                cropped_frames.append(cropped_frame)

                        cropped_frames = np.stack(cropped_frames, axis=0)
                        filename = filename.replace(".mp4", "")
                        file_location = os.path.join(dst, filename + '.npy')
                        with open(file_location, 'wb') as f:
                            # print("npy saved at ", file_location)
                            np.save(f, cropped_frames)

    def np_to_img(self, video_arr, filename):
        for i, frame in enumerate(video_arr):
            cv2.imwrite("jpg_custom/" + "image" + str(i) + "_" + str(filename) + ".jpg", frame)

    def img_to_video(self, image_folder, video_name, video_path):
        images = [img for img in os.listdir(image_folder) if img.endswith(video_name + ".jpg")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_path + video_name + ".avi", 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    def np_to_video(self, src):
        for root, dirs, filenames in os.walk(src, topdown=False):
            for filename in tqdm(filenames):
                if self.check_file_exists(filename, folder="avi_videos", file_ending=".avi", target_file_ending=".npy") is False:
                    print("opening numpy:", os.path.join(src, filename))
                    with open(os.path.join(src, filename), 'rb') as r:
                        video_arr = np.load(r)
                    r.close()
                    filename = filename.lower().replace(".npy", "")
                    self.np_to_img(video_arr=video_arr, filename=filename)
                    self.img_to_video(image_folder="jpg_custom", video_name=filename, video_path="avi_custom/")

    def start(self):
        self.crop_videos()
        self.np_to_video()




