import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import cm

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Volumes/ExtremeSSDChris')

from facenet_pytorch import MTCNN, InceptionResnetV1


def crop_frame(image, mtcnn):
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    img_cropped = mtcnn(image)

    img_arr = img_cropped.numpy()
    # print(np.rollaxis(img_arr, 0, 3).shape)
    # img_arr = np.rollaxis(img_arr,0,3)
    img_arr = np.transpose(img_arr, (1, 2, 0))
    norm_image = cv2.normalize(img_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # pil_image = norm_image.astype(np.uint8)
    # pil_image = Image.fromarray(pil_image, 'RGB')
    # pil_image
    # print("shape shape ==", norm_image.shape)

    return norm_image


def video_to_frames(video, number_of_frames):
    vidcap = cv2.VideoCapture(video)
    if not vidcap.isOpened():
        print("Error: Could not open file: %s" % (video))
        return None
    # print(vidcap)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(length / number_of_frames)
    # print("Total number of frames is", length)
    # print("Number of frames extracted is", number_of_frames)
    # print("Frame step is", frame_step)
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


def crop_videos(src, dst, number_of_frames=8):
    # If required, create a face detection pipeline using MTCNN:
    image_size = 224
    mtcnn = MTCNN(image_size=image_size, margin=0)
    counter = 0
    for root, dirs, filenames in os.walk(src, topdown=False):
        for filename in filenames:
            video = os.path.join(root, filename)
            # print(video)
            frames = video_to_frames(video, number_of_frames)
            # print(frames)
            if frames is not None:
                cropped_frames = []
                for frame in frames:
                    cropped_frames.append(crop_frame(frame, mtcnn))

                cropped_frames = np.stack(cropped_frames, axis=0)
                filename = filename.lower().replace(".flv", "")
                file_location = os.path.join(dst, filename + '.npy')
                with open(file_location, 'wb') as f:
                    print("npy saved at ", file_location)
                    np.save(f, cropped_frames)


def np_to_img(video_arr, filename):
    # cv2.imwrite("tester.jpg", video_arr[0])
    for i, frame in enumerate(video_arr):
        # cv2.imwrite("jpg_files/image" + str(i) + ".jpg", frame)
        cv2.imwrite("jpg_files/" + "image" + str(i) + "_" + str(filename) + ".jpg", frame)


def img_to_video(image_folder, video_name, video_path):
    # image_folder = 'jpg_files'
    # video_name = 'tester.avi'
    print(video_name + ".jpg®")
    images = [img for img in os.listdir(image_folder) if img.endswith(video_name + ".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_path + video_name + ".avi", 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def np_to_video(src):
    counter = 0
    for root, dirs, filenames in os.walk(src, topdown=False):
        for filename in filenames:
            print("opening numpy:", os.path.join(src, filename))
            with open(os.path.join(src, filename), 'rb') as r:
                video_arr = np.load(r)
            r.close()
            filename = filename.lower().replace(".npy", "")
            np_to_img(video_arr=video_arr, filename=filename)
            img_to_video(image_folder="jpg_files", video_name=filename, video_path="avi_videos/")


src = '/Volumes/ExtremeSSDChris/CREMA-D/VideoFlash'
dst = 'numpy_videos'

crop_videos(src, dst, 8)

src = "numpy_videos"
dst = 'avi_videos/'

np_to_video(src)



