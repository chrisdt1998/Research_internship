"""
This code contains the results of the occlusion testing as well as functions to plot the results.

This code was written and designed by Christopher du Toit.
"""
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from timesformer.visualization.utils import get_confusion_matrix, plot_confusion_matrix

percentage = [0, 10, 20, 30, 40, 50]
frames = [i for i in range(5)]

TIBAV_mix_max = [
    [81.02, 80.48, 79.41, 78.61, 73.53],
    [65.24, 61.23, 59.09, 56.42, 53.21],
    [51.60, 47.59, 42.51, 36.36, 31.55],
    [42.51, 39.04, 37.70, 36.63, 35.56],
    [36.90, 33.96, 34.49, 32.89, 35.56],
    [31.55, 31.82, 32.09, 32.89, 33.69]
                 ]
TIBAV_mix_max = np.array(TIBAV_mix_max)

TIBAV_mix_min = [
    [81.02, 80.21, 81.55, 78.88, 78.34],
    [80.21, 77.81, 79.14, 77.54, 73.50],
    [75.40, 75.40, 76.20, 72.46, 68.98],
    [74.87, 73.26, 74.33, 71.66, 66.84],
    [72.46, 71.12, 70.05, 66.84, 60.96],
    [71.39, 70.05, 67.65, 64.97, 59.89]
]
TIBAV_mix_min = np.array(TIBAV_mix_min)

GradCAM_mix_max =[
    [81.02, 79.14, 78.34, 77.27, 71.66],
    [67.91, 67.38, 63.64, 63.10, 58.56],
    [52.14, 57.49, 56.68, 56.15, 49.73],
    [40.91, 52.41, 52.14, 50.27, 47.59],
    [33.69, 41.71, 41.71, 45.45, 41.71],
    [24.87, 30.75, 31.55, 36.10, 38.77]
]
GradCAM_mix_max = np.array(GradCAM_mix_max)

GradCAM_mix_min =[
    [81.02, 81.55, 81.82, 79.95, 79.14],
    [72.73, 71.12, 69.52, 66.31, 63.37],
    [66.04, 61.50, 57.49, 48.40, 41.44],
    [71.39, 65.78, 60.43, 54.28, 45.72],
    [64.44, 60.70, 56.68, 51.60, 47.06],
    [59.36, 57.75, 54.01, 50.53, 47.33]
]
GradCAM_mix_min = np.array(GradCAM_mix_min)

Rollout_mix_max = [
    [81.02, 81.82, 81.82, 80.75, 72.19],
    [64.17, 63.90, 62.03, 54.28, 50.53],
    [52.67, 43.58, 40.37, 29.68, 28.34],
    [46.52, 37.70, 30.21, 25.13, 21.66],
    [37.43, 36.10, 30.09, 33.42, 32.62],
    [31.82, 27.27, 18.18, 19.58, 16.58]
]
Rollout_mix_max = np.array(Rollout_mix_max)

Rollout_mix_min = [
    [81.02, 79.95, 80.48, 79.95, 76.47],
    [77.27, 77.27, 76.74, 73.26, 68.72],
    [76.20, 75.67, 74.03, 69.79, 65.24],
    [71.39, 72.46, 70.59, 65.51, 59.63],
    [68.72, 69.25, 67.91, 64.71, 56.15],
    [67.38, 66.58, 67.38, 60.43, 51.60]
]
Rollout_mix_min = np.array(Rollout_mix_min)

Rollout_grad_mix_max = [
    [81.02, 78.88, 79.95, 79.41, 71.39],
    [67.91, 66.04, 62.30, 55.88, 47.86],
    [60.43, 56.42, 53.21, 44.92, 37.97],
    [40.64, 34.49, 32.35, 28.88, 27.81],
    [29.41, 26.47, 27.54, 26.74, 27.27],
    [27.27, 29.14, 28.88, 31.02, 31.02]
]
Rollout_grad_mix_max = np.array(Rollout_grad_mix_max)

Rollout_grad_mix_min = [
    [81.02, 79.95, 80.48, 80.21, 78.61],
    [78.07, 78.61, 77.54, 75.67, 71.12],
    [74.87, 76.20, 74.87, 71.66, 69.25],
    [74.60, 73.53, 72.46, 71.39, 65.51],
    [72.46, 70.86, 70.05, 66.84, 62.83],
    [63.90, 65.24, 64.71, 63.64, 58.29]
]
Rollout_grad_mix_min = np.array(Rollout_grad_mix_min)

def plot_mix():
    for i, percent in enumerate(percentage):
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(f"{percent}% Spacial Occlusion")
        ax.plot(frames, TIBAV_mix_max[i], marker='x', color='b', linestyle='dashed', label="TIBAV MAX")
        ax.plot(frames, GradCAM_mix_max[i], marker='x', color='r', linestyle='dashed', label="GradCAM MAX")
        ax.plot(frames, Rollout_mix_max[i], marker='x', color='g', linestyle='dashed', label="Rollout MAX")
        ax.plot(frames, Rollout_grad_mix_max[i], marker='x', color='black', linestyle='dashed', label="Rollout Grad MAX")

        ax.plot(frames, TIBAV_mix_min[i], marker='+', color='b', linestyle='dashed', label="TIBAV MIN")
        ax.plot(frames, GradCAM_mix_min[i], marker='+', color='r',linestyle='dashed', label="GradCAM MIN")
        ax.plot(frames, Rollout_mix_min[i], marker='+', color='g', linestyle='dashed', label="Rollout MIN")
        ax.plot(frames, Rollout_grad_mix_min[i], marker='+', color='black', linestyle='dashed', label="Rollout Grad MIN")

        ax.plot(frames, TIBAV_mix_min[i] - TIBAV_mix_max[i], marker='o', color='b', label="TIBAV DIFF")
        ax.plot(frames, GradCAM_mix_min[i] - GradCAM_mix_max[i], marker='o', color='r', label="GradCAM DIFF")
        ax.plot(frames, Rollout_mix_min[i] - Rollout_mix_max[i], marker='o', color='g', label="Rollout DIFF")
        ax.plot(frames, Rollout_grad_mix_min[i] - Rollout_grad_mix_max[i], marker='o', color='black', label="Rollout Grad DIFF")

        plt.legend(["TIBAV MAX", "GradCAM MAX", "Rollout MAX", "Rollout Grad MAX", "TIBAV MIN", "GradCAM MIN", "Rollout MIN", "Rollout Grad MIN", "TIBAV DIFF", "GradCAM DIFF", "Rollout DIFF", "Rollout Grad MAX Diff"], loc='upper right')
        plt.xlabel("Number of frames occluded")
        plt.ylabel("Model accuracy (%)")
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()

def plot_diff():
    for i, percent in enumerate(percentage):
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(f"{percent}% Spacial Occlusion")

        ax.plot(frames, TIBAV_mix_min[i] - TIBAV_mix_max[i], marker='o', color='b', label="TIBAV DIFF")
        ax.plot(frames, GradCAM_mix_min[i] - GradCAM_mix_max[i], marker='o', color='r', label="GradCAM DIFF")
        ax.plot(frames, Rollout_mix_min[i] - Rollout_mix_max[i], marker='o', color='g', label="Rollout DIFF")
        ax.plot(frames, Rollout_grad_mix_min[i] - Rollout_grad_mix_max[i], marker='o', color='black', label="Rollout Grad DIFF")

        plt.legend(["GradCAM DIFF", "Rollout DIFF", "Rollout Grad MAX Diff"], loc='upper right')
        plt.xlabel("Number of frames occluded")
        plt.ylabel("Difference in accuracies (%)")
        plt.ylim(0, 50)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.show()


def check_accuracy(true_labels, preds_labels):
    pred_labels = torch.argmax(preds_labels, dim=1)
    count = 0
    for i in range(pred_labels.shape[0]):
        if pred_labels[i] == true_labels[i]:
            count += 1
    return count/pred_labels.shape[0] * 100


Eyes = 62.30
Eyes_Mouth = 35.83
Eyes_Mouth_Nose = 28.88
Eyes_Nose = 55.88
Mouth = 62.30
Nose = 76.47
Nose_Mouth = 60.96

# true_labels = []
# with open("CREMA_D/avi_videos/test.csv") as r:
#     csv_file = csv.reader(r, delimiter=' ')
#     for index, row in enumerate(csv_file):
#         true_labels.append(int(row[1]))
# true_labels = np.array(true_labels)


def confusion_matrix(face_occ):
    file = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D\Occlusion\Face_parts\preds_labels", face_occ)
    with open(file, "rb") as f:
        outputs = pickle.load(f)

    preds = outputs[0]
    true_labels = outputs[1]
    class_names = ['happy', 'sad', 'anger', 'fear', 'disgust', 'neutral']
    print(check_accuracy(true_labels, preds))

    cmtx = get_confusion_matrix(preds, true_labels, 6)
    figure, plt = plot_confusion_matrix(cmtx, 6, class_names, title="Parts occluded = " + face_occ)
    plt.show()
    return figure

# import matplotlib; matplotlib.use("TkAgg")
facial_parts = ['No_Occ', 'Eyes', 'Eyes_Mouth', 'Eyes_Mouth_Nose', 'Eyes_Nose', 'Mouth', 'Nose', 'Nose_Mouth']

for part in facial_parts:
    confusion_matrix(part)

# plot_diff()
