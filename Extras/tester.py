import numpy as np
import av
import os
import matplotlib.pyplot as plt
from timesformer.datasets.decoder import decode

path_to_vid = '1002_ITS_NEU_XX.avi'
path_to_vid = os.path.join(r'C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D\Occlusion\TIBAV\MAX\10', path_to_vid)
path_to_vid = r'C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D\Occlusion\TIBAV\MAX\10\1002_ITS_NEU_XX.avi.avi'
container = av.open(path_to_vid)
video = decode(container, 32, 8, -1, 1, )
video = video.numpy()
for frame in video:
    print(frame.shape)
    plt.imshow(frame)
    plt.show()