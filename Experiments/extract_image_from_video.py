"""
This file contains code to extract frames from a video.

This code was written and designed by Christopher du Toit.
"""

import av
import os
from timesformer.datasets.decoder import decode
from PIL import Image

def extract_image_from_video(filename, frame_number, save_path):
    filename = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D", filename)
    video_container = av.open(filename)
    video_container = decode(video_container, 32, 8, -1, 1, )
    image = Image.fromarray(video_container[frame_number].numpy())
    save_path = os.path.join(r"C:\Users\Gebruiker\Documents\GitHub\Research_internship\CREMA_D", save_path)
    image.save(save_path)

    
