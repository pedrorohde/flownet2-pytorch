import numpy as np
from os.path import *
# from scipy.misc import imread
from . import flow_utils 
from imageio import imread
def read_gen(file_name):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []

import random
class Image_transform:
    def __init__(self, sel=None):
        reverse_opt = [False, True]    
        flip_opt = [None, np.flipud, np.fliplr]
        rotate_opt = [None, 1, 2, 3]
        if sel:
            self.reverse = reverse_opt[sel[0]]
            self.flip_f = flip_opt[sel[1]]
            self.rotate = rotate_opt[sel[2]]
        else: 
            self.reverse = random.choice(reverse_opt)
            self.flip_f = random.choice(flip_opt)
            self.rotate = random.choice(rotate_opt)

    def __call__(self, imgs):
        if self.reverse:
            imgs_t = [imgs[2], imgs[1], imgs[0]]
        else:
            imgs_t = [imgs[0], imgs[1], imgs[2]]

        if self.flip_f != None:
            for i in range(3):
                imgs_t[i] = self.flip_f(imgs_t[i])

        if self.rotate != None:
            for i in range(3):
                imgs_t[i] = np.rot90(imgs_t[i], self.rotate)

        return imgs_t