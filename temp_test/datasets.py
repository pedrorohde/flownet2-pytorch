import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils

from imageio import imread

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class ImagesFromFolderInterpol(data.Dataset):
  def __init__(self, args, is_cropped=False, scanSubdir=False, root = '/path/to/frames/only/folder', iext = 'png', replicates = 1):
    self.args = args
    self.render_size = [-1,-1]
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.replicates = replicates
    
    self.in_imgs = []
    self.ref_imgs = []
    def parseTrainData(path):
        images = sorted( glob( join(path, '*.' + iext) ) )
        for i in range(0,len(images)-2, 2):
            im1 = images[i]
            ref = images[i+1]
            im2 = images[i+2]
            self.in_imgs += [ [ im1, im2 ] ]
            self.ref_imgs += [ [ ref ] ]

    if scanSubdir:
        print("WARNING: assuming that all samples have the same resolution")
        subdir_paths = [f.path for f in os.scandir(root) if f.is_dir()]
        for subdir in subdir_paths:
            parseTrainData(subdir)
    else:
        parseTrainData(root)

    self.size = len(self.in_imgs)
    print(f"Total samples: {self.size}")

    self.frame_size = frame_utils.read_gen(self.in_imgs[0][0]).shape
    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

    args.inference_size = self.render_size

    assert (len(self.in_imgs) == len(self.ref_imgs))

  def __getitem__(self, index):
    index = index % self.size

    in_img1 = frame_utils.read_gen(self.in_imgs[index][0])
    in_img2 = frame_utils.read_gen(self.in_imgs[index][1])
    ref_img = frame_utils.read_gen(self.ref_imgs[index][0])

    in_images = [in_img1, in_img2]
    ref_images = [ref_img]
    image_size = in_img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)

    in_images = list(map(cropper, in_images))
    in_images = np.array(in_images).transpose(3,0,1,2)
    in_images = torch.from_numpy(in_images.astype(np.float32))
    ref_images = list(map(cropper, ref_images))
    ref_images = np.array(ref_images).transpose(3,0,1,2)
    ref_images = torch.from_numpy(ref_images.astype(np.float32))

    return [in_images], [ref_images]

  def __len__(self):
    return self.size * self.replicates




class MpiSintel(data.Dataset):
    def __init__(self, args, is_cropped = False, root = '', dstype = 'clean', replicates = 1):
        self.args = args
        self.is_cropped = is_cropped
        self.crop_size = args.crop_size
        self.render_size = args.inference_size
        self.replicates = replicates

        flow_root = join(root, 'flow')
        image_root = join(root, dstype)

        file_list = sorted(glob(join(flow_root, '*/*.flo')))

        self.flow_list = []
        self.image_list = []

        for file in file_list:
            if 'test' in file:
                # print file
                continue

            fbase = file[len(flow_root)+1:]
            fprefix = fbase[:-8]
            fnum = int(fbase[-8:-4])

            img1 = join(image_root, fprefix + "%04d"%(fnum+0) + '.png')
            img2 = join(image_root, fprefix + "%04d"%(fnum+1) + '.png')

            if not isfile(img1) or not isfile(img2) or not isfile(file):
                continue

            self.image_list += [[img1, img2]]
            self.flow_list += [file]

        self.size = len(self.image_list)

        self.frame_size = frame_utils.read_gen(self.image_list[0][0]).shape

        if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
            self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
            self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        args.inference_size = self.render_size

        assert (len(self.image_list) == len(self.flow_list))

    def __getitem__(self, index):

        index = index % self.size

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = frame_utils.read_gen(self.flow_list[index])

        images = [img1, img2]

        image_size = img1.shape[:2]

        if self.is_cropped:
            cropper = StaticRandomCrop(image_size, self.crop_size)
        else:
            cropper = StaticCenterCrop(image_size, self.render_size)
        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates


class MpiSintelFinal(MpiSintel):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(MpiSintelFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'final', replicates = replicates)

