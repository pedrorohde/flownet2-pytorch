# python -i main.py --training_dataset ImagesFromFolderInterpol --training_dataset_root PATH --training_dataset_scanSubdir True
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse, os, sys, subprocess
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from glob import glob
from os.path import *

import datasets
import models
from utils import tools


def show_rgb(img):
    from PIL import Image
    nimg = Image.fromarray(img.astype('uint8'))
    nimg.show()

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")

tools.add_arguments_for_module(parser, models, argument_for_class='model', default='TestNet')
tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal', 
                                skip_params=['is_cropped'],
                                parameter_defaults={'root': './MPI-Sintel/flow/training'})


args = parser.parse_args()
args.training_dataset_class = tools.module_to_dict(datasets)[args.training_dataset]


train_dataset = args.training_dataset_class(args, **tools.kwargs_from_args(args, 'training_dataset'))
print('Training Dataset: {}'.format(args.training_dataset))
print('Training Input: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][0]])))
print('Training Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in train_dataset[0][1]])))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# test_images, test_labels = next(iter(train_loader))
# in_pair = test_images[0][0].numpy().transpose(1,2,3,0)
# img1 = in_pair[0]
# show_rgb(img1)

args.model_class = tools.module_to_dict(models)[args.model]


from torchsummary import summary
model = args.model_class()
# in_shape = (3, 100, 100)
# summary(model, in_shape)

def inference(args, epoch, data_loader, model, offset=0):
    model.eval()
    
    inference_n_batches = 1

    progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), inference_n_batches), desc='Inferencing ', 
        leave=True, position=offset)

    statistics = []
    total_loss = 0
    for batch_idx, (data, target) in enumerate(progress):
        print("IDX")
        data, target = [Variable(d) for d in data], [Variable(t) for t in target]

    with torch.no_grad():
        output = model(data[0])
    progress.close()

    return
# title = 'Validating Epoch {}'.format(epoch)
# progress = tqdm(tools.IteratorTimer(train_loader), ncols=100, total=np.minimum(len(train_loader), 1), leave=True, position=0, desc="BANANA"

# Primary epoch loop
best_err = 1e8
progress = tqdm(list(range(0, 0 + 1)), miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
offset = 1
last_epoch_time = progress._time()
global_iteration = 0
# inference(args,0 , train_dataset, model)

for epoch in progress:
    stats = inference(args=args, epoch=epoch - 1, data_loader=train_loader, model=model, offset=offset)
    offset += 1