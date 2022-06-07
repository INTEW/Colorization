# -*- coding: utf-8 -*-

import os, time, shutil, argparse
from functools import partial
import pickle
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from PIL import Image
from skimage import io

from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
from model import ColorNet
from config import config
use_gpu = torch.cuda.is_available()

def main(args):
    global  use_gpu
    model = ColorNet()
    
    # Use GPU if available
    if use_gpu:
        model.cuda()
        print('Loaded model onto GPU.')
    
    if os.path.isfile(args.resume):
        print('Loading checkpoint {}...'.format(args.resume))
        checkpoint = torch.load(args.resume) if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        print('Finished loading checkpoint. Resuming from epoch {}'.format(checkpoint['epoch']))
    else:
        print('Checkpoint filepath incorrect.')
        return
    
    img_original = np.asarray(Image.open(args.image).convert('RGB'))
    img_original = torch.from_numpy(img_original).unsqueeze(0).float()
    input_gray_variable = Variable(img_original, volatile=True).cuda() if use_gpu else Variable(img_original, volatile=True)
    end = time.time()
    output_ab = model(input_gray_variable) # throw away class predictions
    color_time =time.time() - end
    
    color_image = torch.cat((img_original, output_ab), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))  
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
    color_image = lab2rgb(color_image.astype(np.float64))
    plt.imsave(arr=color_image, fname='output.jpg')
    print(color_time)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume','-R', default='', type=str, metavar='PATH', help='Specified resume path to .pth')
    parser.add_argument('-image', '-i', default='', type=str, metavar='PATH', help='gray scale image')
    args = parser.parse_args()
    print(args)
    main(args)