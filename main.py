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
from skimage import io

from utils import save_checkpoint, AverageMeter, visualize_image, GrayscaleImageFolder
from model import ColorNet
from config import config
best_losses = 1000.0
use_gpu = torch.cuda.is_available()

def main(args):
    global best_losses, use_gpu
    model = ColorNet()
    
    # Use GPU if available
    if use_gpu:
        model.cuda()
        print('Loaded model onto GPU.')
    
    # Create loss function, optimizer #criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
    criterion = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start_epoch=config['start_epoch']
    # Resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint {}...'.format(args.resume))
            checkpoint = torch.load(args.resume) if use_gpu else torch.load(args.resume, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_losses = checkpoint['best_losses']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Finished loading checkpoint. Resuming from epoch {}'.format(checkpoint['epoch']))
        else:
            print('Checkpoint filepath incorrect.')
            return
    
    if not args.evaluate:
        train_directory = config['image_folder_train']
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)
        train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'])
        print('Loaded training data.')
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])
    val_directory = config['image_folder_val']
    val_imagefolder = GrayscaleImageFolder(val_directory , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=config['batch_size'], shuffle=False, num_workers=config['workers'])
    print('Loaded validation data.')   

    if args.evaluate:
        save_images = True
        epoch = 0
        initial_losses = validate(val_loader, model, criterion, save_images, epoch)
        return  

    validate(val_loader, model, criterion, False, 0) # validate before training
    for epoch in range(start_epoch, config['epochs']):
        
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
        save_images = False
        if epoch % 10 == 0:
            save_images = True #(epoch % 3 == 0)
        losses = validate(val_loader, model, criterion, save_images, epoch)
        
        # Save checkpoint, and replace the old best model if the current model is better
        is_best_so_far = losses < best_losses
        best_losses = max(losses, best_losses)
        save_checkpoint({
            'epoch': epoch + 1,
            'best_losses': best_losses,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },config['save'], is_best_so_far, 'checkpoint-epoch-{}.pth.tar'.format(epoch))
        lr = optimizer.param_groups[0]['lr']
        train_logger.add_scalar('lr',  lr, epoch)
        train_logger.add_scalar('loss',  losses, epoch)
    return best_losses


def train(train_loader, model, criterion, optimizer, epoch):
    '''Train model on data in train_loader for a single epoch'''
    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to train mode
    model.train()
    
    # Train for single eopch
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        
        # Use GPU if available
        input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)
        input_ab_variable = Variable(input_ab).cuda() if use_gpu else Variable(input_ab)
        target_variable = Variable(target).cuda() if use_gpu else Variable(target)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable) # MSE
        
        # Record loss and measure accuracy
        losses.update(loss.item(), input_gray.size(0))
        
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % config['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses)) 

    print('Finished training epoch {}'.format(epoch))

def validate(val_loader, model, criterion, save_images, epoch):
    '''Validate model on data in val_loader'''
    print('Starting validation.')

    # Prepare value counters and timers
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    # Switch model to validation mode
    model.eval()
    
    # Run through validation set
    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        
        # Use GPU if available
        target = target.cuda() if use_gpu else target
        input_gray_variable = Variable(input_gray, volatile=True).cuda() if use_gpu else Variable(input_gray, volatile=True)
        input_ab_variable = Variable(input_ab, volatile=True).cuda() if use_gpu else Variable(input_ab, volatile=True)
        # target_variable = Variable(target, volatile=True).cuda() if use_gpu else Variable(target, volatile=True)
        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray_variable) # throw away class predictions
        loss = criterion(output_ab, input_ab_variable) # check this!
        
        # Record loss and measure accuracy
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images:
            for j in range(len(output_ab)):
                save_path = {'grayscale': 'D:\\Colorization-master\\Colorization-resnet\\test_outputs\\gray\\', 'colorized': 'D:\\Colorization-master\\Colorization-resnet\\test_outputs\\color\\', 'original':'D:\\Colorization-master\\Colorization-resnet\\test_outputs\\ori\\'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path, save_name=save_name)

        # Record time to do forward passes and save imagesD:\Colorization-master\Colorization-resnet
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % config['print_freq'] == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--valitate','-V', type=bool, default=True, help='Decide whether to use cross validation')
    parser.add_argument('--train','-T', type=bool, default=True, help='Decide to train or not')
    # parser.add_argument('--pretrained','-P', type=bool, default=True, help='use pretrained model or not')
    parser.add_argument('--resume','-R', default='', type=str, metavar='PATH', help='Specified resume path to .pth')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='use this flag to validate without training')
    args = parser.parse_args()
    train_logger = SummaryWriter(log_dir = os.path.join(config['save'], 'train'), comment = 'training')
    print(args)
    main(args)