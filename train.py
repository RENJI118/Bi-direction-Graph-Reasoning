import SimpleITK as sitk
import os
import argparse
from glob import glob
from collections import OrderedDict
import warnings

import numpy as np
from tqdm import tqdm
from thop import profile
from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.Dataset import Dataset

from dataset.metric import dice_coef, iou_score
from dataset import loss
from model.layer import count_parameters,str2bool
import pandas as pd
from model.layer import get_norm_layer
from model import unet
from dataset.loss import BCEDiceLoss,Cross_entropy_loss




arch_names = list(unet.__dict__.keys())
loss_names = list(loss.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

IMG_PATH = glob("/root/autodl-tmp/data-brats/outputImg/*.npy")
MASK_PATH = glob("/root/autodl-tmp/data-brats/outputMask/*.npy")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='model',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: model)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="brats19",
                        help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=10, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--num_classes', default=4, type=int,
                        help='num_classes')
    parser.add_argument('--width', default=64, type=int,
                        help='width')
    parser.add_argument('--norm_layer', default='group')
    parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
    parser.add_argument('--resume', default='', help='checkpoint')    

    args = parser.parse_args()

    return args



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    dices_s = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):

        input = input.cuda()
        target = target.cuda()
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input, target = input.to(device), target.to(device)
        
        # flops, params = profile(model, inputs=(input,))
        # print(f"Model FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 Giga FLOPs
        # print(f"Model Parameters: {params / 1e6:.2f} M")  # 转换为 Million
        

        region, out_region,out_contour = model(input)


        # compute output
        loss_region = criterion(out_region, target)
        loss_contour =Cross_entropy_loss(out_contour, target)
        loss_fusion = criterion(region, target) 

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        total_loss = loss_contour + loss_region + loss_fusion
        total_loss.backward()
        dice = dice_coef(region, target)
        optimizer.step()

        losses.update(total_loss.item(), input.size(0))
        
        dices_s.update(torch.tensor(dice), input.size(0))
     


    log = OrderedDict([
        ('loss', losses.avg)
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    dices_s = AverageMeter()


    # switch to evaluate mode
    model.eval()

    for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):

        input = input.cuda()
        target = target.cuda()
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input, target = input.to(device), target.to(device)
        

        region, out_region,out_contour = model(input)

        # compute output
        loss_region = criterion(out_region, target)
        loss_contour =Cross_entropy_loss(out_contour, target)
        loss_fusion = criterion(region, target) 

        # compute gradient and do optimizing step
        total_loss = loss_contour + loss_region + loss_fusion
        total_loss.backward()
        

        # iou=iou_score(out_contour, target)
        dice = dice_coef(region, target)
   

        losses.update(total_loss.item(), input.size(0))
        dices_s.update(torch.tensor(dice), input.size(0))
        


    log = OrderedDict([
        ('loss', losses.avg)
    ])

   
    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)
            

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    
    criterion=BCEDiceLoss().cuda()

    cudnn.benchmark = True

    # Data loading code
    img_paths = IMG_PATH
    mask_paths = MASK_PATH


    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
            train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    # create model
    print("=> creating model %s" % args.arch)
    model_maker = getattr(unet, args.arch)

    model = model_maker(
        4, 3,
        width=args.width, deep_supervision=args.deepsupervision,
        norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    model = model.cuda()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for param in model.parameters():
        param = param.to(device)

    print(count_parameters(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        


    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'val_loss'
    ])


    torch.cuda.empty_cache()
    
    best_loss = 100
    best_iou = 0
    trigger = 0

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)

        # scheduler.step()
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        # print('loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
        #           %(train_log['loss'],train_log['dice_1'], train_log['dice_2'], val_log['loss'], val_log['dice_1'], val_log['dice_2']))

        # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
        print('loss %.4f - val_loss %.4f 
              % (train_log['loss'], val_log['loss']))


        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            val_log['loss'],
        ], index=['epoch', 'lr', 'loss', 'val_loss''])

        log = pd.concat([log, tmp], ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name,timestamp), index=False)
        

        trigger += 1

        val_loss = val_log['loss']
        # if val_loss < best_loss:
        torch.save(model.state_dict(), 'models/%s/model.pth' % args.name)
        best_loss = val_loss
        print("=> saved best model")
        trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()
    




    
    
if __name__ == '__main__':
    main()
