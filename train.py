from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def cmd_parser():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
    # train_set = parser.add_mutually_exclusive_group()
    # 训练集与基础网络设定
    parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                        type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default=os.path.expanduser('~') + '/Dataset/VOC0712trainval',
                        help='Dataset root directory path')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    # 文件保存路径
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_iter', default=2, type=int,
                        help='Directory for saving checkpoint models')
    # 恢复训练
    parser.add_argument('--basenet', default='vgg16_withoutfc.pkl',
                        help='Pretrained base model')
    parser.add_argument('--weight_file', default='VOC.pth', type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--para_file', default='para.pth', type=str,
                        help='Checkpoint para_dict file to resume training from')
    # 优化器参数设置
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')

    args = parser.parse_args()
    return args, parser


def get_dataset(args, parser):
    cfg = get_cfg(args)
    dataset = None
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))
    return data.DataLoader(dataset, args.batch_size,
                           num_workers=args.num_workers,
                           shuffle=True, collate_fn=detection_collate,
                           pin_memory=True)


def get_cfg(args):
    cfg = None
    if args.dataset == 'COCO':
        cfg = coco
        pass
    elif args.dataset == 'VOC':
        cfg = voc
    return cfg


def load_para(args):
    para = None
    para_path = os.path.join(args.save_folder, args.para_file)
    print('===> Checking the para_path:{}'.format(para_path))
    if os.path.exists(para_path):
        para = torch.load(para_path)
        print('===> Exist, loading the para...')
    else:
        print('===> Not exist, initilizing the para...')
        para = {
            'iter': 200000,
            'optimizer_param': None,
            'loss_list': [],
            'result_list': [],
        }
        save_safely(para, filename=args.para_file, folder=args.save_folder)
    return para


def load_model(args):
    # 恢复或者新建模型
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    cfg = get_cfg(args)
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    ssd_net = ssd_net.to(DEVICE)

    if args.weight_file is not None:
        print('Resuming training, loading {}...'.format(args.weight_file))
        ssd_net.load_weights(os.path.join(args.save_folder, args.weight_file))
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
    return ssd_net


def save_safely(file, filename, folder):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        temp = os.path.join(folder, filename + '.temp')
        torch.save(file, temp)
        os.remove(path)
        os.rename(temp, path)
        pass
    else:
        torch.save(file, path)


def train(args, parser):
    para = load_para(args)
    ssd_net = load_model(args)
    loss_list = para['loss_list']
    result_list = para['result_list']
    iter = para['iter']
    cfg = get_cfg(args)
    optimizer = optim.SGD(ssd_net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    if para['optimizer_param'] is not None:
        print('===> Loading the saved optimizer para')
        print(para['optimizer_param'])
        optimizer.state_dict().update(para['optimizer_param'])
    else:
        print('===> Using initial optimizer para')

    adjust_learning_rate(optimizer, args, cfg=get_cfg(args), iter=iter)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, use_gpu=torch.cuda.is_available())

    # 得到训练集
    print('Loading the dataset...')
    data_loader = get_dataset(args, parser)

    step_index = 0
    print('===> Batch_num:{}, '.format(len(data_loader)))

    ssd_net.train()
    loc_loss = 0
    conf_loss = 0

    while True:
        for _, batch_iterator in enumerate(data_loader, para['iter']):
            iter += 1
            t0 = time.perf_counter()
            # load train data
            images, targets = batch_iterator
            images, targets = images.to(DEVICE), [i.to(DEVICE) for i in targets]
            # print(images.device, targets[0].device)
            out = ssd_net(images)
            # for i in out:
            #     i.to(DEVICE)
            #     print(i.device)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            # loc_loss += loss_l.item()
            # conf_loss += loss_c.item()

            loss_list.append(loss.item())
            t1 = time.perf_counter()
            if iter != 0 and iter % args.save_iter == 0:
                print('Saving state, iter:', iter)
                save_safely(ssd_net.state_dict(), filename=args.weight_file, folder=args.save_folder)
                save_safely({
                    'iter': iter,
                    'optimizer_param': optimizer.state_dict()['param_groups'][0],
                    'loss_list': loss_list,
                    'result_list': result_list,
                }, filename=args.para_file, folder=args.save_folder)
                adjust_learning_rate(optimizer, args, cfg=get_cfg(args), iter=iter)
            print('timer: {:.4f} sec'.format(t1 - t0))
            print('iter ' + repr(iter) + ' || Loss: %.4f ||' % (loss.item()), end=' ')


def adjust_learning_rate(optimizer, args, cfg, iter):
    """
    Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    step = 0
    for i in cfg['lr_steps']:
        if iter > i:
            step += 1
        else:
            break
    lr = args.lr * (args.gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('===> Adjust_learning_rate:{}'.format(optimizer.param_groups[0]['lr']))


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    args, parser = cmd_parser()
    train(args, parser)
