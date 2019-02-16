import os
import sys
import time
import glob
import numpy as np
import torch
from cell_utils import AvgrageMeter, accuracy, create_exp_dir, \
    save, make_optimizer, count_parameters_in_MB
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from datasets import CRCDataLoader
from model.cellnet import CellDet
from model.loss import CellDetLoss
from evaluate import eval

parser = argparse.ArgumentParser()
parser.add_argument("--extend", type=bool, default=True)
parser.add_argument("--image_per_gpu", type=int, default=2)
parser.add_argument("--epoch", default=300, type=int)
parser.add_argument("--det_loss_weight", default=1, type=int)
parser.add_argument("--type", default='focal')
parser.add_argument("--backbone", default='resnet50', type=str)
parser.add_argument("--test_img", default=15, type=int)
parser.add_argument("--data", default='crop', type=str)
parser.add_argument("--det_weight", type=float, default=0.1)

parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def setup_logger(logfile='log.txt'):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, logfile))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return logging

train_logger = setup_logger('train_log.txt')
test_logger = setup_logger('test_log.txt')


def main():
    if not torch.cuda.is_available():
      train_logger.info('no gpu device available')
      sys.exit(1)

    np.random.seed(args.seed)
    #torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    train_logger.info("args = %s", args)

    weight_det = torch.Tensor([0.1, 0.9])
    #criterion = nn.NLLLoss(weight=weight_det)#nn.CrossEntropyLoss()
    criterion = CellDetLoss()#nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = CellDet()
    #if args.ngpus > 1:
    model = model.cuda()

    devices = list(range(torch.cuda.device_count()))
    args.batch_size = args.batch_size * len(devices)
    sync = True if len(devices) > 1 else False
    if sync:
        model = nn.DataParallel(model)

    train_logger.info("param size = %fMB", count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, nesterov=True)# make_optimizer(args, model)


    args.data = os.path.join(os.getcwd(), 'data')
    print('roooooooot: ', args.data)
    #print('roooooooot: ', os.path.expanduser(args.data))
    train_data = CRCDataLoader(type='train', imgdir=args.data)
    valid_data = CRCDataLoader(type='validation', imgdir=args.data)
    test_data = CRCDataLoader(type='test', imgdir=args.data)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                              pin_memory=True, num_workers=4)
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                              pin_memory=True, num_workers=4)
    test_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                             pin_memory=True, num_workers=4)
    # use initial learning rate
    lr = args.learning_rate
    print('start training')
    for epoch in range(args.epochs):
        train_logger.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        train_logger.info('train_f1 %f train_obj %f', train_acc, train_obj)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        train_logger.info('valid_f1 %f valid_obj %f', valid_acc, valid_obj)

        save(model, os.path.join(args.save, 'weights.pt'))
    test_acc, test_obj = infer(test_queue, model, criterion)
    train_logger.info('test_f1 %f valid_obj %f', test_acc, test_obj)


def train(train_queue, model, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        if torch.cuda.is_available():
            input = input.float()
            input = input.cuda()
            #print('target.shape: ', target.shape)
            target = target.long()
            target = target.cuda(async=True)


        optimizer.zero_grad()

        logits = model(input)
        #print('output: ', logits)
        loss = criterion(target, logits)
        loss.backward()
        optimizer.step()

        #prec1= accuracy(logits, target)
        objs.update(loss.data, n)
        #top1.update(prec1.data, n)

        #if step % args.report_freq == 0:
        #    train_logger.info('train step:%03d loss:%e', step, objs.avg)
            #train_logger.info('train step:%03d weight_loss:%e acc:%f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()

    with torch.no_grad():
        for step, (input, target, gt) in enumerate(valid_queue):
            if torch.cuda.is_available():
                input = input.float()
                input = input.cuda()

                target = target.long()
                target = target.cuda()

            logits = model(input)
            loss = criterion(target, logits)

            target = target.cpu().detach().numpy()
            logits = logits.cpu().detach().numpy()
            gt = gt.cpu().detach().numpy()
            prec1 = eval(output=logits, gt=gt, mask=target, prob_thresh=0.7)
            #prec1= accuracy(logits, target)
            prec1 = torch.Tensor([prec1])
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            #if step % args.report_freq == 0:
            #    train_logger.info('valid step:%03d loss:%e acc:%f', step, objs.avg, top1.avg)
    return top1.avg, objs.avg


if __name__ == '__main__':
    main()