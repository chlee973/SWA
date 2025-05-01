import argparse
import os
import shutil
import time
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import util


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--exploring_epochs', default=150, type=int, metavar='N',
                    help='number of total exploring epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--weight_avg_period', default=5, type=int, metavar='N',
                    help='compute running average of weights for every "weight_avg_period" iterations')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    total_len = len(train_dataset)
    train_len = int(0.9 * total_len)
    valid_len = total_len - train_len

    new_train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_len, valid_len],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(
        new_train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        acc = validate(val_loader, test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        best_acc = max(acc, best_acc)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
        }, filename=os.path.join(args.save_dir, 'model.th'))
    

    swa_optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=0,
                                    weight_decay=args.weight_decay)
    
    swa_model = copy.deepcopy(model)

    for epoch in range(args.exploring_epochs):
        train_swa(train_loader, model, swa_model, criterion, optimizer, epoch)
        update_batchnorm(swa_model, train_loader)
        acc = validate(val_loader, test_loader, model, criterion)
        best_acc= max(acc, best_acc)
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': swa_model.state_dict(),
                'best_acc': best_acc,
            }, filename=os.path.join(args.save_dir, f'swa_model_{epoch+1}.th'))

        save_checkpoint({
            'state_dict': swa_model.state_dict(),
            'best_acc': best_acc,
        }, filename=os.path.join(args.save_dir, 'swa_model.th'))


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        accs.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'ACC {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, acc=accs))
            
def train_swa(train_loader, model, swa_model, criterion, optimizer, epoch):
    """
        Run one train epoch with SWA
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        acc = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        accs.update(acc.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'ACC {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, acc=accs))
            
        start_iter = epoch * len(train_loader) + 1
        n_models = start_iter // args.weight_avg_period + 1
        if (start_iter + i)%5==0:
            for swa_param, param in zip(swa_model.parameters(), model.parameters()):
                swa_param.data *= n_models / (n_models + 1)
                swa_param.data += param.data / (n_models + 1)



def validate(val_loader, test_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    nlls = AverageMeter()
    cnlls = AverageMeter()
    accs = AverageMeter()
    eces = AverageMeter()
    # switch to evaluate mode
    end = time.time()
    model.eval()
    temperature_scaler = util.tune_temperature(model, val_loader)
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            ece_criterion = util._ECELoss(15).cuda()

            logit = model(input_var)
            nll = criterion(logit, target_var)
            cnll = criterion(temperature_scaler(logit), target_var)

            ece = ece_criterion(logit, target_var)
            logit = logit.float()
            nll = nll.float()
            cnll = cnll.float()
            ece = ece.float()
            # measure accuracy and record loss
            acc = accuracy(logit.data, target)[0]
            nlls.update(nll.item(), input.size(0))
            cnlls.update(cnll.item(), input.size(0))
            accs.update(acc.item(), input.size(0))
            eces.update(ece.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * ACC {acc.avg:.3f}\t'
          'NLL {nll.avg:.4f}\t'
          'cNLL {cnll.avg:.4f}\t'
          'ECE {ece.avg:.3f}'
          .format(acc=accs, nll=nlls, cnll=cnlls, ece=eces))

    return accs.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output: [batch_size, num_class]
    # target: [batch_size]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # pred: [batch_size, maxk]
    pred = pred.t()
    # pred: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # target.view(1, -1).expand_as(pred): [maxk(target이 여럿 복사된 부분), batch_size]
    # correct: [maxk, batch_size]
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def update_batchnorm(swa_model, train_loader):
    swa_model.train()
    for i, (input, _) in enumerate(train_loader):
        _ = swa_model(input)

if __name__ == '__main__':
    main()