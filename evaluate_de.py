import argparse
import os
import shutil
import time
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import util


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='checkpoints/de', type=str)
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')

def main():
    global args, best_acc
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        print(f"{args.save_dir} does not exists")
        return
    models = []
    for i in range(10):
        model = resnet.__dict__[args.arch]()
        model.cuda()
        filename = f"model_0{i}"
        model.load_state_dict(torch.load(f'{args.save_dir}/{filename}.th')['state_dict'])
        model.eval()
        models.append(model)
    

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
    
    # temperature scaling
    de_models = [util.Ensemble(models[:i+1]).cuda() for i in range(10)]
    print("------ DE --------")
    for de_model in de_models:
        validate(val_loader, test_loader, de_model, criterion)
    print("------ Individual models --------")
    for model in models:
        validate(val_loader, test_loader, model, criterion)



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

if __name__ == '__main__':
    main()