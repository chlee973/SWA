import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, input):
        # Stack logits from each model: shape (M, B, C)
        logits = torch.stack([model(input) for model in self.models], dim=0)

        # Log softmax per model
        log_probs = F.log_softmax(logits, dim=2)  # shape (M, B, C)

        # Log of average probability
        ensemble_log_prob = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(len(self.models), dtype=logits.dtype, device=logits.device))
        return ensemble_log_prob  # shape (B, C)
    
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, log_probs):
        return log_probs / self.temperature

def tune_temperature(model, valid_loader, device='cuda'):
    model.eval()
    temperature_scaler = TemperatureScaler().to(device)

    log_probs_list = []
    labels_list = []

    # 모든 validation 데이터를 logits, labels로 수집
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            log_probs = model(inputs)
            log_probs_list.append(log_probs)
            labels_list.append(labels)

    log_probs = torch.cat(log_probs_list)
    labels = torch.cat(labels_list)

    # optimizer 설정 (LBFGS가 temperature 하나 튜닝할 때 적절)
    optimizer = torch.optim.LBFGS([temperature_scaler.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled_log_probs = temperature_scaler(log_probs)
        loss = F.cross_entropy(scaled_log_probs, labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    print(f"Optimal temperature: {temperature_scaler.temperature.item():.4f}")
    return temperature_scaler

def predict_with_temperature(model, temperature_scaler, inputs, device='cuda'):
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        logits = model(inputs)
        log_probs = F.log_softmax(logits)
        scaled_logits = temperature_scaler(log_probs)
        probs = F.softmax(scaled_logits, dim=1)
    return probs



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
    temperature_scaler = tune_temperature(model, val_loader)
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            ece_criterion = _ECELoss(15).cuda()

            logit = model(input_var)
            nll = criterion(logit, target_var)
            log_prob = F.log_softmax(logit)
            cnll = criterion(temperature_scaler(log_prob), target_var)

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

    return 

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
