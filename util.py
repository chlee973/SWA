import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F

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

    def forward(self, logits):
        return logits / self.temperature

def tune_temperature(model, valid_loader, device='cuda'):
    model.eval()
    temperature_scaler = TemperatureScaler().to(device)

    logits_list = []
    labels_list = []

    # 모든 validation 데이터를 logits, labels로 수집
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # optimizer 설정 (LBFGS가 temperature 하나 튜닝할 때 적절)
    optimizer = torch.optim.LBFGS([temperature_scaler.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled_logits = temperature_scaler(logits)
        loss = F.cross_entropy(scaled_logits, labels)
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
        scaled_logits = temperature_scaler(logits)
        probs = F.softmax(scaled_logits, dim=1)
    return probs