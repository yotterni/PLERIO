import torch
import torch.nn as nn
import torchmetrics

import torch
from torchmetrics.metric import Metric


class BalancedAccuracy(Metric):
    """
    Balanced Accuracy for binary classification.
    Definition:
        balanced_accuracy = (Recall_0 + Recall_1) / 2
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

        # State: true positives, true negatives, positives, negatives
        self.add_state("tp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds  – probabilities or logits (float), shape [N] or [N,1]
        target – ground truth labels {0,1}, shape [N] or [N,1]
        """

        # Flatten
        preds = preds.view(-1)
        target = target.view(-1).long()

        # Threshold probabilities/logits
        preds_bin = (preds >= self.threshold).long()

        # Compute confusion matrix pieces
        self.tp += torch.sum((preds_bin == 1) & (target == 1)).float()
        self.tn += torch.sum((preds_bin == 0) & (target == 0)).float()
        self.fp += torch.sum((preds_bin == 1) & (target == 0)).float()
        self.fn += torch.sum((preds_bin == 0) & (target == 1)).float()

    def compute(self):
        """
        Returns balanced accuracy = (TPR + TNR) / 2
        """

        # recall for class 1 (sensitivity)
        tpr = self.tp / (self.tp + self.fn + 1e-12)

        # recall for class 0 (specificity)
        tnr = self.tn / (self.tn + self.fp + 1e-12)

        return (tpr + tnr) / 2


class BinaryMetrics(nn.Module):
    def __init__(self, from_logits=True, threshold=0.5):
        super().__init__()
        self.from_logits = from_logits

        # Prob-based metrics
        self.auroc = torchmetrics.AUROC(task="binary")
        self.ap = torchmetrics.AveragePrecision(task="binary")

        # Threshold-based metrics
        self.acc = torchmetrics.Accuracy(task="binary", threshold=threshold)
        self.bal_acc = BalancedAccuracy(threshold=threshold)
        self.f1 = torchmetrics.F1Score(task="binary", threshold=threshold)
        self.prec = torchmetrics.Precision(task="binary", threshold=threshold)
        self.rec = torchmetrics.Recall(task="binary", threshold=threshold)

        self.metrics = {
            "auroc": self.auroc,
            "ap": self.ap,
            "acc": self.acc,
            "bal_acc": self.bal_acc,
            "f1": self.f1,
            "precision": self.prec,
            "recall": self.rec,
        }

    def _prepare(self, preds, targets):
        """
        Converts everything into a single, TorchMetrics-safe format:

        preds:   float, shape [B, 1]
        targets: int,   shape [B, 1]
        """

        # convert logits -> sigmoid probabilities
        if self.from_logits:
            preds = torch.sigmoid(preds.float())
        else:
            preds = preds.float()

        # preds must be [B, 1]
        preds = preds.view(-1, 1)

        # targets must be [B, 1] for all binary metrics
        targets = targets.view(-1, 1).long()

        return preds, targets

    def update(self, preds, targets):
        preds, targets = self._prepare(preds, targets)

        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
