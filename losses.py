import torch
import torch.nn as nn
import numpy as np


from torchmetrics import Metric

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()


class EWCLoss(nn.Module):
    def __init__(self, frozen_model, fisher, lambda_ewc=1e4):
        super(EWCLoss, self).__init__()
        self.frozen_model = frozen_model
        self.fisher = fisher
        self.lambda_ewc = lambda_ewc

    def forward(self, cur_model):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(cur_model.named_parameters(), self.frozen_model.named_parameters()):
            if 'fc' in name:
                continue
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.lambda_ewc * loss_reg

def pred2labels(pred, threshold):

    labels = []
    # Calculate the conditional probability
    for p in pred:
        if p >= threshold:
            labels.append("NORMAL")
        else:
            labels.append("OTHER")

    return np.array(labels)


# Assume the names are passed in
def gt2labels(ygt):
    labels = []
    # Change based on best guess
    for y in ygt:
        if torch.round(y) == 1:
            # Loop through items in the label dictionary find the correct example
            labels.append("NORMAL")
        elif torch.round(y) == 0:
            labels.append("OTHER")

    # Return converted to numpy array for calculations
    return np.array(labels)

# Custom accuracy metric
class ClassAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, threshold=0.5):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        assert preds.shape == targets.shape

        # change preds/targets into class labels?
        preds_labels = pred2labels(preds, threshold=self.threshold)
        gt_labels = gt2labels(targets)

        self.correct += np.sum(preds_labels == gt_labels, axis=0)
        self.total += gt_labels.shape[0]
        # print("Correct: {}".format(self.correct))
        # print("Total: {}".format(self.total))

    def compute(self):
        return self.correct.float() / self.total