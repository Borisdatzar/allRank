import os
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger
from allrank.utils.tensorboard_utils import TensorboardSummaryWriter

logger = get_logger()


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, scaler, opt=None):
    mask = (yb == PADDED_Y_VALUE)

    with autocast():  # Enable mixed precision for the forward pass
        loss = loss_func(model(xb, mask, indices), yb)

    if opt is not None:
        scaler.scale(loss).backward()  # Scale the loss and backpropagate

        if gradient_clipping_norm:
            scaler.unscale_(opt)  # Unscale the gradients before clipping
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)

        scaler.step(opt)  # Update the weights using the scaled gradients
        scaler.update()   # Update the scale for the next iteration
        opt.zero_grad()

    return loss.item(), len(xb)



def metric_on_batch(metric, model, xb, yb, indices):
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)


def metric_on_epoch(metric, model, dl, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
             for xb, yb, indices in dl]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, dl, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))

    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    return summary

def test_summary(test_metrics):
    summary = 'Test summary\n'

    for metric_name, metric_value in test_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def eval_model(model, test_dl, config, device):

    model.eval()
    with torch.no_grad():
        test_metrics = compute_metrics(config.metrics, model, test_dl, device)

    logger.info(test_summary(test_metrics))

    return {
        "test_metrics": test_metrics,
    }
