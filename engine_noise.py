import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import torch.nn as nn

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

import utils as myutils

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torchattacks
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from autoregressive_poisoning.create_ar_poisons_3channel import *
from utils import *

def sample_configs(choices, is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_visual_prompt_tuning == False and is_adapter == False and is_LoRA == False and is_prefix==False:
        visual_prompt_depth = random.choice(choices['visual_prompt_depth'])
        lora_depth = random.choice(choices['lora_depth'])
        adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        config['visual_prompt_dim'] = [random.choice(choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth)
        config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        config['adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth)
        config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)
    else:
        if is_visual_prompt_tuning:
            config['visual_prompt_dim'] = [choices['super_prompt_tuning_dim']] * (depth)
        else:
            config['visual_prompt_dim'] = [0] * (depth)
        
        if is_adapter:
             config['adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    alpha = 0.8

    y_out = []
    y_pred = []
    y_true = []
    my_ar_process = create_ar_processes('svhn')

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if delta is not None:
            if delta.size(0) != samples.size(0):
                cur_delta = delta.repeat(samples.size(0), 1, 1, 1)
            mask = torch.zeros_like(samples).cuda()
            mask[..., 0:16, 0:16] = 1
            patch = cur_delta * mask
            samples = samples + patch
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            if args.patch_fool:
                outputs, attn = model(samples)
            else:
                outputs = model(samples)

            # If, images are normalized:
            # adv_images = atk(samples, targets)
            # advinputs = perturb_with_ar_process(my_ar_process, samples, targets, 0, (228, 228), 4, eps=8)
            # outputs = model(advinputs)

            if teacher_model:
                with torch.no_grad():
                    if args.patch_fool:
                        if delta is not None:
                            # if args.patch_fool:
                            #     mask = get_attn_mask(teacher_model, samples, targets, args)
                            # else:
                            #     mask = torch.zeros_like(samples).cuda()
                            mask[..., 0:16, 0:16] = 1
                            patch = delta * mask

                        teach_output, t_attn = teacher_model(samples + patch)
                    else:
                        teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 0.8 * criterion(outputs, targets) + 0.2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)
        
        if args.teacher_model:
            attn_loss = 0.0
            for i in range(len(attn)):
                # attn_loss += kl_loss(attn[i], t_attn[i])
                attn_loss += torch.cdist(attn[i], t_attn[i], p=2).mean()
            metric_logger.update(attn_loss=attn_loss)
            
            loss_value = loss.item() + alpha * attn_loss
            
            loss += alpha * attn_loss
        else:
            loss_value = loss.item()

        y_out.extend(outputs.detach().cpu().numpy())
        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) # Save Prediction
        labels = targets.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

def train_one_epoch_delta(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None, opt=None, scheduler=None):
    model.eval()
    # set random seed
    random.seed(epoch)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10000
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        adv_images, delta = myutils.patch_fool_fixed(model, samples, targets, delta, opt, scheduler, args)
    print("Finish Training Delta One epoch")
    return delta

def train_one_epoch_patchfool(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)
    
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    alpha = 0.8
    
    # atk = torchattacks.PGD(model, eps=2/255, alpha=2/255, steps=4)
    atk = torchattacks.RFGSM(model, eps=8/255, alpha=10/255)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    y_out = []
    y_pred = []
    y_true = []
    my_ar_process = create_ar_processes('svhn')

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if not args.use_backbone:
            if mode == 'super':
                # sample
                config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
                # print("current iter config: {}".format(config))
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            elif mode == 'retrain':
                config = retrain_config
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)
            
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        # Generate Patch fool images
        adv_images = myutils.patch_fool(model, samples, targets, args)
        # model.module.get_attn(False)
        # adv_images = atk(samples, targets)
        # model.module.get_attn(True)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(adv_images)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(adv_images)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(adv_images)
                    loss = criterion(outputs, targets)
        else:
            if args.patch_fool:
                outputs, attn = model(adv_images)
            else:
                outputs = model(adv_images)

            if teacher_model:
                with torch.no_grad():
                    if args.patch_fool:
                        teach_output, t_attn = teacher_model(adv_images)
                    else:
                        teach_output = teacher_model(adv_images)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        if args.teacher_model:
            attn_loss = 0.0
            for i in range(len(attn)):
                # attn_loss += kl_loss(attn[i], t_attn[i])
                attn_loss += torch.cdist(attn[i], t_attn[i], p=2).mean()
            metric_logger.update(attn_loss=attn_loss)
            
            loss_value = loss.item() + alpha * attn_loss
            
            loss += alpha * attn_loss
        else:
            loss_value = loss.item()

        y_out.extend(outputs.detach().cpu().numpy())
        outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
        y_pred.extend(outputs) # Save Prediction
        labels = targets.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged noise stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    y_out = []
    y_pred = []
    y_true = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            if args.patch_fool:
                output, attn = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        y_out.extend(output.cpu().numpy())
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = target.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

@torch.no_grad()
def evaluate_delta(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    y_out = []
    y_pred = []
    y_true = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
      
        if delta.size(0) != images.size(0):
            cur_delta = delta.repeat(images.size(0), 1, 1, 1)

        mask = torch.zeros_like(images).cuda()
        mask[..., 0:16, 0:16] = 1
        patch = cur_delta * mask
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images + patch)
                loss = criterion(output, target)
        else:
            if args.patch_fool:
                output, attn = model(images + patch)
            else:
                output = model(images + patch)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        y_out.extend(output.cpu().numpy())
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = target.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Noise_Acc@1 {top1.global_avg:.3f} Noise_Acc@5 {top5.global_avg:.3f} Noise_ loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    torch.cuda.empty_cache()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred, y_true, y_out

def evaluate_patchfool(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False, args=None, delta=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))
    
    y_out = []
    y_pred = []
    y_true = []
    # atk = torchattacks.PGD(model, eps=2/255, alpha=2/255, steps=4)
    atk = torchattacks.RFGSM(model, eps=8/255, alpha=10/255)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # Generate patchfool test sample
        images = myutils.patch_fool(model, images, target, args)
        
        # model.module.get_attn(False)
        # adv_images = atk(images, target)
        # model.module.get_attn(True)
        
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            if args.patch_fool:
                output, attn = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        # y_out.extend(output.cpu().detach().numpy())
        # output = (torch.max(torch.exp(output), 1)[1]).data.detach().cpu().numpy()
        # y_pred.extend(output) # Save Prediction
        # labels = target.data.cpu().detach().numpy()
        # y_true.extend(labels) # Save Truth

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, None, None, None