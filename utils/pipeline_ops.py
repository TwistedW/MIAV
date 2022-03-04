import os

import torch
import torch.nn as nn
import torch.optim.optimizer as optim
import torch.optim.lr_scheduler as sche
import numpy as np
from torch.optim import Adam, SGD
from utils.misc import av_preds
from config import arg_config

from utils.misc import construct_print

par = arg_config

def get_total_loss(train_preds_v: torch.Tensor, train_masks_v: torch.Tensor, train_preds_a: torch.Tensor,
                   train_masks_a: torch.Tensor, train_vel: torch.Tensor, train_preds_o: torch.Tensor,
                   fake_A_logit: torch.Tensor, fake_V_logit: torch.Tensor, fake_AV_logit: torch.Tensor,
                   real_topo: torch.Tensor, fake_A_topo:torch.Tensor,
                   fake_V_topo: torch.Tensor, fake_AV_topo:torch.Tensor,
                   loss_funcs: dict) -> (float, list):
    """
    return the sum of the list of loss functions with train_preds and train_masks
    
    Args:
        train_preds (torch.Tensor): predictions
        train_masks (torch.Tensor): masks
        loss_funcs (list): the list of loss functions

    Returns: the sum of all losses and the list of result strings

    """
    loss_list = []
    loss_item_list = []

    # train_preds_a, train_preds_v = av_preds(train_preds_v, train_preds_a)
    assert len(loss_funcs) != 0, "请指定损失函数`loss_funcs`"
    total_vel = train_masks_a + train_masks_v
    preds_vel = (train_preds_a + train_preds_v) * train_vel
    dif_v = (train_vel - train_preds_a) * train_preds_v
    dif_a = (train_vel - train_preds_v) * train_preds_a
    for key, loss in loss_funcs.items():
        if key == "use_aux_loss":
            loss_out = par["aux_weight"] * (loss(train_preds_v, train_masks_v) + loss(train_preds_a, train_masks_a)
                                            + loss(train_preds_o, train_vel))
        elif key == "use_dice_loss":
            loss_out = par["dice_weight"] * (loss(train_preds_v, train_masks_v) + loss(train_preds_a, train_masks_a) +
                                             loss(train_preds_o, train_vel))
        elif key == "use_en_loss":
            loss_out = par["en_weight"] * (loss(train_preds_v, train_masks_v) + loss(train_preds_a, train_masks_a) +
                                           loss(train_preds_o, train_vel))
        elif key == "dif_loss":
            loss_out = par["dif_weight"] * (loss(dif_v, dif_a, train_masks_v, train_masks_a) +
                                            loss(train_preds_v, train_preds_a, train_masks_v, train_masks_a))
        elif key == "base_loss":
            loss_out = par["base_weight"] * (loss(train_preds_v, train_masks_v) + loss(train_preds_a, train_masks_a)
                                             + loss(train_preds_o, train_vel))
        elif key == "topo_loss":
            loss_out = 0
            for i in range(len(real_topo)):
                loss_out += loss(fake_A_topo[i], real_topo[i])
                loss_out += loss(fake_V_topo[i], real_topo[i])
                loss_out += loss(fake_AV_topo[i], real_topo[i])
            loss_out = par["topo_weight"] * loss_out
        elif key == "lsgan_loss":
            G_ad_loss = par["fake_A_weight"] * loss(fake_A_logit, torch.ones_like(fake_A_logit)) + \
                        par["fake_V_weight"] * loss(fake_V_logit, torch.ones_like(fake_V_logit)) + \
                        par["fake_AV_weight"] * loss(fake_AV_logit, torch.ones_like(fake_AV_logit))
            loss_out = par["gan_weight"] * G_ad_loss
        elif key == "hinge_loss":
            G_ad_loss = - par["fake_A_weight"] * fake_A_logit.mean() - \
                          par["fake_V_weight"] * fake_V_logit.mean() - \
                          par["fake_AV_weight"] * fake_A_logit.mean()
            loss_out = par["gan_weight"] * G_ad_loss
        if loss_out.size():
            loss_out = sum(loss_out)
        loss_list.append(loss_out)
        loss_item_list.append(f"{loss_out.item():.5f}")

    train_loss = sum(loss_list)
    return train_loss, loss_item_list


def get_D_total_loss(real_G_logit: torch.Tensor, fake_A_logit: torch.Tensor,
                     fake_V_logit: torch.Tensor, fake_AV_logit: torch.Tensor,
                     loss_funcs: dict) -> (float, list):
    loss_list = []
    loss_item_list = []
    for key, loss in loss_funcs.items():
        if key == "lsgan_loss":
            D_ad_loss = loss_funcs["lsgan_loss"](real_G_logit, torch.ones_like(real_G_logit)) + \
                        par["fake_A_weight"] * loss_funcs["lsgan_loss"](fake_A_logit, torch.zeros_like(fake_A_logit)) + \
                        par["fake_V_weight"] * loss_funcs["lsgan_loss"](fake_V_logit, torch.zeros_like(fake_V_logit)) + \
                        par["fake_AV_weight"] * loss_funcs["lsgan_loss"](fake_AV_logit, torch.zeros_like(fake_AV_logit))

        elif key == "hinge_loss":
            D_ad_loss = torch.nn.ReLU()(1.0 - real_G_logit).mean() + \
                        par["fake_A_weight"] * torch.nn.ReLU()(1.0 + fake_A_logit).mean() + \
                        par["fake_V_weight"] * torch.nn.ReLU()(1.0 + fake_V_logit).mean() + \
                        par["fake_AV_weight"] * torch.nn.ReLU()(1.0 + fake_AV_logit).mean()

    loss_list.append(par["gan_weight"] * D_ad_loss)
    D_loss = sum(loss_list)
    loss_item_list.append(f"{D_ad_loss.item():.5f}")

    return D_loss, loss_item_list


def save_checkpoint(
        model: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: sche._LRScheduler = None,
        amp=None,
        exp_name: str = "",
        current_epoch: int = 1,
        full_net_path: str = "",
        state_net_path: str = "",
):
    """
    保存完整参数模型（大）和状态参数模型（小）

    Args:
        model (nn.Module): model object
        optimizer (optim.Optimizer): optimizer object
        scheduler (sche._LRScheduler): scheduler object
        amp (): apex.amp
        exp_name (str): exp_name
        current_epoch (int): in the epoch, model **will** be trained
        full_net_path (str): the path for saving the full model parameters
        state_net_path (str): the path for saving the state dict.
    """

    state_dict = {
        "arch": exp_name,
        "epoch": current_epoch,
        "net_state": model.state_dict(),
        "opti_state": optimizer.state_dict(),
        "sche_state": scheduler.state_dict(),
        "amp_state": amp.state_dict() if amp else None,
    }
    if torch.cuda.device_count() > 1:
        state_dict["net_state"] = model.module.state_dict()
        torch.save(state_dict, full_net_path)
        torch.save(model.module.state_dict(), state_net_path)
    else:
        torch.save(state_dict, full_net_path)
        torch.save(model.state_dict(), state_net_path)


def save_net_checkpoint(model: nn.Module = None, state_net_path: str = ""):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), state_net_path)
    else:
        torch.save(model.state_dict(), state_net_path)


def resume_checkpoint(
        model: nn.Module = None,
        optimizer: optim.Optimizer = None,
        scheduler: sche._LRScheduler = None,
        amp=None,
        exp_name: str = "",
        load_path: str = "",
        mode: str = "all",
):
    """
    从保存节点恢复模型

    Args:
        model (nn.Module): model object
        optimizer (optim.Optimizer): optimizer object
        scheduler (sche._LRScheduler): scheduler object
        amp (): apex.amp
        exp_name (str): exp_name
        load_path (str): 模型存放路径
        mode (str): 选择哪种模型恢复模式:
            - 'all': 回复完整模型，包括训练中的的参数；
            - 'onlynet': 仅恢复模型权重参数

    Returns mode: 'all' start_epoch; 'onlynet' None
    """
    if os.path.exists(load_path) and os.path.isfile(load_path):
        construct_print(f"Loading checkpoint '{load_path}'")
        checkpoint = torch.load(load_path)
        if mode == "all":
            if exp_name and exp_name != checkpoint["arch"]:
                # 如果给定了exp_name，那么就必须匹配对应的checkpoint["arch"]，否则不作要求
                raise Exception(f"We can not match {exp_name} with {load_path}.")

            start_epoch = checkpoint["epoch"]
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["net_state"])
            else:
                model.load_state_dict(checkpoint["net_state"])
            optimizer.load_state_dict(checkpoint["opti_state"])
            scheduler.load_state_dict(checkpoint["sche_state"])
            if checkpoint.get("amp_state", None):
                if amp:
                    amp.load_state_dict(checkpoint["amp_state"])
                else:
                    construct_print("You are not using amp.")
            else:
                construct_print("The state_dict of amp is None.")
            construct_print(
                f"Loaded '{load_path}' " f"(will train at epoch" f" {checkpoint['epoch']})"
            )
            return start_epoch
        elif mode == "onlynet":
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            construct_print(
                f"Loaded checkpoint '{load_path}' " f"(only has the model's weight params)"
            )
        else:
            raise NotImplementedError
    else:
        raise Exception(f"{load_path}路径不正常，请检查")


def make_scheduler(
        optimizer: optim.Optimizer, total_num: int, scheduler_type: str, scheduler_info: dict
) -> sche._LRScheduler:
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        # curr_epoch start from 0
        # total_num = iter_num if args["sche_usebatch"] else end_epoch
        if scheduler_type == "poly":
            coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "poly_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = pow((1 - float(curr_epoch) / total_num), scheduler_info["lr_decay"])
        elif scheduler_type == "cosine_warmup":
            turning_epoch = scheduler_info["warmup_epoch"]
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1 / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = (1 + np.cos(np.pi * curr_epoch / total_num)) / 2
        elif scheduler_type == "f3_sche":
            coefficient = 1 - abs((curr_epoch + 1) / (total_num + 1) * 2 - 1)
        else:
            raise NotImplementedError
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler


def make_optimizer(model: list, optimizer_type: str, optimizer_info: dict) -> optim.Optimizer:
    if optimizer_type == "sgd_trick":
        # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
        if len(model) == 1:
            params = [
                {
                    "params": [
                        p for name, p in model[0].named_parameters() if ("bias" in name or "bn" in name)
                    ],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p
                        for name, p in model[0].named_parameters()
                        if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
        else:
            params = [
                {
                    "params": [
                        p for name, p in model[0].named_parameters() if ("bias" in name or "bn" in name)
                    ],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p
                        for name, p in model[0].named_parameters()
                        if ("bias" not in name and "bn" not in name)
                    ]
                },
                {
                    "params": [
                        p for name, p in model[1].named_parameters() if ("bias" in name or "bn" in name)
                    ],
                    "weight_decay": 0,
                },
                {
                    "params": [
                        p
                        for name, p in model[1].named_parameters()
                        if ("bias" not in name and "bn" not in name)
                    ]
                },
            ]
        optimizer = SGD(
            params,
            lr=optimizer_info["lr"],
            momentum=optimizer_info["momentum"],
            weight_decay=optimizer_info["weight_decay"],
            nesterov=optimizer_info["nesterov"],
        )
    elif optimizer_type == "sgd_r3":
        if len(model) == 1:
            params = [
                # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
                # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
                # 到减少模型过拟合的效果。
                {
                    "params": [
                        param for name, param in model[0].named_parameters() if name[-4:] == "bias"
                    ],
                    "lr": 2 * optimizer_info["lr"],
                },
                {
                    "params": [
                        param for name, param in model[0].named_parameters() if name[-4:] != "bias"
                    ],
                    "lr": optimizer_info["lr"],
                    "weight_decay": optimizer_info["weight_decay"],
                },
            ]
        else:
            params = [
                {
                    "params": [
                        param for name, param in model[0].named_parameters() if name[-4:] == "bias"
                    ],
                    "lr": 2 * optimizer_info["lr"],
                },
                {
                    "params": [
                        param for name, param in model[0].named_parameters() if name[-4:] != "bias"
                    ],
                    "lr": optimizer_info["lr"],
                    "weight_decay": optimizer_info["weight_decay"],
                },
                {
                    "params": [
                        param for name, param in model[1].named_parameters() if name[-4:] == "bias"
                    ],
                    "lr": 2 * optimizer_info["lr"],
                },
                {
                    "params": [
                        param for name, param in model[1].named_parameters() if name[-4:] != "bias"
                    ],
                    "lr": optimizer_info["lr"],
                    "weight_decay": optimizer_info["weight_decay"],
                },
            ]
        optimizer = SGD(params, momentum=optimizer_info["momentum"])
    elif optimizer_type == "sgd_all":
        if len(model) == 1:
            optimizer = SGD(
                [{'params': model[0].parameters()}],
                lr=optimizer_info["lr"],
                weight_decay=optimizer_info["weight_decay"],
                momentum=optimizer_info["momentum"],
            )
        else:
            optimizer = SGD(
                [{'params': model[0].parameters()},
                 {'params': model[1].parameters()}],
                lr=optimizer_info["lr"],
                weight_decay=optimizer_info["weight_decay"],
                momentum=optimizer_info["momentum"],
            )
    elif optimizer_type == "adam":
        if len(model) == 1:
            optimizer = Adam(
                model[0].parameters(),
                lr=optimizer_info["lr"],
                betas=(0.5, 0.999),
                eps=1e-8,
                weight_decay=optimizer_info["weight_decay"],
            )
        else:
            optimizer = Adam(
                [{'params': model[0].parameters()},
                 {'params': model[1].parameters()}],
                lr=optimizer_info["lr"],
                betas=(0.5, 0.999),
                eps=1e-8,
                weight_decay=optimizer_info["weight_decay"],
            )
    # elif optimizer_type == "f3_trick":
    #     backbone, head = [], []
    #     for name, params_tensor in model.named_parameters():
    #         if name.startswith("div_2"):
    #             pass
    #         elif name.startswith("div"):
    #             backbone.append(params_tensor)
    #         else:
    #             head.append(params_tensor)
    #     params = [
    #         {"params": backbone, "lr": 0.1 * optimizer_info["lr"]},
    #         {"params": head, "lr": optimizer_info["lr"]},
    #     ]
    #     optimizer = SGD(
    #         params=params,
    #         momentum=optimizer_info["momentum"],
    #         weight_decay=optimizer_info["weight_decay"],
    #         nesterov=optimizer_info["nesterov"],
    #     )
    else:
        raise NotImplementedError

    print("optimizer = ", optimizer)
    return optimizer


if __name__ == "__main__":
    a = torch.rand((3, 3)).bool()
    print(isinstance(a, torch.FloatTensor), a.type())
