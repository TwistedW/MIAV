import os
import random
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

import numpy as np
import torch
from torchvision import transforms
import cv2

np.set_printoptions(threshold=np.inf)


def misc_measures_evaluation(true_vessels, pred_vessels_bin, img_v, img_a, mask_v, mask_a, box):
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    true_vessels = np.array(torch.squeeze(true_vessels.int()))
    pro = np.array(torch.squeeze((transform_tensor(pred_vessels_bin))))
    pred_vessels_bin = np.array(torch.squeeze((transform_tensor(pred_vessels_bin)).ge(0.5).int()))
    box = np.array(torch.squeeze(box.int()))
    box = np.resize(box, true_vessels.shape)
    true_vessels = true_vessels[box == 1]
    pred_vessels_bin = pred_vessels_bin[box == 1]
    pro = pro[box == 1]

    pred_a = np.array(torch.squeeze((transform_tensor(img_a)).ge(0.9).int()))
    pred_v = np.array(torch.squeeze((transform_tensor(img_v)).ge(0.9).int()))
    pred_av = np.zeros_like(pred_a)
    pred_av[pred_a == 1] = 2
    pred_av[pred_v == 1] = 3

    mask_a = np.array(torch.squeeze(mask_a.int()))
    mask_v = np.array(torch.squeeze(mask_v.int()))
    mask_av = np.zeros_like(mask_a)
    mask_av[mask_a == 1] = 2
    mask_av[mask_v == 1] = 3

    mask_av = mask_av[box == 1]
    pred_av = pred_av[box == 1]

    TP_AV = ((pred_av == 3) & (mask_av == 3)).sum()
    TN_AV = ((pred_av == 2) & (mask_av == 2)).sum()
    FN_AV = ((pred_av == 2) & (mask_av == 3)).sum()
    FP_AV = ((pred_av == 3) & (mask_av == 2)).sum()
    sensitivity_av = TP_AV / (TP_AV + FN_AV)
    specificity_av = TN_AV / (TN_AV + FP_AV)
    precision_av = TP_AV / (TP_AV + FP_AV)
    balanced_accuracy_av = (sensitivity_av + specificity_av) / 2
    f1_av = _f1_score(precision_av, sensitivity_av)

    # return sensitivity_av, specificity_av, balanced_accuracy_av

    TP = ((pred_vessels_bin == 1) & (true_vessels == 1)).sum()
    TN = ((pred_vessels_bin == 0) & (true_vessels == 0)).sum()
    FN = ((pred_vessels_bin == 0) & (true_vessels == 1)).sum()
    FP = ((pred_vessels_bin == 1) & (true_vessels == 0)).sum()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = _f1_score(precision, sensitivity)
    auc_value = roc_auc_score(true_vessels.flatten(), pro.flatten())

    acc_av = (TP_AV + TN_AV) / (TP_AV + FN_AV + TN_AV + FP_AV)

    # acc_av = (acc_a + acc_v) / 2
    return acc, sensitivity, specificity, f1, auc_value, sensitivity_av, specificity_av, f1_av, balanced_accuracy_av, acc_av


def misc_measures_evaluation_box(true_vessels, pred_vessels_bin, img_v, img_a, mask_v, mask_a, box):
    transform_tensor = transforms.Compose([transforms.ToTensor()])
    true_vessels = np.array(torch.squeeze(true_vessels.int()))
    pro = np.array(torch.squeeze((transform_tensor(pred_vessels_bin))))
    pred_vessels_bin = np.array(torch.squeeze((transform_tensor(pred_vessels_bin)).ge(0.5).int()))
    box = np.array(torch.squeeze(box.int()))

    pred_a = np.array(torch.squeeze((transform_tensor(img_a)).ge(0.6).int()))
    pred_v = np.array(torch.squeeze((transform_tensor(img_v)).ge(0.6).int()))
    pred_av = np.zeros_like(pred_a)
    pred_av[pred_a == 1] = 2
    pred_av[pred_v == 1] = 3

    mask_a = np.array(torch.squeeze(mask_a.int()))
    mask_v = np.array(torch.squeeze(mask_v.int()))
    mask_av = np.zeros_like(mask_a)
    mask_av[mask_a == 1] = 2
    mask_av[mask_v == 1] = 3

    TP_AV = ((pred_av == 3) & (mask_av == 3)).sum()
    TN_AV = ((pred_av == 2) & (mask_av == 2)).sum()
    FN_AV = ((pred_av == 2) & (mask_av == 3)).sum()
    FP_AV = ((pred_av == 3) & (mask_av == 2)).sum()
    sensitivity_av = TP_AV / (TP_AV + FN_AV)
    specificity_av = TN_AV / (TN_AV + FP_AV)
    balanced_accuracy_av = (sensitivity_av + specificity_av) / 2

    # return sensitivity_av, specificity_av, balanced_accuracy_av

    TP = ((pred_vessels_bin == 1) & (true_vessels == 1)).sum()
    TN = ((pred_vessels_bin == 0) & (true_vessels == 0)).sum()
    FN = ((pred_vessels_bin == 0) & (true_vessels == 1)).sum()
    FP = ((pred_vessels_bin == 1) & (true_vessels == 0)).sum()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = _f1_score(precision, sensitivity)
    auc_value = roc_auc_score(true_vessels.flatten(), pro.flatten())

    # v misc
    TP_v = ((pred_v == 1) & (mask_v == 1)).sum()
    TN_v = ((pred_v == 0) & (mask_v == 0)).sum()
    FN_v = ((pred_v == 0) & (mask_v == 1)).sum()
    FP_v = ((pred_v == 1) & (mask_v == 0)).sum()
    acc_v = (TP_v + TN_v) / (TP_v + TN_v + FP_v + FN_v)

    # a misc
    TP_a = ((pred_a == 1) & (mask_a == 1)).sum()
    TN_a = ((pred_a == 0) & (mask_a == 0)).sum()
    FN_a = ((pred_a == 0) & (mask_a == 1)).sum()
    FP_a = ((pred_a == 1) & (mask_a == 0)).sum()
    acc_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a)

    acc_av = (acc_a + acc_v) / 2
    return acc, sensitivity, specificity, f1, auc_value, sensitivity_av, specificity_av, balanced_accuracy_av, acc_av

def _f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


class AvgMeter(object):
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


def set_seed(seed, use_cudnn_benchmark):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    if use_cudnn_benchmark:
        construct_print("We will use `torch.backends.cudnn.benchmark`")
    else:
        construct_print("We will not use `torch.backends.cudnn.benchmark`")
    torch.backends.cudnn.benchmark = use_cudnn_benchmark
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)


def pre_mkdir(path_config: dict):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    write_data_to_file(f"=== te_log {datetime.now()} ===", path_config["te_log"])
    write_data_to_file(f"=== tr_log {datetime.now()} ===", path_config["tr_log"])

    # 提前创建好存储预测结果和存放模型以及tensorboard的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])
    check_mkdir(path_config["tb"])


def check_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def write_data_to_file(data_str, file_path):
    with open(file_path, encoding="utf-8", mode="a") as f:
        f.write(data_str + "\n")


def construct_print(out_str: str, total_length: int = 80):
    if len(out_str) >= total_length:
        extended_str = "=="
    else:
        extended_str = "=" * ((total_length - len(out_str)) // 2 - 4)
    out_str = f" {extended_str}>> {out_str} <<{extended_str} "
    print(out_str)


def construct_path(proj_root: str, exp_name: str, xlsx_name: str) -> dict:
    ckpt_path = os.path.join(proj_root, "output")

    pth_log_path = os.path.join(ckpt_path, exp_name)
    tb_path = os.path.join(pth_log_path, "tb")
    save_path = os.path.join(pth_log_path, "pre")
    pth_path = os.path.join(pth_log_path, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth.tar")
    final_state_path = os.path.join(pth_path, "state_final.pth")
    final_Dis_model_path = os.path.join(pth_path, "checkpoint_Dis_final.pth.tar")
    final_Dis_state_path = os.path.join(pth_path, "state_Dis_final.pth")
    final_vgg_state_path = os.path.join(pth_path, "state_vgg_final.pth")

    tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
    te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
    cfg_log_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())[:10]}.txt")
    trainer_log_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())[:10]}.txt")
    xlsx_path = os.path.join(ckpt_path, xlsx_name)

    path_config = {
        "ckpt_path": ckpt_path,
        "pth_log": pth_log_path,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "final_Dis_net": final_Dis_model_path,
        "final_Dis_state_net": final_Dis_state_path,
        "final_vgg_state_net": final_vgg_state_path,
        "tr_log": tr_log_path,
        "te_log": te_log_path,
        "cfg_log": cfg_log_path,
        "trainer_log": trainer_log_path,
        "xlsx": xlsx_path,
    }
    return path_config


def construct_exp_name(arg_dict: dict):
    # If you know the function of these two lines, you can uncomment out them.
    # if arg_dict.get("special_name", None):
    #     return arg_dict["special_name"].replace("@", "_")

    # You can modify and supplement it according to your needs.
    focus_item = OrderedDict(
        {
            "input_size": "s",
            "batch_size": "bs",
            "epoch_num": "e",
            "info": "info-",
        }
    )
    exp_name = f"{arg_dict['model']}"
    for k, v in focus_item.items():
        item = arg_dict[k]
        if isinstance(item, bool):
            item = "Y" if item else "N"
        elif isinstance(item, (list, tuple)):
            item = "Y" if item else "N"  # 只是判断是否飞空
        elif isinstance(item, str):
            if not item:
                continue
            if "_" in item:
                item = item.replace("_", "")
        elif item is None:
            item = "N"

        if isinstance(item, str):
            item = item.lower()
        exp_name += f"_{v.upper()}{item}"
    return exp_name


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_green(x):
    x = cv2.cvtColor(x*255, cv2.COLOR_GRAY2BGR)
    x[:, :, 0] = 0
    x[:, :, 1][x[:, :, 1] >= 165] = 255
    x[:, :, 1][x[:, :, 1] < 165] = 0
    x[:, :, 2] = 0
    return x


def draw_red(x):
    x = cv2.cvtColor(x * 255, cv2.COLOR_GRAY2BGR)
    x[:, :, 0] = 0
    x[:, :, 1] = 0
    x[:, :, 2][x[:, :, 2] < 165] = 0
    x[:, :, 2][x[:, :, 2] >= 165] = 255
    return x


def av_preds(vein, artery):
    # artery[artery < vein] = 0
    # vein[vein < artery] = 0

    return np.array(artery), np.array(vein)


def draw_pre_red(x, patch):
    if patch:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) / 255
    else:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    x[:, :, 0] = 0
    x[:, :, 1] = 0
    x[:, :, 2][x[:, :, 2] <= 165] = 0
    x[:, :, 2][x[:, :, 2] > 165] = 255
    # x[:, :, 2][x[:, :, 2] <= 230] = 0
    # x[:, :, 2][x[:, :, 2] > 230] = 255
    return x


def draw_pre_blue(x, patch):
    if patch:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) / 255
    else:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    x[:, :, 0][x[:, :, 0] > 165] = 255
    x[:, :, 0][x[:, :, 0] <= 165] = 0
    # x[:, :, 0][x[:, :, 0] > 230] = 255
    # x[:, :, 0][x[:, :, 0] <= 230] = 0
    x[:, :, 1] = 0
    x[:, :, 2] = 0
    return x


def draw_pre_av(out_img_o, out_img_v, out_img_a, patch):
    if patch:
        o = cv2.cvtColor(out_img_o, cv2.COLOR_GRAY2BGR) / 255
        v = out_img_v
        a = out_img_a
    else:
        o = cv2.cvtColor(out_img_o, cv2.COLOR_GRAY2BGR)
        v = out_img_v
        a = out_img_a
    o[:, :, :][o[:, :, :] > 128] = 255
    o[:, :, :][o[:, :, :] <= 128] = 0
    o[:, :, 1][v > a] = 0
    o[:, :, 2][v > a] = 0
    o[:, :, 0][v < a] = 0
    o[:, :, 1][v < a] = 0
    o = o[:, :, ::-1]
    return o


def draw_vessel(x):
    x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) / 255
    # print(x[100])
    x[:, :, :][x[:, :, :] > 128] = 255
    x[:, :, :][x[:, :, :] <= 128] = 0
    return x


if __name__ == "__main__":
    print("=" * 8)
    out_str = "lartpang"
    construct_print(out_str, total_length=8)
