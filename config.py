import os

__all__ = ["proj_root", "arg_config"]

from collections import OrderedDict

proj_root = os.path.dirname(__file__)
datasets_root = "E:\Python_code\datasets"

Data_TR_path = os.path.join(datasets_root, "Eye_AV", "Data_TR/test")
DRIVE_AV_train = os.path.join(datasets_root, "Eye_AV", "DRIVE_AV/training")
DRIVE_AV_path = os.path.join(datasets_root, "Eye_AV", "DRIVE_AV/test")
HRF_AV_path = os.path.join(datasets_root, "Eye_AV", "HRF_AV/test")
KAILUAN_AV_path = os.path.join(datasets_root, "Eye_AV", "KAILUAN_AV/test")
LES_AV_path = os.path.join(datasets_root, "Eye_AV", "LES_AV/test")

arg_config = {
    "model": "PCNet_ISE",  # 实际使用的模型，需要在`network/__init__.py`中导入
    "Discriminator": "Discriminator",
    "info": "chan32-drive",  # 关于本次实验的额外信息说明，这个会附加到本次试验的exp_name的结尾，如果为空，则不会附加内容。
    "use_amp": False,  # 是否使用amp加速训练
    "resume_mode": "test",  # the mode for resume parameters: ['train', 'test', '']
    "save_pre": True,  # 是否保留最终的预测结果
    "epoch_num": 8000,  # 训练周期, 0: directly test model
    "lr": 0.0002,  # 微调时缩小100倍
    "channel": 32,
    "xlsx_name": "",  # the name of the record file
    # 数据集设置
    "rgb_data": {
        "tr_data_path": DRIVE_AV_train,
        "te_data_list": OrderedDict(
            {
                # "Data_TR": Data_TR_path,
                "DRIVE_AV": DRIVE_AV_path,
                # "HRF_AV": HRF_AV_path,
                # "KAILUAN_AV": KAILUAN_AV_path,
                # "LES_AV": LES_AV_path,
            },
        ),
    },
    # 训练过程中的监控信息
    "tb_update": 10,  # >0 则使用tensorboard
    "print_freq": 10,  # >0, 保存迭代过程中的信息
    "save_vis_freq": 100,  # >0，保存可视化结果
    "save_middle_res": 100,  # 保存和测试中间结果
    # img_prefix, gt_prefix，用在使用索引文件的时候的对应的扩展名
    "prefix": (".jpg", ".png"),
    "size_list": None,  # 不使用多尺度训练
    "reduction": "mean",  # 损失处理的方式，可选“mean”和“sum”
    # 优化器与学习率衰减
    "optim": "adam",  # 自定义部分的学习率
    "weight_decay": 5e-4,  # 微调时设置为0.0001
    "momentum": 0.9,
    "patch": True,
    "nesterov": False,
    "sche_usebatch": False,
    "lr_type": "poly",
    "warmup_epoch": 1,
    # depond on the special lr_type, only lr_type has 'warmup', when set it to 1, it means no warmup.
    "lr_decay": 0.9,  # poly
    "use_bigt": True,  # 训练时是否对真值二值化（阈值为0.5）
    "batch_size": 8,  # 要是继续训练, 最好使用相同的batchsize
    "num_workers": 0,  # 不要太大, 不然运行多个程序同时训练的时候, 会造成数据读入速度受影响
    "input_size": 256,

    # 损失函数控制
    "base_loss": True,
    "use_aux_loss": True,  # 是否使用辅助损失
    "use_en_loss": False,
    "use_dice_loss": False,
    "dif_loss": True,
    "topo_loss": True,
    # GAN loss - [lsgan, hinge] you only choose one to True
    "lsgan_loss": True,
    "hinge_loss": False,

    # Noisy Labels控制
    "AFM": True,

    # Fca控制
    "Fca": False,

    # Dropout控制
    "Dropout": True,

    # 超参数设置
    "aux_weight": 0.8,  # 利用辅助的损失，及CEL损失，可在loss下的CEL中查看
    "dice_weight": 0.01,  # 鲁棒的Dice
    "en_loss": 0.1,  # 连通性loss
    "dif_weight": 0.1,  # bce loss应用在差分中
    "base_weight": 1,  # 最基础的损失，我们采用的是BCE损失
    "topo_weight": 0.5,
    "gan_weight": 1,  # 控制GAN损失的参数

    # 控制动静脉判别器的权重分配，和为1
    "fake_A_weight": 0.2,
    "fake_V_weight": 0.2,
    "fake_AV_weight": 0.6,
}
