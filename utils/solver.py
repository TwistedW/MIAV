import os
from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import network as network_lib
from pathlib import Path
from torchvision.transforms import Normalize
import cv2
import torch.nn as nn
from utils.dataloader import create_loader, create_boxes
from utils.metric import cal_maxf, cal_pr_mae_meanf, ACC
from utils.image_fusion import imgFusion, Stitcher
from utils.misc import (
    AvgMeter, misc_measures_evaluation,
    construct_print,
    write_data_to_file, check_mkdir, draw_vessel, draw_pre_av, misc_measures_evaluation_box,
    draw_probmap, draw_green, draw_red, draw_pre_blue, draw_pre_red, av_preds
)
from utils.pipeline_ops import (
    get_total_loss,
    get_D_total_loss,
    make_optimizer,
    make_scheduler,
    resume_checkpoint,
    save_checkpoint,
    save_net_checkpoint,
)
from utils.recorder import TBRecorder, Timer, XLSXRecoder


class MAV(nn.Module):
    def __init__(self, arg_dict: dict, path_dict: dict):
        super(MAV, self).__init__()
        self.arg_dict = arg_dict
        self.path_dict = path_dict
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if hasattr(network_lib, self.arg_dict["model"]):
            if torch.cuda.device_count() > 1:
                self.net = getattr(network_lib, self.arg_dict["model"])()
                self.net = torch.nn.DataParallel(self.net).to(self.dev)
                self.DisNet = getattr(network_lib, self.arg_dict["Discriminator"])()
                self.DisNet = torch.nn.DataParallel(self.DisNet).to(self.dev)
            else:
                self.net = getattr(network_lib, self.arg_dict["model"])().to(self.dev)
                self.DisNet = getattr(network_lib, self.arg_dict["Discriminator"])(image_size=self.arg_dict["input_size"]).to(self.dev)
        else:
            raise AttributeError

        self.opti = make_optimizer(
            model=[self.net],
            optimizer_type=self.arg_dict["optim"],
            optimizer_info=dict(
                lr=self.arg_dict["lr"],
                momentum=self.arg_dict["momentum"],
                weight_decay=self.arg_dict["weight_decay"],
                nesterov=self.arg_dict["nesterov"],
            ),
        )

        self.opti_D = make_optimizer(
            model=[self.DisNet],
            optimizer_type=self.arg_dict["optim"],
            optimizer_info=dict(
                lr=self.arg_dict["lr"] / 5,
                momentum=self.arg_dict["momentum"],
                weight_decay=self.arg_dict["weight_decay"],
                nesterov=self.arg_dict["nesterov"],
            ),
        )

        # AMP
        if self.arg_dict["use_amp"]:
            construct_print("Now, we will use the amp to accelerate training!")
            from apex import amp

            self.amp = amp
            self.net, self.opti = self.amp.initialize(self.net, self.opti, opt_level="O1")
            self.DisNet, self.opti_D = self.amp.initialize(self.DisNet, self.opti_D, opt_level="O1")
        else:
            self.amp = None

    def forward(self, x):
        return x


class Solver:
    def __init__(self, exp_name: str, arg_dict: dict, path_dict: dict, apex=None):
        super(Solver, self).__init__()
        self.exp_name = exp_name
        self.arg_dict = arg_dict
        self.patch = self.arg_dict["patch"]
        self.path_dict = path_dict

        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()

        self.tr_data_path = self.arg_dict["rgb_data"]["tr_data_path"]
        self.te_data_list = self.arg_dict["rgb_data"]["te_data_list"]

        self.save_path = self.path_dict["save"]
        self.save_pre = self.arg_dict["save_pre"]
        self.in_size = self.arg_dict["input_size"]

        if self.arg_dict["tb_update"] > 0:
            self.tb_recorder = TBRecorder(tb_path=self.path_dict["tb"])
        if self.arg_dict["xlsx_name"]:
            self.xlsx_recorder = XLSXRecoder(xlsx_path=self.path_dict["xlsx"])

        # 依赖与前面属性的属性


        pprint(self.arg_dict)

        self.MAV = MAV(arg_dict=arg_dict, path_dict=path_dict)

        if self.arg_dict["resume_mode"] == "test":
            # resume model only to test model.
            # self.start_epoch is useless
            resume_checkpoint(
                model=self.MAV.net, load_path=self.path_dict["final_state_net"], mode="onlynet",
            )
            return

        if self.arg_dict["resume_mode"] == "train":
            # resume model to train the model
            self.start_epoch = resume_checkpoint(
                model=self.MAV.net,
                optimizer=self.MAV.opti,
                scheduler=self.sche,
                amp=self.MAV.amp,
                exp_name=self.exp_name,
                load_path=self.path_dict["final_full_net"],
                mode="all",
            )
            resume_checkpoint(
                model=self.MAV.DisNet, load_path=self.path_dict["final_Dis_state_net"], mode="onlynet",
            )
        else:
            # only train a new model.
            self.start_epoch = 0

        return

    def test(self):
        self.MAV.eval()

        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader = create_loader(
                data_path=data_path,
                patch=self.arg_dict["patch"],
                training=False,
                prefix=self.arg_dict["prefix"],
                get_length=False,
            )
            self.box_loader = create_boxes(
                data_path=data_path,
            )
            self.save_path = os.path.join(self.path_dict["save"], data_name)
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self._test_process(save_pre=self.save_pre)
            msg = f"Results on the testset({data_name}:'{data_path}'): \n {results}"
            construct_print(msg)
            write_data_to_file(msg, self.path_dict["te_log"])

            total_results[data_name] = results

        # if self.arg_dict["xlsx_name"]:
        #     # save result into xlsx file.
        #     self.xlsx_recorder.write_xlsx(self.exp_name, total_results)

    def _test_process(self, save_pre):
        loader = self.te_loader
        box_loader = self.box_loader

        tqdm_iter = tqdm(enumerate(zip(loader, box_loader)), total=len(loader), leave=False)
        ACC_value, SE, SP, F1, AUC, SE_AV, SP_AV, F1_AV, BACC, ACC_AV = [], [], [], [], [], [], [], [], [], []
        for test_batch_id, te_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{test_batch_id + 1}")
            test_data, box_data = te_data
            if self.patch:
                with torch.no_grad():
                    in_imgs, in_norms, in_mask_v, in_mask_a, in_mask, in_names, h, w, in_ori = test_data
                    box, _ = box_data
                    h, w = h[0], w[0]
                    in_imgs = in_imgs.to(self.dev, non_blocking=True)
                    in_norms = in_norms.to(self.dev, non_blocking=True)
                    outputs_v, outputs_a, outputs = self.merge_img(h, w, in_imgs, in_norms)

                # outputs_np = outputs.sigmoid().cpu().detach()
                for item_id, mask_ in enumerate(in_mask):
                    gt_img = self.to_pil(mask_)
                    out_img, cal_img = self.get_merge(h, w, outputs)
                    out_img_v, cal_img_v = self.get_merge(h, w, outputs_v)
                    out_img_a, cal_img_a = self.get_merge(h, w, outputs_a)
                    crop_w, crop_h = in_ori.shape[2], in_ori.shape[3]
                    crop = transforms.CenterCrop((crop_w, crop_h))
                    out_img, cal_img = crop(out_img), Tensor_crop(cal_img, crop_w, crop_h)
                    out_img_v, cal_img_v = crop(out_img_v), Tensor_crop(cal_img_v, crop_w, crop_h)
                    out_img_a, cal_img_a = crop(out_img_a), Tensor_crop(cal_img_a, crop_w, crop_h)
                    gt_img = crop(gt_img)
                    self.save_test_visualization(out_img_a, out_img_v, out_img, in_names, gt_img)
                    acc, se, sp, f1, auc, se_av, sp_av, f1_av, bacc, acc_av = misc_measures_evaluation(in_mask, cal_img,
                                                                                                       cal_img_v,
                                                                                                       cal_img_a,
                                                                                                       in_mask_v,
                                                                                                       in_mask_a,
                                                                                                       box)
                    ACC_value.append(acc), SE.append(se), SP.append(sp), F1.append(f1), AUC.append(auc)
                    SE_AV.append(se_av), SP_AV.append(sp_av), F1_AV.append(f1_av), BACC.append(bacc), ACC_AV.append(
                        acc_av)

            else:
                with torch.no_grad():
                    in_imgs, in_norms, in_mask_v, in_mask_a, in_mask, in_names, in_ori = test_data
                    box, _ = box_data
                    in_imgs = in_imgs.to(self.dev, non_blocking=True)
                    in_norms = in_norms.to(self.dev, non_blocking=True)
                    outputs_v, outputs_a, outputs_o = self.MAV.net(in_imgs, in_norms)
                self.save_test_visualization(outputs_a, outputs_v, outputs_o, in_names, in_mask_paths)
                acc, se, sp, f1, auc, se_av, sp_av, f1_av, bacc, acc_av = misc_measures_evaluation(in_mask, outputs_o,
                                                                                                   outputs_v,
                                                                                                   outputs_a,
                                                                                                   in_mask_v,
                                                                                                   in_mask_a,
                                                                                                   box)
                ACC_value.append(acc), SE.append(se), SP.append(sp), F1.append(f1), AUC.append(auc)
                SE_AV.append(se_av), SP_AV.append(sp_av), F1_AV.append(f1_av), BACC.append(bacc), ACC_AV.append(
                    acc_av)

        return {"ACC:": np.mean(ACC_value), "SE:": np.mean(SE), "SP:": np.mean(SP), "F1:": np.mean(F1),
                "AUC:": np.mean(AUC), "SE_AV:": np.mean(SE_AV), "SP_AV:": np.mean(SP_AV), "F1_AV": np.mean(F1_AV),
                "BACC:": np.mean(BACC), "ACC_AV:": np.mean(ACC_AV)}

    def mid_test(self, curr_iter):
        self.MAV.net.eval()

        total_results = {}
        for data_name, data_path in self.te_data_list.items():
            construct_print(f"Testing with testset: {data_name}")
            self.te_loader = create_loader(
                data_path=data_path,
                patch=self.arg_dict["patch"],
                training=False,
                prefix=self.arg_dict["prefix"],
                get_length=False,
            )
            self.box_loader = create_boxes(
                data_path=data_path,
            )
            self.save_path = os.path.join(self.path_dict["save"], data_name + "_" + str(curr_iter))
            if not os.path.exists(self.save_path):
                construct_print(f"{self.save_path} do not exist. Let's create it.")
                os.makedirs(self.save_path)
            results = self._test_process_mid(save_pre=self.save_pre)
            msg = f"Results on the testset, curr_iter:'{curr_iter}'({data_name}:'{data_path}'): \n {results}"
            construct_print(msg)
            write_data_to_file(msg, self.path_dict["te_log"])

            total_results[data_name] = results

        self.MAV.train()

    def _test_process_mid(self, save_pre):
        loader = self.te_loader
        box_loader = self.box_loader

        tqdm_iter = tqdm(enumerate(zip(loader, box_loader)), total=len(loader), leave=False)
        ACC_value, SE, SP, F1, AUC, SE_AV, SP_AV, F1_AV, BACC, ACC_AV = [], [], [], [], [], [], [], [], [], []
        for test_batch_id, te_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.exp_name}: te=>{test_batch_id + 1}")
            test_data, box_data = te_data
            if self.patch:
                with torch.no_grad():
                    in_imgs, in_norms, in_mask_v, in_mask_a, in_mask, in_names, h, w, in_ori = test_data
                    box, _ = box_data
                    h, w = h[0], w[0]
                    in_imgs = in_imgs.to(self.dev, non_blocking=True)
                    in_norms = in_norms.to(self.dev, non_blocking=True)
                    outputs_v, outputs_a, outputs = self.merge_img(h, w, in_imgs, in_norms)

                # outputs_np = outputs.sigmoid().cpu().detach()
                for item_id, mask_ in enumerate(in_mask):
                    gt_img = self.to_pil(mask_)
                    out_img, cal_img = self.get_merge(h, w, outputs)
                    out_img_v, cal_img_v = self.get_merge(h, w, outputs_v)
                    out_img_a, cal_img_a = self.get_merge(h, w, outputs_a)
                    crop_w, crop_h = in_ori.shape[2], in_ori.shape[3]
                    crop = transforms.CenterCrop((crop_w, crop_h))
                    out_img, cal_img = crop(out_img), Tensor_crop(cal_img, crop_w, crop_h)
                    out_img_v, cal_img_v = crop(out_img_v), Tensor_crop(cal_img_v, crop_w, crop_h)
                    out_img_a, cal_img_a = crop(out_img_a), Tensor_crop(cal_img_a, crop_w, crop_h)
                    gt_img = crop(gt_img)
                    self.save_test_visualization(out_img_a, out_img_v, out_img, in_names, gt_img)
                    acc, se, sp, f1, auc, se_av, sp_av, f1_av, bacc, acc_av = misc_measures_evaluation(in_mask, cal_img,
                                                                                                       cal_img_v,
                                                                                                       cal_img_a,
                                                                                                       in_mask_v,
                                                                                                       in_mask_a,
                                                                                                       box)
                    ACC_value.append(acc), SE.append(se), SP.append(sp), F1.append(f1), AUC.append(auc)
                    SE_AV.append(se_av), SP_AV.append(sp_av), F1_AV.append(f1_av), BACC.append(bacc)
                    ACC_AV.append(acc_av)
            else:
                with torch.no_grad():
                    in_imgs, in_norms, in_mask_v, in_mask_a, in_mask, in_names, in_ori, in_mask_paths = test_data
                    box, _ = box_data
                    in_imgs = in_imgs.to(self.dev, non_blocking=True)
                    in_norms = in_norms.to(self.dev, non_blocking=True)
                    outputs_v, outputs_a, outputs_o = self.MAV.net(in_imgs, in_norms)
                self.save_test_visualization(outputs_a, outputs_v, outputs_o, in_names, in_mask_paths)
                outputs_a, outputs_v, outputs_o = outputs_a.sigmoid().cpu().detach(), outputs_v.sigmoid().cpu().detach(), outputs_o.sigmoid().cpu().detach()
                outputs_a, outputs_v, outputs_o = self.to_pil(outputs_a.squeeze(0)), self.to_pil(outputs_v.squeeze(0)), self.to_pil(outputs_o.squeeze(0))
                acc, se, sp, f1, auc, se_av, sp_av, f1_av, bacc, acc_av = misc_measures_evaluation(in_mask, outputs_o,
                                                                                                   outputs_v,
                                                                                                   outputs_a,
                                                                                                   in_mask_v,
                                                                                                   in_mask_a,
                                                                                                   box)
                ACC_value.append(acc), SE.append(se), SP.append(sp), F1.append(f1), AUC.append(auc)
                SE_AV.append(se_av), SP_AV.append(sp_av), F1_AV.append(f1_av), BACC.append(bacc), ACC_AV.append(
                    acc_av)

        return {"ACC:": np.mean(ACC_value), "SE:": np.mean(SE), "SP:": np.mean(SP), "F1:": np.mean(F1),
                "AUC:": np.mean(AUC), "SE_AV:": np.mean(SE_AV), "SP_AV:": np.mean(SP_AV),
                "F1_AV": np.mean(F1_AV),
                "BACC:": np.mean(BACC), "ACC_AV:": np.mean(ACC_AV)}

    def merge_img(self, h, w, img, norms):
        outputs, outputs_v, outputs_a = [], [], []
        img = img.squeeze()
        norms = norms.squeeze(0)
        for i in range((2 * h - 1) * (2 * w - 1)):
            outputs_v_, outputs_a_, outputs_o_ = self.MAV.net(img[i:i + 1], norms[i:i + 1])
            outputs.append(outputs_o_.squeeze())
            outputs_v.append(outputs_v_.squeeze())
            outputs_a.append(outputs_a_.squeeze())

        return outputs_v, outputs_a, outputs

    def get_merge(self, h, w, img):
        pre_img = img
        img = []
        img_merge = pre_img
        for i in range(2 * h - 1):
            for j in range(2 * w - 2):
                if j == 0:
                    img1 = pre_img[i * (2 * w - 1) + j][:, :]
                    img1 = img1.sigmoid().cpu().detach()
                    img1 = self.to_pil(img1.unsqueeze(0))
                else:
                    img1 = img_merge
                img1 = np.array(img1)
                img1 = (img1 - np.min(img1)) / (img1.ptp() + 1e-4)
                img2 = pre_img[i * (2 * w - 1) + j + 1][:, :]
                img2 = img2.sigmoid().cpu().detach()
                img2 = self.to_pil(img2.unsqueeze(0))
                img2 = np.array(img2)
                img2 = (img2 - np.min(img2)) / (img2.ptp() + 1e-4)
                img_merge = imgFusion(img1, img2, overlap=self.in_size // 2, left_right=True)
            if w > 1:
                img.append(img_merge)
            else:
                if i == 0:
                    for k in range(len(img_merge)):
                        img_merge[k] = img_merge[k].sigmoid().cpu().detach()
                img.extend(img_merge)
        for i in range(len(img) - 1):
            if i == 0:
                img1 = img[i]
            else:
                img1 = img_merge
            img1 = np.array(img1)
            img1 = (img1 - np.min(img1)) / (img1.ptp() + 1e-4)
            img2 = img[i + 1]
            img2 = np.array(img2)
            img2 = (img2 - np.min(img2)) / (img2.ptp() + 1e-4)
            img_merge = imgFusion(img1, img2, overlap=self.in_size // 2, left_right=False)
        cal_img = img_merge
        img_merge = Image.fromarray(np.uint16(img_merge * 65535))
        return img_merge, cal_img

    def save_visualization(self, img, mask_A, mask_V, outputs_A, outputs_V, outputs_O, global_step, prefix):
        self.save_path = self.path_dict["save"]
        output_images_path = Path(self.save_path + '/' + prefix)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'  # 可以借鉴

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = img
        instance_masks_A = mask_A
        instance_masks_V = mask_V

        normalization = {
            'mean': [.485, .456, .406],
            'std': [.229, .224, .225]
        }
        mean = torch.tensor(normalization['mean'], dtype=torch.float32)
        std = torch.tensor(normalization['std'], dtype=torch.float32)

        denormalizator = Normalize((-mean / std), (1.0 / std))
        image_blob = images[0]
        image = denormalizator(image_blob).cpu().numpy() * 255
        image = image.transpose((1, 2, 0))  # 初始为图像的输出形式

        # print(instance_masks_A.cpu().numpy())
        # print(torch.sigmoid(outputs_A).detach().cpu().numpy())
        gt_instance_masks_A = instance_masks_A.cpu().numpy()
        predicted_instance_masks_A = torch.sigmoid(outputs_A).detach().cpu().numpy()
        gt_instance_masks_V = instance_masks_V.cpu().numpy()
        predicted_instance_masks_V = torch.sigmoid(outputs_V).detach().cpu().numpy()
        predicted_instance_masks_O = torch.sigmoid(outputs_O).detach().cpu().numpy()

        image = image[:, :, ::-1]

        viz_image = []
        for gt_mask_A, predicted_mask_A, gt_mask_V, predicted_mask_V, predicted_mask_O in zip(gt_instance_masks_A,
                                                                                              predicted_instance_masks_A,
                                                                                              gt_instance_masks_V,
                                                                                              predicted_instance_masks_V,
                                                                                              predicted_instance_masks_O):
            gt_mask_A = draw_green(gt_mask_A.transpose((1, 2, 0)))
            predicted_A = draw_probmap(predicted_mask_A.transpose((1, 2, 0)))
            gt_mask_V = draw_red(gt_mask_V.transpose((1, 2, 0)))
            predicted_V = draw_probmap(predicted_mask_V.transpose((1, 2, 0)))
            predicted_O = draw_probmap(predicted_mask_O.transpose((1, 2, 0)))
            pre_v = draw_red(predicted_mask_V.transpose((1, 2, 0)))
            pre_a = draw_green(predicted_mask_A.transpose((1, 2, 0)))
            mask_av = cv2.addWeighted(gt_mask_A, 1, gt_mask_V, 1, 0)
            merge_av = cv2.addWeighted(pre_a, 1, pre_v, 1, 0)
            viz_image.append(
                np.hstack((image, predicted_O, gt_mask_A, predicted_A, gt_mask_V, predicted_V, mask_av, merge_av)))
            break
        viz_image = np.vstack(viz_image)

        result = viz_image.astype(np.uint8)
        _save_image('instance_segmentation', result)

    def save_test_visualization(self, out_a, out_v, out_o, name, in_mask_paths):
        output_images_path = Path(self.save_path)

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)

        def _save_image(idx, image):
            cv2.imwrite(str(output_images_path / f'{name[idx]}_full.png'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if self.patch:
            outputs_np_v, outputs_np_a, outputs_np_o = out_v, out_a, out_o
        else:
            outputs_np_v = out_v.sigmoid().cpu().detach()
            outputs_np_a = out_a.sigmoid().cpu().detach()
            outputs_np_o = out_o.sigmoid().cpu().detach()

        if self.patch:
            gt_img = in_mask_paths
            out_img_v = outputs_np_v.resize(gt_img.size, resample=Image.NEAREST)
            out_img_a = outputs_np_a.resize(gt_img.size, resample=Image.NEAREST)
            out_img_o = outputs_np_o.resize(gt_img.size, resample=Image.NEAREST)
            out_img_v, out_img_a = av_preds(np.array(out_img_v), np.array(out_img_a))
            out_img_v_ = draw_pre_blue(out_img_v, patch=self.patch)
            out_img_a_ = draw_pre_red(out_img_a, patch=self.patch)
            res = Image.fromarray(np.uint8(cv2.addWeighted(out_img_a_, 1, out_img_v_, 1, 0)))

            if self.save_pre:
                oimg_path = os.path.join(self.save_path, name[0] + ".png")
                res.save(oimg_path)

            res, out_o = np.array(res), np.array(out_img_o)
            res = res[:, :, ::-1]
            viz_image = []
            # img_o = cv2.cvtColor(out_o, cv2.COLOR_GRAY2BGR)
            out_img_o_av = draw_pre_av(out_o, out_img_v, out_img_a, patch=self.patch)
            img_o = draw_vessel(out_o)
            viz_image.append(np.hstack((img_o, res, out_img_o_av)))
            viz_image = np.vstack(viz_image)

            result = viz_image.astype(np.uint8)
            _save_image(0, result)
        else:
            for item_id, out_item in enumerate(zip(outputs_np_v, outputs_np_a, outputs_np_o)):
                gimg_path = os.path.join(in_mask_paths[item_id])
                gt_img = Image.open(gimg_path).convert("L")
                out_item_v, out_item_a, out_item_o = out_item
                out_img_v = self.to_pil(out_item_v).resize(gt_img.size, resample=Image.NEAREST)
                out_img_a = self.to_pil(out_item_a).resize(gt_img.size, resample=Image.NEAREST)
                out_img_o = self.to_pil(out_item_o).resize(gt_img.size, resample=Image.NEAREST)
                out_img_v, out_img_a = av_preds(np.array(out_img_v), np.array(out_img_a))
                out_img_v = draw_pre_blue(out_img_v, patch=self.patch)
                out_img_a = draw_pre_red(out_img_a, patch=self.patch)
                res = self.to_pil(cv2.addWeighted(out_img_a, 1, out_img_v, 1, 0))

                if self.save_pre:
                    oimg_path = os.path.join(self.save_path, name[item_id] + ".png")
                    res.save(oimg_path)

                res, out_o = np.array(res), np.array(out_img_o)
                res = res[:, :, ::-1]
                viz_image = []
                img_o = draw_vessel(out_o)
                viz_image.append(np.hstack((img_o, res)))
                viz_image = np.vstack(viz_image)

                result = viz_image.astype(np.uint8)
                _save_image(item_id, result)


def Tensor_crop(cal, w, h):
    img_w, img_h = cal.shape
    cut_w = (img_w - w) // 2
    cut_h = (img_h - h) // 2
    cal = cal[img_w - cut_w - w:img_w - cut_w, img_h - cut_h - h:img_h - cut_h]
    return cal
