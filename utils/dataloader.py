# -*- coding: utf-8 -*-
import math
import os
import random
from functools import partial

import torch
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)

from config import arg_config
from utils.joint_transforms import Compose, JointResize, RandomHorizontallyFlip, RandomRotate
from utils.misc import construct_print


def _get_suffix(path_list):
    ext_list = list(set([os.path.splitext(p)[1] for p in path_list]))
    if len(ext_list) != 1:
        if ".png" in ext_list:
            ext = ".png"
        elif ".jpg" in ext_list:
            ext = ".jpg"
        elif ".bmp" in ext_list:
            ext = ".bmp"
        elif ".gif" in ext_list:
            ext = ".gif"
        else:
            raise NotImplementedError
        construct_print(f"数据文件夹中包含多种扩展名，这里仅使用{ext}")
    else:
        ext = ext_list[0]
    return ext


def _make_box(root):
    box_path = os.path.join(root, "Box")
    box_list = os.listdir(box_path)
    return [
        (
            os.path.join(box_path, box_name)
        )
        for box_name in box_list
    ]


def _make_dataset(root):
    img_path = os.path.join(root, "Image")
    mask_path = os.path.join(root, "av")

    img_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)
    # for index in range(len(img_list)):
    #     print(img_list[index])
    #     print(mask_list[index])

    img_suffix = _get_suffix(img_list)
    mask_suffix = _get_suffix(mask_list)

    img_list = [os.path.splitext(f)[0] for f in mask_list if f.endswith(mask_suffix)]
    return [
        (
            os.path.join(img_path, img_name + img_suffix),
            os.path.join(mask_path, img_name + mask_suffix),
        )
        for img_name in img_list
    ]


def _read_list_from_file(list_filepath):
    img_list = []
    with open(list_filepath, mode="r", encoding="utf-8") as openedfile:
        line = openedfile.readline()
        while line:
            img_list.append(line.split()[0])
            line = openedfile.readline()
    return img_list


def _make_dataset_from_list(list_filepath, prefix=(".jpg", ".png")):
    img_list = _read_list_from_file(list_filepath)
    return [
        (
            os.path.join(
                os.path.join(os.path.dirname(img_path), "Image"),
                os.path.basename(img_path) + prefix[0],
            ),
            os.path.join(
                os.path.join(os.path.dirname(img_path), "av"),
                os.path.basename(img_path) + prefix[1],
            ),
        )
        for img_path in img_list
    ]


class BoxFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        self.in_size = in_size

        if os.path.isdir(root):
            self.boxes = _make_box(root)
        elif os.path.isfile(root):
            self.boxes = _make_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError

        self.box_transform = transforms.ToTensor()

    def __getitem__(self, index):
        box_path = self.boxes[index]
        box = Image.open(box_path).convert("L")
        box = self.box_transform(box) * 255
        box = box.ge(0.5).float()
        return box, box_path

    def __len__(self):
        return len(self.boxes)


class ImageFolder(Dataset):
    def __init__(self, root, in_size, training, prefix, patch, use_bigt=False):
        self.training = training
        self.patch = patch
        self.use_bigt = use_bigt
        self.in_size = in_size

        if os.path.isdir(root):
            construct_print(f"{root} is an image folder, we will test on it.")
            self.imgs = _make_dataset(root)
        elif os.path.isfile(root):
            construct_print(
                f"{root} is a list of images, we will use these paths to read the "
                f"corresponding image"
            )
            self.imgs = _make_dataset_from_list(root, prefix=prefix)
        else:
            raise NotImplementedError

        if self.training:
            self.joint_transform = Compose(
                [JointResize(in_size), RandomHorizontallyFlip(), RandomRotate(20)]
            )
            img_transform = [transforms.ColorJitter(0.1, 0.1, 0.1)]
            self.mask_transform = transforms.ToTensor()
        elif self.patch:
            # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
            # img_transform = [transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),]
            img_transform = []
            self.mask_transform = transforms.ToTensor()
        else:
            img_transform = [
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
            ]
            self.mask_transform = transforms.Compose([
                transforms.Resize((in_size, in_size), interpolation=Image.BILINEAR),
                transforms.ToTensor()])
        self.img_transform = transforms.Compose(
            [
                *img_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        if self.training:
            if index < len(self.imgs):
                img_path, mask_path = self.imgs[index]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img = Image.open(img_path).convert("RGB")
                img_norm = Image.open(img_path).convert("L")
                img_norm = norm(img_norm)
                mask = Image.open(mask_path).convert("L")
            else:
                splice_choice = [2, 4]
                id_index = random.choice(splice_choice)
                img, img_norm, mask = Splice(self.imgs, id_index)
                img_name = ""
            if self.patch:
                img, img_norm, mask = patch(img, img_norm, mask, self.in_size)
            img, img_norm, mask = self.joint_transform(img, img_norm, mask)
            mask_v, mask_a, mask = mask_av(mask)
            img = self.img_transform(img)
            img_norm = self.mask_transform(img_norm)
            mask_v = self.mask_transform(mask_v) * 255
            mask_a = self.mask_transform(mask_a) * 255
            mask = self.mask_transform(mask) * 255
            if self.use_bigt:
                mask_v = mask_v.ge(0.5).float()  # 二值化
                mask_a = mask_a.ge(0.5).float()
                mask = mask.ge(0.5).float()
            return img, img_norm, mask_v, mask_a, mask, img_name
        else:
            if self.patch:
                img_path, mask_path = self.imgs[index]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img = Image.open(img_path).convert("RGB")
                mask_ori = Image.open(mask_path).convert("L")
                img_norm = Image.open(img_path).convert("L")
                img_norm = norm(img_norm)
                img, img_norm, mask, h, w = self.patch_test(img, img_norm, mask_ori, self.in_size)
                mask_ori = self.mask_transform(mask_ori)
                mask_v, mask_a, mask = mask_av(mask)
                mask_v = self.mask_transform(mask_v) * 255
                mask_a = self.mask_transform(mask_a) * 255
                mask = self.mask_transform(mask) * 255
                if self.use_bigt:
                    mask_v = mask_v.ge(0.5).float()  # 二值化
                    mask_a = mask_a.ge(0.5).float()
                    mask = mask.ge(0.5).float()
                return img, img_norm, mask_v, mask_a, mask, img_name, h, w, mask_ori
            else:
                img_path, mask_path = self.imgs[index]
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img = Image.open(img_path).convert("RGB")
                img_norm = Image.open(img_path).convert("L")
                img_norm = norm(img_norm)
                img = self.img_transform(img)
                img_norm = self.mask_transform(img_norm)
                return img, img_norm, mask_path, img_name

    def __len__(self):
        if self.training:
            return len(self.imgs)*5
        else:
            return len(self.imgs)

    def patch_test(self, img, img_norm, label, in_size):
        w, h = img.size
        p = in_size
        w_ = w % p
        if w_ > 0:
            img = F.pad(img, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
            img_norm = F.pad(img_norm, ((p - w_) // 2, 0, p - w_ - (p - w_) // 2, 0))
            len_w = w // p + 1
        else:
            len_w = w // p
        h_ = h % p
        if h_ > 0:
            img = F.pad(img, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
            img_norm = F.pad(img_norm, (0, (p - h_) // 2, 0, p - h_ - (p - h_) // 2))
            len_h = h // p + 1
        else:
            len_h = h // p
        img = self.img_transform(img)
        img_norm = self.mask_transform(img_norm)
        data_img = []
        data_img_norm = []
        # print(label.size)
        for i in range(2*len_h-1):
            for j in range(2*len_w-1):
                data_img.append(img[:, i * (in_size//2):(i + 2) * (in_size//2), j * (in_size//2):(j + 2) * (in_size//2)])
                data_img_norm.append(
                    img_norm[:, i * (in_size // 2):(i + 2) * (in_size // 2), j * (in_size // 2):(j + 2) * (in_size // 2)])

        data_img = torch.stack(data_img)
        data_img_norm = torch.stack(data_img_norm)
        return data_img, data_img_norm, label, len_h, len_w


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())


def norm(im):
    im = np.array(im)
    _range = np.max(im) - np.min(im)
    if _range != 0:
        im = ((im - np.min(im)) / _range) * 255
    mu = np.mean(im)
    sigma = np.std(im)
    im = (im - mu) / sigma * 255
    im = Image.fromarray(im.reshape(im.shape))
    im = im.convert('L')
    return im


def mask_av(mask):
    # 0-->bg /29, 30, 31, 32, 33 -->vein /76, 91， 90， 108-->artery/ 255-->unknown
    mask = np.array(mask)
    mask[mask == 29] = 2
    mask[mask == 28] = 2
    mask[mask == 30] = 2
    mask[mask == 31] = 2
    mask[mask == 32] = 2
    mask[mask == 33] = 2

    mask[mask == 76] = 3
    mask[mask == 91] = 3
    mask[mask == 90] = 3
    mask[mask == 105] = 3
    mask[mask == 108] = 3
    mask[mask == 1] = 3

    mask_v = np.copy(mask)
    mask_v[mask_v == 150] = 2
    mask_v[mask_v == 149] = 2
    mask_v[mask_v != 2] = 0
    mask_v[mask_v == 2] = 1

    mask_a = np.copy(mask)
    mask_a[mask_a == 150] = 3
    mask_a[mask_a == 149] = 3
    mask_a[mask_a != 3] = 0
    mask_a[mask_a == 3] = 1

    mask_o = np.copy(mask)
    mask_o[mask_o != 0] = 1

    return Image.fromarray(mask_v), Image.fromarray(mask_a), Image.fromarray(mask_o)


def _collate_fn(batch, size_list):
    size = random.choice(size_list)
    img, mask, image_name = [list(item) for item in zip(*batch)]
    img = torch.stack(img, dim=0)
    img = interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    mask = torch.stack(mask, dim=0)
    mask = interpolate(mask, size=(size, size), mode="nearest")
    return img, mask, image_name


def _mask_loader(dataset, shuffle, drop_last, size_list, batch_size=arg_config["batch_size"]):
    return DataLoaderX(
        dataset=dataset,
        collate_fn=partial(_collate_fn, size_list=size_list) if size_list else None,
        batch_size=batch_size,
        num_workers=arg_config["num_workers"],
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
    )


def create_loader(data_path, training, patch, size_list=None, prefix=(".jpg", ".png"), get_length=False):
    if training:
        construct_print(f"Training on: {data_path}")
        imageset = ImageFolder(
            data_path,
            in_size=arg_config["input_size"],
            prefix=prefix,
            use_bigt=arg_config["use_bigt"],
            patch=patch,
            training=True,
        )
        loader = _mask_loader(imageset, shuffle=True, drop_last=True, size_list=size_list)
    else:
        construct_print(f"Testing on: {data_path}")
        imageset = ImageFolder(
            data_path, patch=patch, in_size=arg_config["input_size"], prefix=prefix, training=False,
        )
        loader = _mask_loader(imageset, shuffle=False, drop_last=False, size_list=size_list, batch_size=1)

    if get_length:
        length_of_dataset = len(imageset)
        return loader, length_of_dataset
    else:
        return loader


def create_boxes(data_path, size_list=None, prefix=(".jpg", ".png", ".gif"), get_length=False):
    construct_print(f"Testing boxes on: {data_path}")
    boxset = BoxFolder(
        data_path,
        in_size=arg_config["input_size"],
        prefix=prefix,
    )
    loader = _mask_loader(boxset, shuffle=False, drop_last=False, size_list=size_list, batch_size=1)

    if get_length:
        length_of_dataset = len(boxset)
        return loader, length_of_dataset
    else:
        return loader


def Splice(imgs, nums):
    mlti_img_path = random.sample(imgs, nums)
    img_path, mask_path = mlti_img_path[0]
    img = Image.open(img_path).convert("RGB")
    size = np.min(img.size) if np.min(img.size) % 2 == 0 else np.min(img.size)-1
    w, h = size, size
    input_size_w, input_size_h = 0, 0
    if nums == 2:
        input_size_w = w // nums
        input_size_h = w * 2 // nums
    elif nums == 8:
        input_size_w = w * 2 // nums
        input_size_h = w * 4 // nums
    else:
        input_size_w = int(w // math.sqrt(nums))
        input_size_h = int(w // math.sqrt(nums))
    w_ = w - input_size_w
    h_ = h - input_size_h
    out_img, out_img_norm, out_mask = np.zeros((w, h, 3)), np.zeros((w, h)), np.zeros((w, h))
    c, r = 0, 0
    for i in range(nums):
        img_path, mask_path = mlti_img_path[i]
        img = Image.open(img_path).convert("RGB")
        img_norm = Image.open(img_path).convert("L")
        img_norm = norm(img_norm)
        mask = Image.open(mask_path).convert("L")
        if h_ > 0:
            x1 = random.choice(range(w_))
            y1 = random.choice(range(h_))
            img = img.crop((x1, y1, x1 + input_size_w, y1 + input_size_h))
            img_norm = img_norm.crop((x1, y1, x1 + input_size_w, y1 + input_size_h))
            mask = mask.crop((x1, y1, x1 + input_size_w, y1 + input_size_h))
        else:
            x1 = random.choice(range(w_))
            img = img.crop((x1, 0, x1 + input_size_w, input_size_h))
            img_norm = img_norm.crop((x1, 0, x1 + input_size_w, input_size_h))
            mask = mask.crop((x1, 0, x1 + input_size_w, input_size_h))
        out_img[c * input_size_h:(c + 1) * input_size_h, r * input_size_w:(r + 1) * input_size_w] = img
        out_img_norm[c * input_size_h:(c + 1) * input_size_h, r * input_size_w:(r + 1) * input_size_w] = img_norm
        out_mask[c * input_size_h:(c + 1) * input_size_h, r * input_size_w:(r + 1) * input_size_w] = mask
        if (r+1)*input_size_w == size:
            c += 1
            r = 0
        else:
            r += 1
    out_img = Image.fromarray(np.uint8(out_img))
    out_img_norm = Image.fromarray(np.uint8(out_img_norm))
    out_mask = Image.fromarray(np.uint8(out_mask))
    return out_img, out_img_norm, out_mask


def patch(img, img_norm, label, input_size):
    w, h = img.size
    w_ = w - input_size
    h_ = h - input_size

    if w_ > 0 and h_ > 0:
        x1 = random.choice(range(w_))
        y1 = random.choice(range(h_))
        img = img.crop((x1, y1, x1 + input_size, y1 + input_size))
        img_norm = img_norm.crop((x1, y1, x1 + input_size, y1 + input_size))
        label = label.crop((x1, y1, x1 + input_size, y1 + input_size))
    # elif w_ > 0 and h_ < 0:
    #     x1 = random.choice(range(w_))
    #     y1 = 0
    #     img = img.crop((x1, y1, x1 + h, y1 + h))
    #     img_norm = img_norm.crop((x1, y1, x1 + h, y1 + h))
    #     label = label.crop((x1, y1, x1 + h, y1 + h))
    #     img = img.resize((input_size, input_size), Image.ANTIALIAS)
    #     img_norm = img_norm.resize((input_size, input_size), Image.ANTIALIAS)
    #     label = label.resize((input_size, input_size), Image.ANTIALIAS)
    # elif w_ < 0 and h_ > 0:
    #     x1 = 0
    #     y1 = random.choice(range(h_))
    #     img = img.crop((x1, y1, x1 + w, y1 + w))
    #     img_norm = img_norm.crop((x1, y1, x1 + w, y1 + w))
    #     label = label.crop((x1, y1, x1 + w, y1 + w))
    #     img = img.resize((input_size, input_size), Image.ANTIALIAS)
    #     img_norm = img_norm.resize((input_size, input_size), Image.ANTIALIAS)
    #     label = label.resize((input_size, input_size), Image.ANTIALIAS)
    # else:
    #     img = img.resize((input_size, input_size), Image.ANTIALIAS)
    #     img_norm = img_norm.resize((input_size, input_size), Image.ANTIALIAS)
    #     label = label.resize((input_size, input_size), Image.ANTIALIAS)
    return img, img_norm, label


if __name__ == "__main__":
    te_data_list = arg_config["rgb_data"]["te_data_list"]
    for data_name, data_path in te_data_list.items():
        box_loader = create_boxes(
            data_path=data_path,
            get_length=False,
        )

        te_loader = create_loader(
            data_path=data_path,
            patch=True,
            training=False,
            get_length=False,
        )

        for idx, te_data in enumerate(zip(box_loader, te_loader)):
            # train_inputs, train_masks_v, train_masks_a, *train_other_data = box_data
            # print(f"" f"batch: {idx} ", train_inputs.size(), train_masks_v.size(), train_masks_a.size())
            box_data, test_data = te_data
            box, box_path = box_data
            in_imgs, in_norms, in_mask_v, in_mask_a, in_mask, in_names, h, w, in_ori = test_data
            print(f"" f"batch: {idx} ", box.size(), box_path, in_names)

    # loader = create_loader(
    #     data_path=arg_config["rgb_data"]["tr_data_path"],
    #     training=True,
    #     get_length=False,
    #     size_list=arg_config["size_list"],
    # )
    #
    # for idx, train_data in enumerate(loader):
    #     train_inputs, train_masks_v, train_masks_a, *train_other_data = train_data
    #     print(f"" f"batch: {idx} ", train_inputs.size(), train_masks_v.size(), train_masks_a.size())
