import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import os
import numpy as np
from glob import glob
from dataset import *
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import time
from score import *
from tqdm import tqdm
from model import *


class Train:
    def __init__(
        self,
        model_name="",
        train_path="data/train",
        result_path="result",
        color_map=[],
        img_size=512,
        num_epochs=200,
        batch_size=50,
        num_workers=0,
        lr=1e-4,
        weight_decay=1e-4,
        threshold=0.5,
        device="cuda:0",
        resume=False,
    ):
        self.model_name = model_name
        self.train_path = train_path
        self.result_path = result_path
        self.color_map = color_map
        self.img_size = img_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.device = device
        self.resume = resume

        self.train_loss = []
        self.train_iou = []
        self.train_f1 = []
        self.valid_loss = []
        self.valid_iou = []
        self.valid_f1 = []
        self.start_epoch = 1

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def dataload(self):
        train_transform = A.Compose(
            [
                A.RandomResizedCrop(self.img_size, self.img_size),
                A.CoarseDropout(
                    max_holes=2,
                    max_height=50,
                    max_width=50,
                    min_holes=1,
                    min_height=10,
                    min_width=10,
                    fill_value=[230, 230, 220],
                    mask_fill_value=[255, 255, 255],
                    p=0.5,
                ),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(transpose_mask=True),
            ]
        )
        path_len = len(get_paths(self.train_path, "train")[0])
        train_size = int(path_len * 0.95)
        valid_size = path_len - train_size
        train_iter, valid_iter = torch.utils.data.random_split(
            Dataset(self.train_path, self.color_map, train_transform, "train"),
            [train_size, valid_size],
        )
        self.train_iter = DataLoader(
            train_iter,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.valid_iter = DataLoader(
            valid_iter,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def build_model(self, model):
        self.model = model.to(self.device)

    def define_loss(self):
        self.seg_criterion = nn.BCEWithLogitsLoss().to(device=self.device)

    def define_optim(self):
        self.optim = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optim, T_max=10, eta_min=1e-2 * self.lr)

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params["model"] = self.model.state_dict()
        params["optim"] = self.optim.state_dict()
        params["scheduler"] = self.scheduler.state_dict()
        params["train_loss"] = self.train_loss
        params["train_iou"] = self.train_iou
        params["train_f1"] = self.train_f1
        params["valid_loss"] = self.valid_loss
        params["valid_iou"] = self.valid_iou
        params["valid_f1"] = self.valid_f1
        params["start_epoch"] = self.start_epoch
        torch.save(params, os.path.join(path, "model_params_%07d.pt" % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.model.load_state_dict(params["model"])
        self.optim.load_state_dict(params["optim"])
        self.scheduler.load_state_dict(params["scheduler"])
        self.train_loss = params["train_loss"]
        self.train_iou = params["train_iou"]
        self.train_f1 = params["train_f1"]
        self.valid_loss = params["valid_loss"]
        self.valid_iou = params["valid_iou"]
        self.valid_f1 = params["valid_f1"]
        self.start_epoch = params["start_epoch"]

    def train(self):
        if self.resume:
            model_list = glob(os.path.join(self.result_path, self.model_name, "*.pt"))
            if len(model_list) != 0:
                model_list.sort()
                start_step = int(model_list[-1].split("_")[-1].split(".")[0])
                self.load_model(
                    os.path.join(self.result_path, self.model_name), start_step
                )
                print("load success!")
        print(model_name + ", training starts!")
        start_time = time.time()
        for epoch in range(self.start_epoch, 1 + self.num_epochs):
            self.model.train()
            train_loss = 0
            train_iou = 0
            train_f1 = 0
            valid_loss = 0
            valid_iou = 0
            valid_f1 = 0
            # train
            print("train:")
            for x, y in tqdm(self.train_iter, total=len(self.train_iter)):
                self.optim.zero_grad()
                x, y = (
                    x.to(dtype=torch.float, device=self.device),
                    torch.unsqueeze(y, 1).to(dtype=torch.float, device=self.device),
                )
                out = self.model(x).to(self.device)
                loss = self.seg_criterion(out, y)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_iou += get_JS(
                    torch.sigmoid(out), y.to(torch.uint8), self.threshold
                )
                train_f1 += get_F1(
                    torch.sigmoid(out), y.to(torch.uint8), self.threshold
                )
            self.scheduler.step()
            train_loss /= len(self.train_iter)
            train_iou /= len(self.train_iter)
            train_f1 /= len(self.train_iter)
            self.train_loss.append(train_loss)
            self.train_iou.append(train_iou)
            self.train_f1.append(train_f1)

            # valid
            self.model.eval()
            with torch.no_grad():
                print("test:")
                for x, y in tqdm(self.valid_iter, total=len(self.valid_iter)):
                    x, y = (
                        x.to(dtype=torch.float, device=self.device),
                        torch.unsqueeze(y, 1).to(dtype=torch.float, device=self.device),
                    )
                    out = self.model(x).to(self.device)
                    loss = self.seg_criterion(out, y)
                    valid_loss += loss.item()
                    valid_iou += get_JS(
                        torch.sigmoid(out), y.to(torch.uint8), self.threshold
                    )
                    valid_f1 += get_F1(
                        torch.sigmoid(out), y.to(torch.uint8), self.threshold
                    )
                valid_loss /= len(self.valid_iter)
                valid_iou /= len(self.valid_iter)
                valid_f1 /= len(self.valid_iter)
                self.valid_loss.append(valid_loss)
                self.valid_iou.append(valid_iou)
                self.valid_f1.append(valid_f1)

            print(
                "Epoch %d time: %4.4f. \
                    Train Loss: %f, Train IOU: %f, Train F1: %f, Train Score: %f, \
                        Valid Loss: %f, Valid IOU: %f, Valid F1: %f, Valid Score: %f"
                % (
                    epoch,
                    time.time() - start_time,
                    train_loss,
                    train_iou,
                    train_f1,
                    train_iou + train_f1,
                    valid_loss,
                    valid_iou,
                    valid_f1,
                    valid_iou + valid_f1,
                )
            )

            if epoch % 1 == 0:
                self.start_epoch = epoch
                self.save_model(os.path.join(self.result_path, self.model_name), epoch)


if __name__ == "__main__":
    model_dic = get_model_dic()
    batch_sizes = [10, 8]
    i = 0
    for model_name, model in model_dic.items():
        train = Train(
            model_name=model_name,
            train_path="data/train",
            result_path="result",
            color_map=[[0, 0, 0], [255, 255, 255]],
            img_size=512,
            num_epochs=110,
            batch_size=batch_sizes[i],
            num_workers=8,
            lr=3e-4,
            weight_decay=1e-4,
            device="cuda:1",
            threshold=0.4,
            resume=True,
        )
        i += 1
        train.setup_seed(2022 + i)
        train.dataload()
        train.build_model(model)
        train.define_loss()
        train.define_optim()
        train.train()
        print(model_name + ", training finished!")
