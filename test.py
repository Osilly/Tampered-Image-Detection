import torch
import os
from glob import glob
import numpy as np
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import *


class Test:
    def __init__(
        self,
        model_name="",
        test_path="data/test",
        result_path="result",
        color_map=[],
        img_size=512,
        batch_size=50,
        num_workers=0,
        model_step=0,
        threshold=0.5,
        device="cuda:0",
    ):
        self.model_name = model_name
        self.test_path = test_path
        self.result_path = result_path
        self.color_map = color_map
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_step = model_step
        self.threshold = threshold
        self.device = device

        self.fake_mask = torch.tensor([])
        self.shapes = []

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def dataload(self):
        test_transform = A.Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
        test_iter = Dataset(self.test_path, self.color_map, test_transform, "test")
        self.test_iter = DataLoader(
            test_iter,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def build_model(self, model):
        self.model = model.to(self.device)

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, "model_params_%07d.pt" % step))
        self.model.load_state_dict(params["model"])

    def test(self):
        model_list = glob(os.path.join(self.result_path, self.model_name, "*.pt"))
        model_step = 0
        if not len(model_list) == 0:
            if self.model_step and os.path.exists(
                os.path.join(
                    os.path.join(self.result_path, self.model_name),
                    "model_params_%07d.pt" % self.model_step,
                )
            ):
                model_step = self.model_step
            else:
                model_list.sort()
                model_step = int(model_list[-1].split("_")[-1].split(".")[0])
            self.load_model(os.path.join(self.result_path, self.model_name), model_step)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.model.eval()
        with torch.no_grad():
            for i, (x, shape) in enumerate(
                tqdm(self.test_iter, total=len(self.test_iter))
            ):
                x = x.to(dtype=torch.float, device=self.device)
                out = self.model(x).to(self.device)
                fake_mask = torch.sigmoid(out).detach().cpu()
                for j, img in enumerate(fake_mask):
                    img = label2mask(img, shape[j], self.threshold)
                    dir = os.path.join(self.result_path, self.model_name, "images")
                    folder = os.path.exists(dir)
                    if not folder:
                        os.makedirs(dir)
                    save_img(
                        img,
                        os.path.join(dir, str(i * self.batch_size + j + 1) + ".png"),
                    )


def save_img(img, path):
    """tensor -> PIL image, save image in path

    Args:
        img (tesor): H*W*C
    """
    img = transforms.ToPILImage()(img).convert("RGB")
    img.save(path)


def label2mask(label, shape, threshold=0.5):
    label = ((label > threshold) * 255.0).to(torch.uint8)
    label = np.asarray(label.permute(1, 2, 0))
    mask = A.Resize(shape[0], shape[1])(image=label)["image"]
    return mask


if __name__ == "__main__":
    model_dic = get_model_dic()
    batch_sizes = [50, 50]
    for i, (model_name, model) in enumerate(model_dic.items()):
        test = Test(
            model_name=model_name,
            test_path="data/test",
            result_path="result",
            color_map=[[0, 0, 0], [255, 255, 255]],
            img_size=512,
            batch_size=batch_sizes[i],
            num_workers=8,
            model_step=0,
            threshold=0.5,
            device="cuda:1",
        )
        test.setup_seed(2022 + i)
        test.dataload()
        test.build_model(model)
        test.test()
