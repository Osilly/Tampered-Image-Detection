from xml.dom import xmlbuilder
import torch
import os
from glob import glob
import numpy as np
import random
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from model import get_model_dic
from dataset import *
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from tqdm import tqdm


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
    result_path = "result"
    device = "cuda:1"
    color_map = [[0, 0, 0], [255, 255, 255]]
    test_path = "data/test"
    model_dic = get_model_dic()
    batch_size = 45
    models = []
    w = [0.4, 0.6]
    model_steps = [108, 110]
    threshold = 0.4

    for i, (model_name, model) in enumerate(model_dic.items()):
        model_list = glob(os.path.join(result_path, model_name, "*.pt"))
        model_step = model_steps[i]
        if not len(model_list) == 0:
            if model_step and os.path.exists(
                os.path.join(
                    os.path.join(result_path, model_name),
                    "model_params_%07d.pt" % model_step,
                )
            ):
                model_step = model_step
            else:
                model_list.sort()
                model_step = int(model_list[-1].split("_")[-1].split(".")[0])
            params = torch.load(
                os.path.join(
                    os.path.join(result_path, model_name),
                    "model_params_%07d.pt" % model_step,
                ),
            )
            model.load_state_dict(params["model"])
            model.to(device=device)
            models.append(model)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            break

    test_transform = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )
    test_iter = Dataset(test_path, color_map, test_transform, "test")
    test_iter = DataLoader(
        test_iter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    with torch.no_grad():
        for i, (x, shape) in enumerate(tqdm(test_iter, total=len(test_iter))):
            fake_mask_fusion = torch.zeros(batch_size, 1, 512, 512)
            for j, model in enumerate(models):
                model.eval()
                x = x.to(dtype=torch.float, device=device)
                out = model(x).to(device)
                fake_mask = torch.sigmoid(out).detach().cpu()
                fake_mask_fusion += w[j] * fake_mask
            for j, img in enumerate(fake_mask_fusion):
                img = label2mask(img, shape[j], threshold)
                dir = os.path.join(result_path, "model_fusion", "images")
                folder = os.path.exists(dir)
                if not folder:
                    os.makedirs(dir)
                save_img(
                    img,
                    os.path.join(dir, str(i * batch_size + j + 1) + ".png"),
                )
