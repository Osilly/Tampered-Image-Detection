import numpy as np
import torch


def cutmix(img, mask, num_class, min_l, max_l, min_similarity):
    """cutmix in only one image

    Args:
        img (tensor): C*H*W, image
        mask (tensor): C*H*W, mask
        num_class (int): the maximum number box to cutmix
        min_l (int): the minimum length of cutmix box
        max_l (int): the maximum length of cutmix box
        min_similarity (float): the minimum similarity threshold of cutmix box

    Returns:
        tuple(tensor): (x, y)
            x: C*H*W, cutmixed image
            y: C*H*W, cutmixed mask
    """
    x, y = img, mask
    for _ in range(num_class):
        l = np.random.randint(min_l, max_l)
        h1, w1 = np.random.randint(0, x.shape[1] - max_l), np.random.randint(
            0, x.shape[2] - max_l
        )
        h2, w2 = np.random.randint(0, x.shape[1] - max_l), np.random.randint(
            0, x.shape[2] - max_l
        )
        a = x[:, h1 : h1 + l, w1 : w1 + l].reshape(1, -1).to(torch.float)
        b = x[:, h2 : h2 + l, w2 : w2 + l].reshape(1, -1).to(torch.float)
        similarity = torch.cosine_similarity(a, b, dim=1)
        if similarity[0] <= min_similarity:
            x[:, h1 : h1 + l, w1 : w1 + l] = x[:, h2 : h2 + l, w2 : w2 + l]
            y[:, h1 : h1 + l, w1 : w1 + l] = 255
    return x, y
