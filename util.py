import os
import torch
import numpy as np
import cv2

def img_save(img, img_path):
    cv2.imwrite(img_path, tensor2uint(img))

def save_tensor_to_npy(tensor, filename):
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1,2,0)
    numpy_array = tensor.cpu().numpy()
    np.save(filename, numpy_array)

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def make_dirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_match_dict(model, model_path):
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)