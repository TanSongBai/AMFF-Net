import os
from scipy import io as sio
import torch
import functools
import numpy as np
import pandas as pd
import clip
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


def get_class_names(name, mtl):
    dir_txt = {0: './IQA_Database/AGIQA-3K_index.txt',
               1: './IQA_Database/AIGCIQA2023_index.txt',
               3: './IQA_Database/PKU-I2IQA_index.txt'}

    textpath = dir_txt[mtl]
    with open(textpath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    f.close()
    all_name = []
    all_class = []
    for i in lines:
        i_name, i_class = i.split('\t')[1], i.split('\t')[0]
        all_name.append(i_name)
        all_class.append(i_class)
    select_idx = all_name.index(name)
    class_id = all_class[select_idx]
    return class_id


class ImageDataset4(Dataset):
    def __init__(self, txt_file, img_dir, mtl, preprocess, test, is_aigc2013=False,
                 get_loader=get_default_img_loader, get_class=False):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            img_dir (string): Directory of the images.
            preprocess (callable, optional): transform to be applied on a sample.
        """
        self.img_paths = []
        self.is_aigc2013 = is_aigc2013
        self.mos1 = []
        self.mos2 = []
        self.mos3 = []
        self.all_names = []
        self.con_text_prompts = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:  # 读取label文件
                self.img_paths.append(os.path.join(img_dir, line.split('\t')[0]))
                self.all_names.append(os.path.basename(line.split('\t')[0]))
                self.mos1.append(float(line.split('\t')[1]))
                if not is_aigc2013:
                    self.mos2.append(float(line.split('\t')[2]))
                    self.mos3.append(float(line.split('\t')[3]))
                    self.con_text_prompts.append([line.split('\t')[4]])
                else:
                    self.mos2.append(0.0)
                    self.mos3.append(float(line.split('\t')[2]))
                    self.con_text_prompts.append([line.split('\t')[3]])
        # print('%d txt data successfully loaded!' % self.__len__())
        self.mtl = mtl
        self.img_dir = img_dir
        self.get_class = get_class
        self.loader = get_loader()
        self.preprocess = preprocess
        self.test = test

    def __getitem__(self, index):
        image_name = self.img_paths[index]
        I = self.loader(image_name)
        I_l = self.preprocess[0](I)
        I_m = self.preprocess[1](I)
        I_s = self.preprocess[2](I)
        con_text_prompts = self.con_text_prompts[index]
        con_tokens = torch.cat([clip.tokenize(prompt) for prompt in con_text_prompts])
        mos_q = self.mos1[index]
        mos_a = self.mos2[index]
        mos_c = self.mos3[index]
        sample = {'mos_q': mos_q, 'mos_a': mos_a, 'mos_c': mos_c,
                  'img_l': I_l, 'img_m': I_m, 'img_s': I_s,
                  'con_tokens': con_tokens, 'img_name': image_name}
        if self.get_class:
            sample['class'] = get_class_names(self.all_names[index], self.mtl)
        return sample

    def __len__(self):
        return len(self.img_paths)

