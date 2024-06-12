from torch.utils.data import DataLoader
from ImageDataset4 import ImageDataset4
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def set_dataset4(txt_file, bs, data_set, radius, num_workers, preprocess, mtl, test, get_class=False):
    if mtl == 0:
        is_aigc2023 = True
    else:
        is_aigc2023 = False
    data = ImageDataset4(
        txt_file=txt_file,
        img_dir=data_set,
        mtl=mtl,
        test=test,
        is_aigc2013=is_aigc2023,
        preprocess=preprocess,
        get_class=get_class)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader

class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)

def convert_obj_score(ori_obj_score, MOS):
    """
    func:
        fitting the objetive score to the MOS scale.
        nonlinear regression fit
    """

    def logistic_fun(x, b1, b2, b3, b4, b5):
        return b1 * (0.5 - 1 / np.exp(b2 * (x - b3))) + b4 * x + b5
        # return b5 * np.power(x, 4) + b4 * np.power(x, 3) + b3 * np.power(x, 2) + b2 * x + b1

    # nolinear fit the MOSp
    param_init = [np.max(MOS), np.min(MOS), np.mean(ori_obj_score), 1, np.mean(MOS)]
    popt, pcov = curve_fit(logistic_fun, ori_obj_score, MOS,
                           p0=param_init, ftol=1e-8, maxfev=40000)
    obj_fit_score = logistic_fun(ori_obj_score, popt[0], popt[1], popt[2], popt[3], popt[4])

    return obj_fit_score


def compute_metric(y, y_pred, istrain=False):
    """
    func:
        calculate the sorcc etc
    """
    index_to_del = []
    y = y.flatten()
    y_pred = y_pred.flatten()
    MSE = mean_squared_error
    if not istrain:
        y_pred = convert_obj_score(y_pred, y)
        for i in range(len(y_pred)):
            if y_pred[i] <= 0 or np.isnan(y_pred[i]):
                print("your prediction seems like not quit good, we reconmand you remove it   ", y_pred[i])
                index_to_del.append(i)
        y_pred = np.delete(y_pred, index_to_del)
        y = np.delete(y, index_to_del)
        RMSE = MSE(convert_obj_score(y_pred, y), y) ** 0.5
        PLCC = stats.pearsonr(convert_obj_score(y_pred, y), y)[0]
    else:
        RMSE = MSE(y_pred, y) ** 0.5
        PLCC = stats.pearsonr(y_pred, y)[0]
    SROCC = stats.spearmanr(y_pred, y)[0]
    KROCC = stats.kendalltau(y_pred, y)[0]

    return RMSE, PLCC, SROCC, KROCC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess2(size):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess3(size):
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

