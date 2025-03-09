# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential

from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
from pathlib import Path


# from DETR.ops.modules.ms_deform_attn import MSDeformAttn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3_CCNet(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_CCNet, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.CCNet = CrissCrossAttention(in_channels=c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        x = self.CCNet(x)
        return x


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.que_proj4 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj4 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj4 = nn.Linear(d_model, h * self.d_v)  # value projection
        # key, query, value projections for all heads
        self.que_proj1 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj1 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj1 = nn.Linear(d_model, h * self.d_v)  # value projection
        self.que_proj2 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj2 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj2 = nn.Linear(d_model, h * self.d_v)  # value projection
        self.que_proj3 = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj3 = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj3 = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(3 * h * self.d_v, d_model)  # output projection
        # self.out_proj1 = nn.Linear(h * self.d_v, d_model)  # output projection
        # self.reduce = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        # self.reduce2 = nn.Conv1d(3 * d_model, d_model, kernel_size=1)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, model1, model2, model3):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = model1.shape[:2]
        nk = model1.shape[1]
        q3 = self.que_proj3(model3).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k3 = self.key_proj3(model3).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v3 = self.val_proj3(model3).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        q1 = self.que_proj1(model1).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k1 = self.key_proj1(model1).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v1 = self.val_proj1(model1).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        q2 = self.que_proj2(model2).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k2 = self.key_proj2(model2).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v2 = self.val_proj2(model2).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # (b_s, nq, d_model)
        attn3 = torch.matmul(q3, k3) / np.sqrt(self.d_k)
        attn3 = torch.softmax(attn3, -1)
        attn3 = self.attn_drop(attn3)
        model3_1 = torch.matmul(attn3, v3).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        model3 = model3_1+model3

        attn1 = torch.matmul(q1, k1) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        attn1 = torch.softmax(attn1, -1)
        attn1 = self.attn_drop(attn1)

        attn2 = torch.matmul(q2, k2) / np.sqrt(self.d_k)
        attn2 = torch.softmax(attn2, -1)
        attn2 = self.attn_drop(attn2)

        attn1_3 = torch.matmul(q1, k3) / np.sqrt(self.d_k)
        attn1_3 = torch.softmax(attn1_3, -1)
        attn1_3 = self.attn_drop(attn1_3)

        attn2_3 = torch.matmul(q2, k3) / np.sqrt(self.d_k)
        attn2_3 = torch.softmax(attn2_3, -1)
        attn2_3 = self.attn_drop(attn2_3)

        out1_3 = torch.matmul(attn1_3, v3).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                 self.h * self.d_v)  # (b_s, nq, h*d_v)
        out2_3 = torch.matmul(attn2_3, v3).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                                 self.h * self.d_v)  # (b_s, nq, h*d_v)
        # output
        model1 = torch.matmul(attn1, v3).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                               self.h * self.d_v)

        model2 = torch.matmul(attn2, v3).permute(0, 2, 1, 3).contiguous().view(b_s, nq,
                                                                               self.h * self.d_v)

        model3 = torch.cat([model1 + out1_3, model3, model2 + out2_3], dim=2)
        # x2 = torch.cat([out3, model1, out2_3], dim=2).permute(0, 2, 1)
        # model3 = self.reduce2(model3).permute(0, 2, 1)
        # out2 = self.reduce3(x2).permute(0, 2, 1)
        model3 = self.out_proj(model3)  # (b_s, nq, d_model)
        # model3 = self.resid_drop(self.out_proj1(model3))  # (b_s, nq, d_model)

        return model1, model2, model3


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input1 = nn.LayerNorm(d_model)
        self.ln_output1 = nn.LayerNorm(d_model)
        self.ln_input2 = nn.LayerNorm(d_model)
        self.ln_input3 = nn.LayerNorm(d_model)
        self.ln_output2 = nn.LayerNorm(d_model)
        self.ln_output3 = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h)

        self.mlp1 = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            QuickGELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            QuickGELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            QuickGELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, model1, model2, model3):
        bs, nx, c = model1.size()
        x_fuse = self.sa(self.ln_input1(model1), self.ln_input2(model2), self.ln_input3(model3))
        x1 = model1 + x_fuse[0]
        x2 = model2 + x_fuse[1]
        x3 = model3 +x_fuse[2]
        x1 = x1 + self.mlp1(self.ln_output1(x1))
        x2 = x2 + self.mlp2(self.ln_output2(x2))
        x3 = x3 + self.mlp3(self.ln_output3(x3))
        return x1, x2, x3


class sim(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self):
        super(sim, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)

    def forward(self, model1, model2):
        batch_size, nq, c = model1.shape
        rgb_features = model1.reshape(batch_size, -1)
        ir_features = model2.reshape(batch_size, -1)
        # normalized features
        image_features = rgb_features / rgb_features.norm(dim=1, keepdim=True)
        text_features = ir_features / ir_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_ir = logits_per_image.t()
        adjusted_ir_images = []
        similarity_scores = torch.diag(logits_per_image)
        similarity_scores1 = torch.diag(logits_per_ir)  # Extract the diagonal elements as similarity scores
        adjusted_rgb_images = []

        for i, (rgb_img, ir_img) in enumerate(zip(model1, model2)):
            score = similarity_scores[i]
            adjusted_img = rgb_img * (1 + score)
            adjusted_rgb_images.append(adjusted_img)
            score1 = similarity_scores1[i]
            # Adjust the intensity of the IR image based on the similarity score
            adjusted_ir_img = ir_img * (1 + score1)
            adjusted_ir_images.append(adjusted_ir_img)
        return torch.stack(adjusted_rgb_images), torch.stack(adjusted_ir_images)


class MyTransformer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, n_layer):
        super().__init__()
        self.layers = nn.ModuleList([myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                     for _ in range(n_layer)])
        self.reduce = nn.Linear(2 * d_model, d_model)
        self.sim = sim()
        self._init_weights()

    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x1, x2):
        x3 = self.reduce(torch.cat([x1,x2],dim=2))
        for layer in self.layers:
            x1, x2, x3 = layer(x1, x2, x3)

        #x1, x2 = self.sim(x1, x2)

        return x1, x2, x3


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # transformer
        self.trans_blocks = MyTransformer(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, n_layer)

        # decoder head
        self.ln_f1 = nn.LayerNorm(self.n_embd)
        self.ln_f2 = nn.LayerNorm(self.n_embd)
        self.ln_f3 = nn.LayerNorm(self.n_embd)
        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        x = self.trans_blocks(rgb_fea_flat, ir_fea_flat)  # dim:(B, 2*H*W, C)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # decoder head
        x1 = self.ln_f1(x1)  # dim:(B, 2*H*W, C)
        x1 = x1.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x1 = x1.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        x2 = self.ln_f2(x2)  # dim:(B, 2*H*W, C)
        x2 = x2.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x2 = x2.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x1.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x2.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        x3 = self.ln_f3(x3)  # dim:(B, 2*H*W, C)
        x3 = x3.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x3 = x3.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        x3 = x3.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        x3 = F.interpolate(x3, size=([h, w]), mode='bilinear')
        return rgb_fea_out, ir_fea_out, x3


class GPT_low(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = MyTransformer(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, n_layer)

        # decoder head
        self.ln_f1 = nn.LayerNorm(self.n_embd)
        self.ln_f2 = nn.LayerNorm(self.n_embd)
        self.ln_f3 = nn.LayerNorm(self.n_embd)
        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
       Args:
           x (tuple?)

       """

        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        x = self.trans_blocks(rgb_fea_flat, ir_fea_flat)  # dim:(B, 2*H*W, C)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # decoder head
        x1 = self.ln_f1(x1)  # dim:(B, 2*H*W, C)
        x1 = x1.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x1 = x1.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        x2 = self.ln_f2(x2)  # dim:(B, 2*H*W, C)
        x2 = x2.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x2 = x2.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x1.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x2.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        x3 = self.ln_f3(x3)  # dim:(B, 2*H*W, C)
        x3 = x3.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x3 = x3.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        x3 = x3.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        x3 = F.interpolate(x3, size=([h, w]), mode='bilinear')
        return rgb_fea_out, ir_fea_out, x3


class GPT_middle(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=10, vert_anchors=6, horz_anchors=6,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = MyTransformer(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, n_layer)

        # decoder head
        self.ln_f1 = nn.LayerNorm(self.n_embd)
        self.ln_f2 = nn.LayerNorm(self.n_embd)
        self.ln_f3 = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
      Args:
          x (tuple?)

      """

        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        x = self.trans_blocks(rgb_fea_flat, ir_fea_flat)  # dim:(B, 2*H*W, C)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # decoder head
        x1 = self.ln_f1(x1)  # dim:(B, 2*H*W, C)
        x1 = x1.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x1 = x1.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        x2 = self.ln_f2(x2)  # dim:(B, 2*H*W, C)
        x2 = x2.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x2 = x2.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x1.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x2.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        x3 = self.ln_f3(x3)  # dim:(B, 2*H*W, C)
        x3 = x3.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x3 = x3.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        x3 = x3.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        x3 = F.interpolate(x3, size=([h, w]), mode='bilinear')
        return rgb_fea_out, ir_fea_out, x3


class GPT_high(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=6, vert_anchors=2, horz_anchors=2,
                 embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = MyTransformer(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, n_layer)

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1).permute(0, 2, 1).contiguous()  # flatten the feature
        x = self.trans_blocks(rgb_fea_flat, ir_fea_flat)  # dim:(B, 2*H*W, C)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        # decoder head
        x1 = self.ln_f(x1)  # dim:(B, 2*H*W, C)
        x1 = x1.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x1 = x1.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        x2 = self.ln_f(x2)  # dim:(B, 2*H*W, C)
        x2 = x2.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x2 = x2.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x1.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x2.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        x3 = self.ln_f(x3)  # dim:(B, 2*H*W, C)
        x3 = x3.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        x3 = x3.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        x3 = x3.contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        x3 = F.interpolate(x3, size=([h, w]), mode='bilinear')
        return rgb_fea_out, ir_fea_out, x3


class NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out


class DAnet(nn.Module):
    def __init__(self, in_channels):
        self.decoder = DAHead(in_channels)

    def forward(self, x):
        feats = self.ResNet50(x)
        # self.ResNet50返回的是一个字典类型的数据.
        x = self.decoder(feats["stage4"])
        return x


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h * w).transpose(1, 2), C.view(b, c, h * w)))
        E = torch.matmul(D.view(b, c, h * w), S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        E = self.gamma * E + x

        return E


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x
        return X


class DAHead(nn.Module):
    def __init__(self, in_channels):
        super(DAHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=3, padding=1, bias=False),
        )

        self.PositionAttention = PositionAttention(in_channels // 4)
        self.ChannelAttention = ChannelAttention()

    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)
        PosionAttentionMap = self.PositionAttention(x_PA)
        ChannelAttentionMap = self.ChannelAttention(x_CA)
        # 这里可以额外分别做PAM和CAM的卷积输出,分别对两个分支做一个上采样和预测,
        # 可以生成一个cam loss和pam loss以及最终融合后的结果的loss.以及做一些可视化工作
        # 这里只输出了最终的融合结果.与原文有一些出入.
        output = self.conv3(PosionAttentionMap + ChannelAttentionMap)
        output = nn.functional.interpolate(output, scale_factor=8, mode="bilinear", align_corners=True)
        output = self.conv4(output)
        return output


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).float()
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).float()

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h).float()
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w).float()

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_H.half() + out_W.half()) + x.half()


class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=2048):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )

    def forward(self, x):
        # reduce channels from C to C'   2048->512
        output = self.conv_in(x)

        for i in range(self.recurrence):
            output = self.CCA(output)

        output = self.conv_out(output)
        return output


class AFNPBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, pool_sizes=[1, 3, 6, 8]):
        super(AFNPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # query 接受的是stage5的Xh 所以这里的是in_channels=2048
        self.Conv_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        # key 和 value 接受的是stage4的输出Xl 这里的in_channels//2为1024
        self.Conv_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.Conv_value = nn.Conv2d(self.in_channels, self.value_channels, 1)

        self.ConvOut = nn.Conv2d(self.value_channels, self.out_channels, 1)
        self.ppm = PPMModule(pool_sizes)
        # 给ConvOut初始化为0
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)

    def forward(self, low_feats, high_feats):
        # low_feats = stage4   high_feats = stage5
        b, c, h, w = high_feats.size()

        # value = [batch, -1, value_channels] // 这里-1由pool_sizes决定，目前的设置为110=1+3*3+6*6+8*8
        value = self.ppm(self.Conv_value(low_feats)).permute(0, 2, 1)
        # batch = [batch, key_channels, -1] // 这里-1由pool_sizes决定，目前的设置为110=1+3*3+6*6+8*8
        key = self.ppm(self.Conv_key(low_feats))
        # query = [batch, key_channels, h*w] -> [batch, h*w, key_channels]
        query = self.Conv_query(high_feats).view(b, self.key_channels, -1).permute(0, 2, 1)

        # Concat_QK = [batch, h*w, 110]
        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *high_feats.size()[2:])
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        return Aggregate_QKV


class PPMModule(nn.ModuleList):
    def __init__(self, pool_sizes=[1, 3, 6, 8]):
        super(PPMModule, self).__init__()
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size)
                )
            )

    def forward(self, x):
        out = []
        b, c, _, _ = x.size()
        for index, module in enumerate(self):
            out.append(module(x))
        # 最后输出时将其合并
        return torch.cat([output.view(b, c, -1) for output in out], -1)


class APNBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, value_channels, pool_sizes=[1, 3, 6, 8]):
        super(APNBBlock, self).__init__()

        # Generally speaking, here, in_channels==out_channels and key_channels==value_channles
        self.in_channels = in_channels
        self.out_channles = out_channels
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.pool_sizes = pool_sizes

        self.Conv_Key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        # 这里Conv_Query 和 Conv_Key权重共享，也就是计算出来的query和key是等同的
        self.Conv_Query = self.Conv_Key

        self.Conv_Value = nn.Conv2d(self.in_channels, self.key_channels, 1)
        self.Conv_Out = nn.Conv2d(self.value_channels, self.out_channles, 1)
        nn.init.constant_(self.Conv_Out.weight, 0)
        nn.init.constant_(self.Conv_Out.bias, 0)
        self.ppm = PPMModule(pool_sizes=self.pool_sizes)

    def forward(self, x):
        b, _, h, w = x.size()

        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        value = self.ppm(self.Conv_Value(x)).permute(0, 2, 1)
        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        query = self.Conv_Query(x).view(b, self.key_channels, -1).permute(0, 2, 1)
        # key = [batch, key_channels, 110]  where 110 = sum([s*2 for s in pool_sizes]) 1 + 3*2 + 6*2 + 8*2
        key = self.ppm(self.Conv_Key(x))

        # Concat_QK = [batch, h*w, 110]
        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *x.size()[2:])
        # Conv out
        Aggregate_QKV = self.Conv_Out(Aggregate_QKV)

        return Aggregate_QKV


class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, in_channels):
        super(asymmetric_non_local_network, self).__init__()
        # AFNB and APNB
        self.fusion = AFNPBlock(in_channels, value_channels=256, key_channels=256, pool_sizes=[1, 3, 6, 8])

        self.APNB = APNBBlock(in_channels, in_channels, value_channels=256, key_channels=256,
                              pool_sizes=[1, 3, 6, 8])
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            self.APNB
        )
        self.cls = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        if self.aux_loss:
            return aux_x, x
        return x


class Conv_DANet(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_DANet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.DAnet = DAHead(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.DAnet(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))




class Conv_CCNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_CCNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.CCNet = RCCAModule(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.CCNet(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_GCNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_GCNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ContextBlock = GlobalContextBlock(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ContextBlock(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_ANNNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_ANNNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.APNBBlock = APNBBlock(c2, c2, 256, 256)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.APNBBlock(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential

from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch
import torch.nn as nn
from pathlib import Path


# from DETR.ops.modules.ms_deform_attn import MSDeformAttn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3_CCNet(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3_CCNet, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.CCNet = CrissCrossAttention(in_channels=c2)

    def forward(self, x):
        x = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        x = self.CCNet(x)
        return x


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class feature_fusion(nn.Module):
    def __init__(self, inchannel):
        super(feature_fusion, self).__init__()
        self.fusion_1 = Conv(inchannel, inchannel, 1)
        self.fusion_2 = Conv(inchannel, inchannel, 1)
        self.fusion_3 = Conv(inchannel * 2, 2, 1)

    def forward(self, x1, x2):
        fusion = torch.softmax(self.fusion_3(torch.cat([self.fusion_1(x1), self.fusion_1(x2)], dim=1)), dim=1)
        x1_weight, x2_weight = torch.split(fusion, [1, 1], dim=1)
        return x1 * x1_weight + x2 * x2_weight


class Add(nn.Module):
    #  Add two tensorss
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg
        self.SimAM = SimAM()
        self.feature_fusion = feature_fusion(arg)

    #   self.NonLocal = NonLocalBlockND(in_channels=arg)

    def forward(self, x):
        x1 = x[0]
        x2 = x[1]
        # x1 = self.NonLocal(x1)
        # x2 = self.NonLocal(x2)
        x1 = self.SimAM(x1)
        x2 = self.SimAM(x2)
        x = self.feature_fusion(x1, x2)
        return x


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index
        self.SimAM = SimAM()

    def forward(self, x):
        if self.index == 0:
            x1 = self.SimAM(x[0])
            x2 = x[1][0]
            return x1 + x[1][0]
        elif self.index == 1:
            x1 = self.SimAM(x[0])
            x2 = x[1][1]
            return x1 + x[1][1]


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)




class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale=16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels // scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out


class DAnet(nn.Module):
    def __init__(self, in_channels):
        self.decoder = DAHead(in_channels)

    def forward(self, x):
        feats = self.ResNet50(x)
        # self.ResNet50返回的是一个字典类型的数据.
        x = self.decoder(feats["stage4"])
        return x


class PositionAttention(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttention, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        # 创建一个可学习参数a作为权重,并初始化为0.
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        B = self.convB(x)
        C = self.convB(x)
        D = self.convB(x)
        S = self.softmax(torch.matmul(B.view(b, c, h * w).transpose(1, 2), C.view(b, c, h * w)))
        E = torch.matmul(D.view(b, c, h * w), S.transpose(1, 2)).view(b, c, h, w)
        # gamma is a parameter which can be training and iter
        E = self.gamma * E + x

        return E


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(0.)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()
        X = self.softmax(torch.matmul(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2)))
        X = torch.matmul(X.transpose(1, 2), x.view(b, c, h * w)).view(b, c, h, w)
        X = self.beta * X + x
        return X


class DAHead(nn.Module):
    def __init__(self, in_channels):
        super(DAHead, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=3, padding=1, bias=False),
        )

        self.PositionAttention = PositionAttention(in_channels // 4)
        self.ChannelAttention = ChannelAttention()

    def forward(self, x):
        x_PA = self.conv1(x)
        x_CA = self.conv2(x)
        PosionAttentionMap = self.PositionAttention(x_PA)
        ChannelAttentionMap = self.ChannelAttention(x_CA)
        # 这里可以额外分别做PAM和CAM的卷积输出,分别对两个分支做一个上采样和预测,
        # 可以生成一个cam loss和pam loss以及最终融合后的结果的loss.以及做一些可视化工作
        # 这里只输出了最终的融合结果.与原文有一些出入.
        output = self.conv3(PosionAttentionMap + ChannelAttentionMap)
        output = nn.functional.interpolate(output, scale_factor=8, mode="bilinear", align_corners=True)
        output = self.conv4(output)
        return output


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.channels = in_channels // 8
        self.ConvQuery = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvKey = nn.Conv2d(self.in_channels, self.channels, kernel_size=1)
        self.ConvValue = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.SoftMax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.size()

        # [b, c', h, w]
        query = self.ConvQuery(x)
        # [b, w, c', h] -> [b*w, c', h] -> [b*w, h, c']
        query_H = query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
        # [b, h, c', w] -> [b*h, c', w] -> [b*h, w, c']
        query_W = query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)

        # [b, c', h, w]
        key = self.ConvKey(x)
        # [b, w, c', h] -> [b*w, c', h]
        key_H = key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
        # [b, h, c', w] -> [b*h, c', w]
        key_W = key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)

        # [b, c, h, w]
        value = self.ConvValue(x)
        # [b, w, c, h] -> [b*w, c, h]
        value_H = value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).float()
        # [b, h, c, w] -> [b*h, c, w]
        value_W = value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).float()

        # [b*w, h, c']* [b*w, c', h] -> [b*w, h, h] -> [b, h, w, h]
        energy_H = (torch.bmm(query_H, key_H) + self.INF(b, h, w)).view(b, w, h, h).permute(0, 2, 1, 3)
        # [b*h, w, c']*[b*h, c', w] -> [b*h, w, w] -> [b, h, w, w]
        energy_W = torch.bmm(query_W, key_W).view(b, h, w, w)
        # [b, h, w, h+w]  concate channels in axis=3

        concate = self.SoftMax(torch.cat([energy_H, energy_W], 3))
        # [b, h, w, h] -> [b, w, h, h] -> [b*w, h, h]
        attention_H = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h).float()
        attention_W = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w).float()

        # [b*w, h, c]*[b*w, h, h] -> [b, w, c, h]
        out_H = torch.bmm(value_H, attention_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_W = torch.bmm(value_W, attention_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_H.half() + out_W.half()) + x.half()


class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=2048):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )

    def forward(self, x):
        # reduce channels from C to C'   2048->512
        output = self.conv_in(x)

        for i in range(self.recurrence):
            output = self.CCA(output)

        output = self.conv_out(output)
        return output


class AFNPBlock(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, pool_sizes=[1, 3, 6, 8]):
        super(AFNPBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        # query 接受的是stage5的Xh 所以这里的是in_channels=2048
        self.Conv_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        # key 和 value 接受的是stage4的输出Xl 这里的in_channels//2为1024
        self.Conv_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        self.Conv_value = nn.Conv2d(self.in_channels, self.value_channels, 1)

        self.ConvOut = nn.Conv2d(self.value_channels, self.out_channels, 1)
        self.ppm = PPMModule(pool_sizes)
        # 给ConvOut初始化为0
        nn.init.constant_(self.ConvOut.weight, 0)
        nn.init.constant_(self.ConvOut.bias, 0)

    def forward(self, low_feats, high_feats):
        # low_feats = stage4   high_feats = stage5
        b, c, h, w = high_feats.size()

        # value = [batch, -1, value_channels] // 这里-1由pool_sizes决定，目前的设置为110=1+3*3+6*6+8*8
        value = self.ppm(self.Conv_value(low_feats)).permute(0, 2, 1)
        # batch = [batch, key_channels, -1] // 这里-1由pool_sizes决定，目前的设置为110=1+3*3+6*6+8*8
        key = self.ppm(self.Conv_key(low_feats))
        # query = [batch, key_channels, h*w] -> [batch, h*w, key_channels]
        query = self.Conv_query(high_feats).view(b, self.key_channels, -1).permute(0, 2, 1)

        # Concat_QK = [batch, h*w, 110]
        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *high_feats.size()[2:])
        # Conv out
        Aggregate_QKV = self.ConvOut(Aggregate_QKV)

        return Aggregate_QKV


class PPMModule(nn.ModuleList):
    def __init__(self, pool_sizes=[1, 3, 6, 8]):
        super(PPMModule, self).__init__()
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size)
                )
            )

    def forward(self, x):
        out = []
        b, c, _, _ = x.size()
        for index, module in enumerate(self):
            out.append(module(x))
        # 最后输出时将其合并
        return torch.cat([output.view(b, c, -1) for output in out], -1)


class APNBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels, value_channels, pool_sizes=[1, 3, 6, 8]):
        super(APNBBlock, self).__init__()

        # Generally speaking, here, in_channels==out_channels and key_channels==value_channles
        self.in_channels = in_channels
        self.out_channles = out_channels
        self.value_channels = value_channels
        self.key_channels = key_channels
        self.pool_sizes = pool_sizes

        self.Conv_Key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU()
        )
        # 这里Conv_Query 和 Conv_Key权重共享，也就是计算出来的query和key是等同的
        self.Conv_Query = self.Conv_Key

        self.Conv_Value = nn.Conv2d(self.in_channels, self.key_channels, 1)
        self.Conv_Out = nn.Conv2d(self.value_channels, self.out_channles, 1)
        nn.init.constant_(self.Conv_Out.weight, 0)
        nn.init.constant_(self.Conv_Out.bias, 0)
        self.ppm = PPMModule(pool_sizes=self.pool_sizes)

    def forward(self, x):
        b, _, h, w = x.size()

        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        value = self.ppm(self.Conv_Value(x)).permute(0, 2, 1)
        # query = [batch, key_channels, -1 -> h*w] -> [batch, h*w, key_channels]
        query = self.Conv_Query(x).view(b, self.key_channels, -1).permute(0, 2, 1)
        # key = [batch, key_channels, 110]  where 110 = sum([s*2 for s in pool_sizes]) 1 + 3*2 + 6*2 + 8*2
        key = self.ppm(self.Conv_Key(x))

        # Concat_QK = [batch, h*w, 110]
        Concat_QK = torch.matmul(query, key)
        Concat_QK = (self.key_channels ** -.5) * Concat_QK
        Concat_QK = F.softmax(Concat_QK, dim=-1)

        # Aggregate_QKV = [batch, h*w, Value_channels]
        Aggregate_QKV = torch.matmul(Concat_QK, value)
        # Aggregate_QKV = [batch, value_channels, h*w]
        Aggregate_QKV = Aggregate_QKV.permute(0, 2, 1).contiguous()
        # Aggregate_QKV = [batch, value_channels, h*w] -> [batch, value_channels, h, w]
        Aggregate_QKV = Aggregate_QKV.view(b, self.value_channels, *x.size()[2:])
        # Conv out
        Aggregate_QKV = self.Conv_Out(Aggregate_QKV)

        return Aggregate_QKV


class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, in_channels):
        super(asymmetric_non_local_network, self).__init__()
        # AFNB and APNB
        self.fusion = AFNPBlock(in_channels, value_channels=256, key_channels=256, pool_sizes=[1, 3, 6, 8])

        self.APNB = APNBBlock(in_channels, in_channels, value_channels=256, key_channels=256,
                              pool_sizes=[1, 3, 6, 8])
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            self.APNB
        )
        self.cls = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        if self.aux_loss:
            return aux_x, x
        return x


class Conv_DANet(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_DANet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.DAnet = DAHead(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.DAnet(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))





class Conv_CCNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_CCNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.CCNet = RCCAModule(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.CCNet(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_GCNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_GCNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ContextBlock = GlobalContextBlock(in_channels=c2)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ContextBlock(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv_ANNNet(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_ANNNet, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.APNBBlock = APNBBlock(c2, c2, 256, 256)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.APNBBlock(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
