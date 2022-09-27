'''
python inference.py --ill_rec --save_imgs --path_data '../data/result_control_points_test_6/' --path_result '../data/result_control_points_test_6_doctr/'
'''
import sys
sys.path.append("doctr/")
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
from inference_ill import rec_ill
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
from pathlib import Path
import cv2
import glob
import os
import pdb
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x
        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        return bm
        

def reload_model(model, path=''):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model
        

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


def get_models(args):
    GeoTr_Seg_model = GeoTr_Seg().cuda()
    reload_segmodel(GeoTr_Seg_model.msk, args.Seg_path)
    reload_model(GeoTr_Seg_model.GeoTr, args.GeoTr_path)
    IllTr_model = IllTr().cuda()
    reload_model(IllTr_model, args.IllTr_path)    
    GeoTr_Seg_model.eval()
    IllTr_model.eval()
    return GeoTr_Seg_model, IllTr_model


def run_model(img, img_path, model_geo, model_ill, args):
    img = img[:,:,::-1]
    img = img / 255.
    h, w, _ = img.shape
    im = cv2.resize(img, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)
    

    with torch.no_grad():
        res_1, res_2 = None, None
        bm = model_geo(im.cuda())
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
        
        out = F.grid_sample(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
        res_1 = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)

        if args.save_imgs:
            cv2.imwrite(args.path_result + Path(img_path).stem + '_geo' + '.png', res_1)  # save
        
        if args.ill_rec:
            res_2 = rec_ill(model_ill, res_1)
            if args.save_imgs:
                cv2.imwrite(args.path_result + Path(img_path).stem + '_ill' + '.png', res_2)
    
    print('done: ', img_path)
    return res_1, res_2


def iterate_images(model_geo, model_ill, args):
    img_list = glob.glob(args.path_data + '*.png')
    img_list.extend(glob.glob(args.path_data + '*.jpg'))
    if not os.path.exists(args.path_result):
        os.mkdir(args.path_result)

    for img_path in img_list:
        img = cv2.imread(img_path)
        img_1, img_2 = run_model(img, img_path, model_geo, model_ill, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--path_data',  default='../data/test_1/')
    parser.add_argument('--path_result',  default='../data/result_doctr_test_1/')
    parser.add_argument('--Seg_path',  default='models/seg.pth')
    parser.add_argument('--GeoTr_path',  default='models/geotr.pth')
    parser.add_argument('--IllTr_path',  default='models/illtr.pth')
    parser.add_argument('--ill_rec', default=False, action="store_true")
    parser.add_argument('--save_imgs', default=False, action="store_true")
    args = parser.parse_args()

    model_geo, model_ill = get_models(args)
    iterate_images(model_geo, model_ill, args)
