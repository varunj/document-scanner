import cv2
import os
import numpy as np
import natsort
import copy
from glob import glob
SUBSET = 'stills'


def add_border(img, target_h, target_w):
    h, w, c = img.shape
    result = np.full((target_h, target_w, c), (0, 0, 0), dtype=np.uint8)
    x_center = (target_w - w) // 2
    y_center = (target_h - h) // 2
    result[y_center:y_center+h, x_center:x_center+w] = img
    return result


def get_images(path_img):
    img_name = os.path.basename(path_img)
    path_res_1 = 'data/result_control_points_%s/mark_%s' % (SUBSET, img_name)
    path_res_2 = 'data/result_control_points_%s/%s' % (SUBSET, img_name)
    img = cv2.imread(path_img)
    h, w = img.shape[0], img.shape[1]
    res_1 = add_border(cv2.imread(path_res_1), h, w)
    res_2 = add_border(cv2.imread(path_res_2), h, w)
    return img, res_1, res_2
    

if __name__ == '__main__':
    img_list = glob('data/%s/*.png' % SUBSET)
    nos_images = len(img_list)
    img = cv2.imread(img_list[0])
    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    big_img = np.zeros((h, w*3, c), dtype=np.uint8)

    fps = 30
    video = cv2.VideoWriter('out_%s.mp4' % SUBSET, cv2.VideoWriter_fourcc(*'MP4V'), fps, (w*3, h))

    for path_img in natsort.natsorted(img_list):
        img, res_1, res_2 = get_images(path_img)
        big_img[:,:w,:] = img
        big_img[:,w:2*w,:] = res_1
        big_img[:,2*w:,:] = res_2
        video.write(big_img)
        print('done: ', path_img)

    cv2.destroyAllWindows()
    video.release()
