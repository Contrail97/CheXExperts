""" Preproccessing CXR14 Datasets for RetinaNet
    ChristianNeil
    2022/7/22
"""

import os
import cv2
import csv
import random
import numpy as np
from tqdm import tqdm

CXR14_IMG_PATH = "D:\\dataset\\CXR14\\images\\"
CXR14_SEG_PATH = "D:\\dataset\\CXR14\\segmentations\\"
CSV_PATH = "./"


def cal_rect_ovlap_area(a_x, a_y, a_w, a_h, b_x, b_y, b_w, b_h):
    """ calculate the overlapping area between rectangle a and b
        return negative number if these is no overlapping.
    """
    max_width  = max(a_x + a_w, b_x + b_w) - min(a_x, b_x)
    max_height = max(a_y + a_h, b_y + b_h) - min(a_y, b_y)

    if max_width < (a_w + b_w) and max_height < (a_h + b_h):
        return ((a_w + b_w) - max_width) * ((a_h + b_h) - max_height)
    elif max_width > (a_w + b_w):
        return (a_w + b_w) - max_width
    elif max_height > (a_h + b_h):
        return (a_h + b_h) - max_height
    else:
        return -1


def gen_ran_neg_sample(b_x, b_y, b_w, b_h, max_w, max_h):
    """ Generate random negative sample
    """
    t_x = 0
    t_y = 0
    t_w = random.randint(int(max_w/5), int(max_w*4/5))
    t_h = random.randint(int(max_h/5), int(max_h*4/5))

    min = max_w * max_h
    for i in range(0, 100):
        x = random.randint(0, max_w - t_w)
        y = random.randint(0, max_h - t_h)
        inter = cal_rect_ovlap_area(x, y, t_w, t_h, b_x, b_y, b_w, b_h)
        if inter < min:
            min = inter
            t_x = x
            t_y = y
    return t_x, t_y, t_w, t_h


def create_csv(annos, classes, path):
    """ Generate CSV file for RetinaNet training
    """
    with open(path + "annotations.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annos)

    with open(path + "classes.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(classes)


def preproc_cxr14():
    """ preproccessing CXR14 dataset to CSV file
    """
    filenames = list(set([x for x in os.listdir(CXR14_SEG_PATH)]))
    annos = []
    classes = [
        ["foreground", 0],
        #["background", 1]
    ]

    for f in tqdm(filenames):
        img_path = CXR14_IMG_PATH + f[:12] + '.png'
        seg_path = CXR14_SEG_PATH + f
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)

        # find contours
        thresh = cv2.cvtColor(seg.copy(), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # pick out the largest bounding box
        maxLenght = 0
        target = None
        for c in contours:
            lenght = cv2.arcLength(c, False)
            if(lenght > maxLenght):
                maxLenght = lenght
                target = c

        # find positive bounding box coordinates
        p_x, p_y, p_w, p_h = cv2.boundingRect(target)
        cv2.rectangle(img, (p_x, p_y), (p_x + p_w, p_y + p_h), (255, 0, 0), 1)

        # find negative bounding box coordinates
        #n_x, n_y, n_w, n_h = gen_ran_neg_sample(p_x, p_y, p_w, p_h, img.shape[0], img.shape[1])
        #cv2.rectangle(img, (n_x, n_y), (n_x + n_w, n_y + n_h), (0, 0, 255), 1)

        annos.append([img_path, p_x, p_y, p_x + p_w, p_y + p_h, classes[0][0]])
        #annos.append([img_path, n_x, n_y, n_x + n_w, n_y + n_h, classes[1][0]])

        # cv2.drawContours(seg, contours, -1, (255, 0, 0), 1)
        # cv2.imshow("contours", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    create_csv(annos, classes, CSV_PATH)


def img_check():
    filenames = list(set([x for x in os.listdir(CXR14_IMG_PATH)]))
    for f in tqdm(filenames):
        img = cv2.imread(CXR14_IMG_PATH + f)
        if(img.shape[2] != 3):
            print (img.shape, f)


def img_resize():

    src = cv2.imread("D:\\dataset\\CXR14\\segmentations\\00000001_000_labels.png")
    dst = cv2.resize(src, (128, 128))
    dst = cv2.resize(dst, (256, 256))
    cv2.imshow("ss",dst)
    cv2.waitKey()

if __name__ == '__main__':
    img_resize()
    #img_check()
