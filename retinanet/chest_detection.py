import torch
import numpy as np
import os
import cv2
import tqdm
import argparse
import random


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_chest(image_path, model, threshold=0.999, expand_x=0.10, expand_y=0.10):

    img = cv2.imread(image_path)

    if len(img.shape) > 2:
        if (img.shape[2] > 3):
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = np.expand_dims(img, 2)

    img.astype(np.float32) / 255.0

    image_orig = img.copy()

    rows, cols = img.shape[0], img.shape[1]

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(img, (int(round(cols * scale)), int(round((rows * scale)))))
    image = np.expand_dims(image, 2)
    rows, cols, chls = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, chls)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485]
    image /= [0.229]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        scores, classification, transformed_anchors = model(image.cuda().float())
        idxs = np.where(scores.cpu() >= threshold)
        if len(idxs[0]) == 0:
            return None

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)

            width, height = x2 - x1, y2 - y1
            x1 -= width*expand_x
            x2 += width*expand_x
            y1 -= height*expand_y
            y2 += height*expand_y

            x1 = 0 if x1 < 0 else int(x1)
            y1 = 0 if y1 < 0 else int(y1)
            x2 = image_orig.shape[1] if x2 > image_orig.shape[1] else int(x2)
            y2 = image_orig.shape[0] if y2 > image_orig.shape[0] else int(y2)

    cropped_image = image_orig[y1:y2, x1:x2]
    return cropped_image, [x1, x2, y1, y2]


def get_model(model_path):

    retinaNet = torch.load(model_path)
    if torch.cuda.is_available():
        retinaNet = retinaNet.cuda()
    retinaNet.training = False
    retinaNet.eval()
    return retinaNet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_path', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--corped_save_path', help='Path to save corped image file')
    parser.add_argument('--threshold', help='threshold to filter ROI', default=0.999)

    parser = parser.parse_args()

    all_file = []
    for f in os.listdir(parser.image_path):  # listdir返回文件中所有目录
        f_name = os.path.join(parser.image_path, f)
        all_file.append(f_name)

    model = get_model(parser.model_path)

    for item in tqdm.tqdm(all_file):
        res = detect_chest(item, model, parser.threshold)

        if res:
            # write img to disc for inspect with 1% probability
            write_path = os.path.join(parser.corped_save_path + os.path.basename(item))
            cv2.imwrite(write_path, res[0])
            print(write_path)

