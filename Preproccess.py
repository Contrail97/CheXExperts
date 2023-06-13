""" Preproccessing CXR14 Datasets for RetinaNet
    ChristianNeil
    2022/7/22
"""

import os
import csv
import sys
import random
from tqdm import tqdm

sys.path.append("./retinanet")
from retinanet.chest_detection import detect_chest, get_model


CXR14_IMG_PATH = "D:\\dataset\\CXR14\\images\\"
CXR14_COP_PATH = "D:\\dataset\\CXR14\\images\\croped\\"
CXR14_SEG_PATH = "D:\\dataset\\CXR14\\segmentations\\"
CXR14_BOX_PATH = "D:\\dataset\\CXR14\\BBox_List_2017.csv"
CXR14_ANO_PATH = "D:\\dataset\\CXR14\\Data_Entry_2017_v2020.csv"

CXR14_TRAIN_PATH = "D:\\projects\\儿童医院\\参考代码\\CheXNet-master\\dataset\\train_1.txt"
CXR14_VAL_PATH = "D:\\projects\\儿童医院\\参考代码\\CheXNet-master\\dataset\\val_1.txt"
CXR14_TEST_PATH = "D:\\projects\\儿童医院\\参考代码\\CheXNet-master\\dataset\\test_1.txt"
CSV_OUTPUT_PATH = "./"

RETINANET_MODEL_PATH = "D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\retinanet\\models\\trained_without_neg_sample_res101\\csv_retinanet_epoch3.pt"

file_list = [CXR14_TRAIN_PATH,
             CXR14_VAL_PATH,
             CXR14_TEST_PATH]


def create_dataset(classes, path, annos_train, annos_test, annos_val):
    """ Generate CSV file for RetinaNet training
    """
    with open(path + "train.csv", mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(annos_train)

    if annos_val is not None:
        with open(path + "val.csv", mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(annos_val)

    if annos_test is not None:
        with open(path + "test.csv", mode="w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(annos_test)


def preproc_cxr14_dataset():
    """ preproccessing CXR14 dataset to CSV file
    """
    annos = []
    annos_val = []
    annos_test = []
    dataset = [annos_val, annos_test, annos]
    classes = [
        ["Atelectasis", 0],
        ["Cardiomegaly", 1],
        ["Effusion", 2],
        ["Infiltration", 3],
        ["Mass", 4],
        ["Nodule", 5],
        ["Pneumonia", 6],
        ["Pneumothorax", 7],
        ["Consolidation", 8],
        ["Edema", 9],
        ["Emphysema", 10],
        ["Fibrosis", 11],
        ["Pleural_Thickening", 12],
        ["Hernia", 13],
        ["No Finding",  14]
    ]
    class_counter = {}
    class_map = {}

    model = get_model(RETINANET_MODEL_PATH)

    for c in classes:
        class_counter[c[0]] = 0
        class_map[c[0]] = c[1]

    with open(CXR14_ANO_PATH) as f:

        f_csv = csv.reader(f)
        next(f_csv)
        items = [x for x in csv.reader(f)]

        for row in tqdm(items):
            row = tuple(str(val) for val in row)

            img_path = CXR14_IMG_PATH + row[0]
            labels = row[1].split('|')

            #Divide datasets by 7 :2 :1
            prob = random.randint(0, 9)
            if prob == 0:
                index = 0
            elif prob < 3:
                index = 1
            else:
                index = 2

            result = detect_chest(img_path, model, 0.999)

            if result:
                chest_roi = result[1]
                one_hot = [0 for x in range(len(classes) - 1)]
                for label in labels:
                    if label != "No Finding":
                        one_hot[class_map[label]] = 1
                    target = [row[0]]
                    target.extend(chest_roi)
                    target.extend(one_hot)
                    dataset[index].append(target)
                    class_counter[label] += 1

        print(class_counter)
    create_dataset(classes, CSV_OUTPUT_PATH, annos, annos_test, annos_val)


def preproc_cxr14_anno():

    for path in file_list:

        tmp = []
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            for line in f:
                tmp.append(line[11:])

        with open(CSV_OUTPUT_PATH + path[45:], mode="w", encoding="utf-8", newline="") as f:
            f.writelines(tmp)


def val_clean():
    listsave = []

    # ---- Open file, get image paths and labels
    fileDescriptor = open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\test_1.txt", "r")
    fileDescriptor2 = open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\all.csv", "r")

    dictCsv = {}
    line = True
    while line:
        line = fileDescriptor2.readline()
        # --- if not empty
        if line:
            lineItems = line.split(',')
            dictCsv[lineItems[0]] = line

    line = True
    while line:
        line = fileDescriptor.readline()
        # --- if not empty
        if line:
            lineItems = line.split(' ')
            if lineItems[0] in dictCsv:
                listsave.append(dictCsv[lineItems[0]])

    with open("D:\\projects\\儿童医院\\参考代码\\ChexNet\\CheXNet++\\dataset\\test.csv", "w") as f:
        for item in listsave:
            f.write(item)

    fileDescriptor.close()


if __name__ == '__main__':
    preproc_cxr14_dataset()
    # val_clean()
