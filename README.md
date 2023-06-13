# CheXExperts implementation in PyTorch

Multi expert fusion disease diagnosis model CheXExperts achieved an AUC score of 0.85 and an IoR score of 0.75 in the CXR14 dataset.

## Prerequisites
* Python 3.7
* matplotlib==3.4.3
* multimethod==1.8
* numpy==1.20.3
* opencv_python==4.5.5.64
* Pillow==9.5.0
* pycocotools==2.0.4
* pyswarms==1.3.0
* PyYAML==6.0
* scikit_image==0.19.3
* scikit_learn==0.24.2
* scipy==1.7.1
* torch==1.10.2
* torchvision==0.11.3
* tqdm==4.62.3
* ttach==0.0.3

## Preparation
* Download the ChestX-ray14 database from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)
* Unpack archives in separate directories (e.g. images_001.tar.gz into images_001)
* 
* Download the trained models and cropped chest X-ray images [here]([https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737](https://drive.google.com/drive/folders/1sq13RkeoeRE8n7uibaEMDncz9wG8JZ4u))
* Unpack segmentations.tar.gz to the same level directory as ChestX-ray14 database.
* Move the best_auc_model26-0.8419458151267506.pth.tar to the CheXExperts/checkpoints/withGAA folder.
* Move the best_auc_model29-0.8544614168758137.pth.tar to the CheXExperts/checkpoints/withoutGAA folder.
* Move the csv_retinanet_epoch3.pt to the CheXExperts/retinanet/models/trained_without_neg_sample_res101 folder.

## Usage
*
* For verifying CheXExpert:
*   Open the CheXExperts\cfgs\chexnet++.yaml and edit the following fields to your own dataset directory:
*       images_path: D:\dataset\CXR14\images
*       segment_path: D:\dataset\CXR14\segmentations
*   Run **python Main.py** to run verifying.
*
*
* For Training CheXExperts:
*   1.Training The CheXMHNet without GAA.
*     Copy the CheXExperts\checkpoints\withoutGAA\chexnet++.yaml to CheXExperts\cfgs\ , backup the original chexnet++.yaml file.
*     Open the chexnet++.yaml and edit the following fields to your own dataset directory:
*       images_path: D:\dataset\CXR14\images
*       segment_path: D:\dataset\CXR14\segmentations
*     Run **python Main.py** to run Training
*   2.Training The CheXMHNet with GAA.
*     Copy the CheXExperts\checkpoints\withGAA\chexnet++.yaml to CheXExperts\cfgs\ , backup the original chexnet++.yaml file.
*     Open the chexnet++.yaml and edit the following fields to your own dataset directory:
*       images_path: D:\dataset\CXR14\images
*       segment_path: D:\dataset\CXR14\segmentations
*     Run **python Main.py** to run Training


## Results


## Experimental environment
GPU RTX 3060 12GB
CPU I5 10400F
Mem 16GB

