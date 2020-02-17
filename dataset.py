import _pickle as cPickle
import os
from torch.utils.data import Dataset
import cv2
import numpy as np

class LoadDataset(Dataset):
    #참고 https://wingnim.tistory.com/33
    def __init__(self, root_dir, transform=None): #download, read data 등등을 하는 파트
        self.root_dir = root_dir
        self.transform = transform

        watermarked_data = []
        original_data = []

        path_of_wimages = os.path.join(root_dir, 'watermarked')
        path_of_oimages = os.path.join(root_dir, 'original')

        list_of_images = os.listdir(path_of_wimages)
        for image in list_of_images:
            img = cv2.imread(os.path.join(path_of_wimages, image), 0)
            watermarked_data.append(img)
            
        list_of_images = os.listdir(path_of_oimages)
        for image in list_of_images:
            img = cv2.imread(os.path.join(path_of_oimages, image), 0)
            original_data.append(img)

        watermarked_data = np.array(watermarked_data)
        original_data = np.array(original_data)
        self.len = watermarked_data.shape[0]
        self.watermarked_data=watermarked_data
        self.original_data=original_data

    def __len__(self): #data size를 넘겨주는 파트
        return self.len

    def __getitem__(self, index): #인덱스에 해당하는 아이템을 넘겨주는 파트.
        # Get the sample, and apply any necessary transform (if any).
        sample_watermark = self.watermarked_data[index]
        sample_original = self.original_data[index]

        if self.transform:
             sample_watermark = self.transform(sample_watermark)
             sample_original = self.transform(sample_original)

        return sample_watermark, sample_original