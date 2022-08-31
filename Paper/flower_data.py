import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pandas as pd
import cv2
import numpy as np

import flower_list as flower
# import psutil

class Dataset():
    def __init__(self,split_ratio=[0.8,0.2]):
        # self.cur_path = os.getcwd()
        # file_path = os.path.join(self.cur_path, 'data/input/ButterflyClassification/train.csv')
        mypath = "/Paper/flower_photos"
        tra_image, tra_label, test_image, test_label = flower.get_files(mypath)

        ###========================== Split Data =============================###
        image_total=len(tra_image)
        self.train_total=int(image_total*split_ratio[0])
        self.train_image,self.train_label=tra_image[:self.train_total],tra_label[:self.train_total]
        self.val_image,self.val_label=tra_image[self.train_total:],tra_label[self.train_total:]
        self.test_image,self.test_label=test_image,test_label
        self.start=0
    def train(self):
        xs,ys=[],[]
        for x,y in zip(self.train_image,self.train_label):

            ###========================== read image =============================###

            # image_path = os.path.join(self.cur_path, "data\input\ButterflyClassification", x)
            # image_path = os.path.join(DATA_PATH, "ButterflyClassification", x)
            image = cv2.imread(x)
            image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_cvt, (224, 224))
            image_normal = image_resize / (255 / 2.0) - 1
            # image_normal = image_resize / 255.0
            # print(image_normal.shape)
            ###========================== general label =============================###
            label = [0] * 5
            label[y] = 1
            # print(x, label)
            # print(label)
            xs.append(image_normal)
            ys.append(label)
        return xs, ys

    def validation(self):
        xs, ys = [], []
        for x, y in zip(self.val_image[16:32], self.val_label[16:32]):
            ###========================== read image =============================###

            # image_path = os.path.join(self.cur_path, "data\input\ButterflyClassification", x)
            # image_path = os.path.join(DATA_PATH, "ButterflyClassification", x)
            image = cv2.imread(x)
            image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_cvt, (224, 224))
            image_normal = image_resize / (255 / 2.0) - 1
            # image_normal = image_resize / 255.0
            # print(image_normal.shape)
            ###========================== general label =============================###
            label = [0] * 5
            label[y] = 1
            # print(label)
            xs.append(image_normal)
            ys.append(label)
        # return xs, ys
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)

    def test(self):
        xs, ys = [], []
        for x, y in zip(self.test_image, self.test_label):
            ###========================== read image =============================###

            # image_path = os.path.join(self.cur_path, "data/input/ButterflyClassification", x)
            # image_path = os.path.join(DATA_PATH, "ButterflyClassification", x)
            image = cv2.imread(x)
            image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_cvt, (224, 224))
            image_normal = image_resize / (255 / 2.0) - 1
            # image_normal = image_resize / 255.0
            # print(image_normal.shape)
            ###========================== general label =============================###
            label = [0] * 5
            label[y] = 1
            # print(label)
            xs.append(image_normal)
            ys.append(label)
        return xs, ys

    def train_next_batch(self,batch_size):
        xs, ys = [], []
        # print(self.start)
        while True:
            x,y=self.train_image[self.start],self.train_label[self.start]
            ###========================== read image =============================###

            # image_path = os.path.join(self.cur_path, "data/input/ButterflyClassification", x)
            # image_path = os.path.join(DATA_PATH, "ButterflyClassification", x)
            image = cv2.imread(x)
            image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image_cvt, (224, 224))
            image_normal = image_resize / (255 / 2.0) - 1
            # image_normal = image_resize / 255.0
            # print(image_normal.shape)
            ###========================== general label =============================###
            label = [0] * 5
            label[y] = 1
            # print(label)
            xs.append(image_normal)
            ys.append(label)
            self.start+=1
            if self.start>=self.train_total:
                self.start=0
            if len(xs)>=batch_size:
                break
        # return xs,
        return np.array(xs,dtype=np.float32),np.array(ys,dtype=np.int64)