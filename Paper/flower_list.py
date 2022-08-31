import numpy as np
import random
import os
import math


from matplotlib import pyplot as plt

def get_files(file_dir):
    """
    创建数据文件名列表

    :param file_dir:
    :return:image_list 所有图像文件名的列表,label_list 所有对应标贴的列表
    """
    #step1.获取图片，并贴上标贴
    #新建五个列表，存储文件夹下的文件名
    daisy=[]
    label_daisy=[]
    dandelion=[]
    label_dandelion = []
    roses=[]
    label_roses = []
    sunflowers=[]
    label_sunflowers = []
    tulips=[]
    label_tulips = []
    print(os.getcwd())
    for file in os.listdir(file_dir+"/daisy"):
        daisy.append(file_dir+"/daisy"+"/"+file)
        label_daisy.append(0)

    for file in os.listdir(file_dir+"/dandelion"):
        dandelion.append(file_dir+"/dandelion"+"/"+file)
        label_dandelion.append(1)
    for file in os.listdir(file_dir+"/roses"):
        roses.append(file_dir+"/roses"+"/"+file)
        label_roses.append(2)
    for file in os.listdir(file_dir+"/sunflowers"):
        sunflowers.append(file_dir+"/sunflowers"+"/"+file)
        label_sunflowers.append(3)
    for file in os.listdir(file_dir+"/tulips"):
        tulips.append(file_dir+"/tulips"+"/"+file)
        label_tulips.append(4)

    #step2:对生成的图片路径和标签List做打乱处理
    #把所有图片跟标贴合并到一个列表list（img和lab）
    images_list=np.hstack([daisy,dandelion,roses,sunflowers,tulips])
    labels_list=np.hstack([label_daisy,label_dandelion,label_roses,label_sunflowers,label_tulips])

    #利用shuffle打乱顺序
    temp=np.array([images_list,labels_list]).transpose()
    np.random.shuffle(temp)
    # 从打乱的temp中再取出list（img和lab）
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list_new=[int(i) for i in label_list]

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # 测试样本数, ratio是测试集的比例
    ratio=0.2
    n_sample = len(label_list)
    n_val = int(math.ceil(n_sample * ratio))
    n_train = n_sample - n_val  # 训练样本数
    tra_images = image_list[0:n_train]
    tra_labels = label_list_new[0:n_train]
    #tra_labels = [int(float(i)) for i in tra_labels]  # 转换成int数据类型
    val_images = image_list[n_train:-1]
    val_labels = label_list_new[n_train:-1]
    #val_labels = [int(float(i)) for i in val_labels]  # 转换成int数据类型
    return tra_images, tra_labels, val_images, val_labels