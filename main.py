import pickle
import os
import time
from PIL import Image
import numpy as np


ckpt = open('two_hidden_layer_epoch_5.pkl', 'rb')
model = pickle.load(ckpt)['model']
ckpt.close()

img_root = './data_utils/cole'
while True:
    """
    主循环，实际运行时在这个里面运行，其他的只是与训练有关，与主程序无关
    """
    while sum([len(x) for _, _, x in os.walk(img_root)]) < 10:
        """
        等待图片保存，我考虑的是保存十张以上的图片时开始运行模型进行预测
        """
        time.sleep(0.01)
    imgs = np.zeros([1, 12288])
    for f in os.listdir(img_root):
        """
        我把能读取到的图片都放到一个batch里面，把运行的结果取平均，
        同时读取一张图片就删除一张
        """
        img = Image.open(os.path.join(img_root, f))
        img = img.resize((64, 64))
        img = np.array(img).reshape(shape=[1, 12288])
        imgs = np.concatenate((imgs, img), axis=0)  # 组成一个batch
        os.remove(os.path.join(img_root, f))
    y_pred = model.loss(imgs)
    pred = np.mean(y_pred)  # 这个是预测值
