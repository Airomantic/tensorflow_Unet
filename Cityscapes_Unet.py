
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

#不放心顺序对应，可做列表排序采用lambda，如果文件夹是按顺序来的，则彩色图像和目标label就是说明已经对应好了
#设置GPU
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
#仅限aachen
# img = sorted(glob.glob("Dataset/images/train/aachen/*.png"), key=lambda name:int(name[35:-23]))
img = glob.glob("Dataset/images/train/*/*.png") #彩色图像
# print(len(img))
# print(img[:5]) #前五张路径，实际并未按顺序来
label=glob.glob('Dataset/gtFine/train/*/*_gtFine_labelIds.png') #目标
train_count=len(label)
# print(label[510:515]) #510-515
# print(img)
'''
#这一步可做可不做，只是为了后面的BUFFER_SIZE可以小一点
#对全部的路径img和label做一个乱序
#在对小范围进行沙发，不至于它们全某两个城市或某一个城市
#在img这么长的范围内创建一个乱序，就是为了保证即便是乱序img-label也是对应的
index=np.random.permutation(len(img))
#列表转换成数组
img=np.array(img)[index] #根据index乱序
label=np.array(label)[index]
'''

img_val=glob.glob("Dataset/images/val/*/*.png")
label_val=glob.glob("Dataset/gtFine/val/*/*_gtFine_labelIds.png")
#拿到训练图片的张数
val_count=len(img_val)

print(len(label_val))
#创建数据集
dataset_train=tf.data.Dataset.from_tensor_slices((img,label))
# print(dataset_train)
dataset_val=tf.data.Dataset.from_tensor_slices((img_val,label_val))
def read_png(path):
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img,channels=3)
    return img
def read_png_label(path): #目标图像
    img=tf.io.read_file(path)
    img=tf.image.decode_png(img,channels=1)
    return img
img_l=read_png(img[0])
label_l=read_png_label(label[0])
print(img_l.shape) #(1024, 2048, 3) 图像过大显存爆炸需要缩小
print(label_l.shape) #(1024, 2048, 1) 为1所以裁剪比较麻烦
#数据增强 Unet 不能随机的左右翻转，要同时，因为要保证一一对应，这时还引入了随机性（深度学习要求有随机性）
# tf.random.uniform()
# tf.image.flip_left_right #同时翻转
#随机裁剪
# concat_img=tf.concat([img_l,label_l],axis=-1) #要求shape一致axis最后一维合并故axis=-1
 #img_l为乘3 label_l为乘1 叠加后为乘4
# print(concat_img.shape) #(1024, 2048, 4)
#裁剪函数 mask就是对应label
def crop_img(img,mask):
    concat_img = tf.concat([img,mask], axis=-1) #沿着channel的维度合并在一起，resize到较小的范围内
    concat_img=tf.image.resize(concat_img,(280,280),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR#双线性插值
                               )
    crop_img=tf.image.random_crop(concat_img,[256,256,4]) #随机裁剪
    # return crop_img #这样只能返回一个
    return crop_img[:,:,:3],crop_img[:,:,3:] #第三维选择[256,256,4]的前三个img裁剪之后的img和保留最后一个维度
plt.subplot(1,2,1) #一行两列第一个
img_l,label_l=crop_img(img_l,label_l)
plt.imshow(img_l.numpy())
plt.subplot(1,2,2)
plt.imshow(label_l.numpy())
plt.show()
