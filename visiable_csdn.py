import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio
from geoseg.models.OriMABES import BESNet

plt.rcParams['font.sans-serif'] = ['STSong']
import torchvision.models as models


# # model = models.alexnet(pretrained=False)
# model = BESNet(nclass=6, aux=False, pretrained=True)
#
# # 1.模型查看
# print(model)
# model_features = list(model.children())
# print(model_features[0][3])  # 取第0层Sequential()中的第四层
# for index, layer in enumerate(model_features[0]):
#     print(layer)


# 2. 导入数据
# 以RGB格式打开图像
# Pytorch DataLoader就是使用PIL所读取的图像格式
# 建议就用这种方法读取图像，当读入灰度图像时convert('')
def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')  # 是一幅图片
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)  # torch.Size([3, 224, 224])
    image_info = image_info.unsqueeze(0)  # torch.Size([1, 3, 224, 224])因为model的输入要求是4维，所以变成4维
    return image_info  # 变成tensor数据


# 2. 获取第k层的特征图
'''
args:
k:定义提取第几层的feature map
x:图片的tensor
model_layer：是一个Sequential()特征层
'''


def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):  # model的第一个Sequential()是有多层，所以遍历
            x = layer(x)  # torch.Size([1, 64, 55, 55])生成了64个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(512, 512))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    print(feature_map_num)
    row_num = int(np.ceil(np.sqrt(feature_map_num)))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        # plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        imageio.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1])
        # scipy.misc.imsave()
    plt.show()


if __name__ == '__main__':
    image_dir = r"car_logol.png"
    # 定义提取第几层的feature map
    k = 6
    image_info = get_image_info("D:/home/caoyiwen/data/vaihingen/train/images_1024/top_mosaic_09cm_area1_0_17.tif")

    # model = models.alexnet(pretrained=False)
    model = BESNet(nclass=6, aux=False, pretrained=True)
    model_layer = list(model.children())
    model_layer = model_layer[0]  # model的第一个Sequential()

    feature_map = model.forward()
    show_feature_map(feature_map)
