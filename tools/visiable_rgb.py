import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

savepath = "F:/vis_resnet50/features"
if not os.path.exists(savepath):
    os.mkdir(savepath)


def draw_features(width, height, x, savename):
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.reduce_64 = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.reduce_256 = nn.Conv2d(256, 3, kernel_size=1, stride=1)
        self.reduce_512 = nn.Conv2d(512, 3, kernel_size=1, stride=1)
        self.reduce_1024 = nn.Conv2d(1024, 3, kernel_size=1, stride=1)
        self.reduce_2048 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)

    def forward(self, x):
        if True:  # draw features or not
            x = self.model.conv1(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f1_conv1.png".format(savepath))
            x1 = self.reduce_64(x)
            draw_features(1, 1, x1.cpu().numpy(), "{}/f1_conv1_rgb.png".format(savepath))

            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f2_bn1.png".format(savepath))
            x2 = self.reduce_64(x)
            draw_features(1, 1, x2.cpu().numpy(), "{}/f2_bn1_rgb.png".format(savepath))

            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))
            x3 = self.reduce_64(x)
            draw_features(1, 1, x3.cpu().numpy(), "{}/f3_relu_rgb.png".format(savepath))

            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))
            x4 = self.reduce_64(x)
            draw_features(1, 1, x4.cpu().numpy(), "{}/f4_maxpool_rgb.png".format(savepath))

            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))
            x5 = self.reduce_256(x)
            draw_features(1, 1, x5.cpu().numpy(), "{}/f5_layer1_rgb.png".format(savepath))

            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))
            x6 = self.reduce_512(x)
            draw_features(1, 1, x6.cpu().numpy(), "{}/f6_layer2_rgb.png".format(savepath))

            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))
            x7 = self.reduce_1024(x)
            draw_features(1, 1, x7.cpu().numpy(), "{}/f7_layer3_rgb.png".format(savepath))

            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))
            x8 = self.reduce_2048(x)
            x8 = self.reduce_1024(x8)
            draw_features(1, 1, x8.cpu().numpy(), "{}/f8_layer4_rgb.png".format(savepath))

            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(savepath))
            plt.clf()
            plt.close()

            x = x.view(x.size(0), -1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(savepath))
            plt.clf()
            plt.close()
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x


# model = ft_net().cuda()
model = ft_net()

# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model.eval()
img = cv2.imread('F:/1.tif')
img = cv2.resize(img, (512, 512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# img = transform(img).cuda()
img = transform(img)
img = img.unsqueeze(0)

with torch.no_grad():
    start = time.time()
    out = model(img)
    print("total time:{}".format(time.time() - start))
    result = out.cpu().numpy()
    # ind=np.argmax(out.cpu().numpy())
    ind = np.argsort(result, axis=1)
    for i in range(5):
        print("predict:top {} = cls {} : score {}".format(i + 1, ind[0, 1000 - i - 1], result[0, 1000 - i - 1]))
    print("done")