# Convolutional Neural Network

> Lazy re-implementation. 并没有对模型进行进一步的优化. 目前模型的表现很烂.

改动:

- 只是可以稍微方便地修改 CNN 的内部结构.
- 加了 multistepLR 让训练的时候不那么过拟合.

## DownLoad Datasets

```python
!wget https://drive.google.com/uc?id=19CzXudqN58R3D-1G8KeFWk8UDQwlb8is
```

```python
!unzip food-11.zip
```

## Read image

利用 OpenCV (cv2) 读入照片并存放在 numpy array 中

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
```

```python
class Args(object):
    """

    Attributes:
        dev_ratio: dev set ratio, default = 0.1
        max_iter: max training iteration steps, default = 10
        batch_size: batch size. default = 8
        learning_rate: learning rate, default = 0.2
        seed: random seed, default = 0
    """

    def __init__(self):

        self.channels = np.array([64, 128, 256, 512, 512])
        self.batch_size = 128
        self.num_epochs = 30
        self.learning_rate = 1e-3

args = Args()
```

```python
def readfile(path, need_label):
    """

    Args:
        - path: str, 读取图片的根目录
        - need_label: boolean, 是否为训练模式, 需要标签

    Output:
        - x: ndarray(uint8), shape = (数据大小, 128, 128, 3)
        - y(如果有标签): ndarray(uint8), shape=(数据大小)
    """

    # 根目录下所有图片由 "label_数字" 组成, 先进行排序
    image_dir = sorted(os.listdir(path))

    # 初始化
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)

    for idx, img_dir in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, img_dir))
        x[idx, :, :] = cv2.resize(img, (128, 128))
        if need_label:
            y[idx] = int(img_dir.split("_")[0])

    if need_label:
        return x, y
    else:
        return x
```

```python
workspace_dir = './food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))
```

    Reading data
    Size of training data = 9866
    Size of validation data = 3430
    Size of Testing data = 3347

## Dataset and DataLoader

自定义 dataset 和 dataloader

```python
class ImgDataset(Dataset):
    
    def __init__(self, x, y=None, transforms=None):
        self.x = x
        self.y = y

        # label is required to be a LongTensor
        if y is not None:
            self.y = torch.LongTensor(y)

        self.transforms = transforms

    
    def __len__(self):
        return len(self.x)

    
    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transforms is not None:
            X = self.transforms(X)

        if self.y is not None:
            Y = self.y[idx]
            return X, Y
        else:
            return X


# training 时做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机上下翻转
    transforms.RandomRotation(15),      # 随机旋转
    transforms.ToTensor(),              # 将图片变为 tensor 并且 normalize 到 [0, 1]
])

# testing 时不需要 augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
```

```python
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
```

## CNN Model

使用 add_module 更方便设计网络结构. 需要研究明白这几个 module.

- [pytorch之torch.nn.Conv2d()函数详解_夏普通的博客-CSDN博客_torch.nn.conv2d](https://blog.csdn.net/qq_34243930/article/details/107231539)
- [PyTorch中torch.nn.MaxPool2d参数解释 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/488086621)
- BatchNorm2d 好像没看懂

```python
class ImgClassifier(nn.Module):

    def __init__(self, channels=5, conv_kernel_size=3, conv_stride=1, pool_kernel_size=2, pool_stride=2):

        """
        Args:
            - channels: ndarray(int), 每个值代表这一层的 out_channels
            - conv_kernel_size: int, Conv2d 的 kernel size
            - conv_stride: int, Conv2d 的 stride
            - pool_kernel_size: int, MaxPool2d 的 kernel size
            - pool_stride: int, MaxPool2d 的 stride

        """
        super(ImgClassifier, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [3, 128, 128]

        self.cnn = nn.Sequential()
        in_channels = 3

        # 动态创建 nn.Sequential()
        for idx, out_channels in enumerate(channels):
            self.cnn.add_module("Conv2d-{}".format(idx), nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, 1))
            self.cnn.add_module("BatchNorm2d-{}".format(idx), nn.BatchNorm2d(out_channels))
            self.cnn.add_module("ReLU-{}".format(idx), nn.ReLU())
            self.cnn.add_module("MaxPool2d-{}".format(idx), nn.MaxPool2d(pool_kernel_size, pool_stride, 0))
            in_channels = out_channels

        # 这里懒得动态创建了
        self.fc = nn.Sequential(
            nn.Linear(out_channels*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )


    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
```

## Train and Validate

用的是上一次的代码

```python
def train(model, train_loader, dev_loader, args):

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimization method
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.num_epochs*0.3), int(args.num_epochs*0.7)])

    train_loss = []
    train_acc = []
    dev_acc = []
    dev_loss = []
    correct, total = 0, 0

    # Training
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        # Training
        for idx, (X, Y) in enumerate(train_loader):

            X, Y = X.cuda(), Y.cuda()
            # forward
            logits = model(X)
            loss = loss_fn(logits, Y)    # CE

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute correct examples
            correct += (logits.argmax(1) == Y).type(torch.float).sum().item()    # CE
            total += torch.ones_like(Y).sum().item()
        scheduler.step()

        train_loss.append(loss.item())
        train_acc.append(correct/total)
        correct, total = 0, 0

        # Validate
        model.eval()
        with torch.no_grad():
            for idx, (X, Y) in enumerate(dev_loader):

                X, Y = X.cuda(), Y.cuda()
                logits = model(X)
                loss = loss_fn(logits, Y)
                correct += (logits.argmax(1) == Y).type(torch.float).sum().item()    # CE
                total += torch.ones_like(Y).sum().item()

            dev_loss.append(loss.item())
            dev_acc.append(correct/total)
            correct, total = 0, 0

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Dev Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, args.num_epochs, time.time()-epoch_start_time, \
             train_acc[-1], train_loss[-1], dev_acc[-1], dev_loss[-1]))

    return train_loss, dev_loss, train_acc, dev_acc


# model
model = ImgClassifier(args.channels)
model.cuda()

# train
train_loss, dev_loss, train_acc, dev_acc = train(model, train_loader, val_loader, args)
```

    [001/030] 27.80 sec(s) Train Acc: 0.215792 Loss: 1.699216 | Dev Acc: 0.202332 loss: 1.713886
    [002/030] 27.61 sec(s) Train Acc: 0.310663 Loss: 1.932884 | Dev Acc: 0.212245 loss: 1.978866
    [003/030] 27.53 sec(s) Train Acc: 0.372289 Loss: 1.823330 | Dev Acc: 0.352770 loss: 1.474082
    [004/030] 28.12 sec(s) Train Acc: 0.423981 Loss: 1.648620 | Dev Acc: 0.399125 loss: 1.676793
    [005/030] 27.50 sec(s) Train Acc: 0.466248 Loss: 1.257400 | Dev Acc: 0.275802 loss: 1.894674
    [006/030] 27.33 sec(s) Train Acc: 0.495844 Loss: 1.801906 | Dev Acc: 0.448105 loss: 1.066376
    [007/030] 27.77 sec(s) Train Acc: 0.518346 Loss: 1.312358 | Dev Acc: 0.426531 loss: 1.030666
    [008/030] 27.33 sec(s) Train Acc: 0.543077 Loss: 2.008324 | Dev Acc: 0.412536 loss: 1.490397
    [009/030] 28.45 sec(s) Train Acc: 0.568518 Loss: 1.647052 | Dev Acc: 0.496210 loss: 0.602673
    [010/030] 27.39 sec(s) Train Acc: 0.643726 Loss: 1.159978 | Dev Acc: 0.620408 loss: 0.665697
    [011/030] 28.28 sec(s) Train Acc: 0.667646 Loss: 0.995019 | Dev Acc: 0.609913 loss: 0.701649
    [012/030] 27.52 sec(s) Train Acc: 0.675654 Loss: 1.259305 | Dev Acc: 0.630321 loss: 0.545351
    [013/030] 28.66 sec(s) Train Acc: 0.681938 Loss: 1.279580 | Dev Acc: 0.617784 loss: 0.647482
    [014/030] 27.67 sec(s) Train Acc: 0.690249 Loss: 0.989793 | Dev Acc: 0.635277 loss: 0.575931
    [015/030] 27.64 sec(s) Train Acc: 0.696939 Loss: 1.033612 | Dev Acc: 0.633819 loss: 0.539016
    [016/030] 27.87 sec(s) Train Acc: 0.703629 Loss: 0.556636 | Dev Acc: 0.647230 loss: 0.486073
    [017/030] 27.42 sec(s) Train Acc: 0.714474 Loss: 1.358962 | Dev Acc: 0.646647 loss: 0.612916
    [018/030] 27.34 sec(s) Train Acc: 0.720860 Loss: 0.977474 | Dev Acc: 0.654227 loss: 0.456925
    [019/030] 27.33 sec(s) Train Acc: 0.724508 Loss: 1.235370 | Dev Acc: 0.652478 loss: 0.551262
    [020/030] 27.53 sec(s) Train Acc: 0.728664 Loss: 1.366783 | Dev Acc: 0.650146 loss: 0.423557
    [021/030] 27.62 sec(s) Train Acc: 0.737280 Loss: 0.891979 | Dev Acc: 0.662974 loss: 0.488222
    [022/030] 27.50 sec(s) Train Acc: 0.752686 Loss: 0.478060 | Dev Acc: 0.675510 loss: 0.486420
    [023/030] 27.71 sec(s) Train Acc: 0.756639 Loss: 1.010040 | Dev Acc: 0.674636 loss: 0.463729
    [024/030] 27.93 sec(s) Train Acc: 0.754612 Loss: 0.905501 | Dev Acc: 0.674636 loss: 0.477225
    [025/030] 28.24 sec(s) Train Acc: 0.752686 Loss: 1.441572 | Dev Acc: 0.676968 loss: 0.500953
    [026/030] 27.52 sec(s) Train Acc: 0.756436 Loss: 0.715782 | Dev Acc: 0.677259 loss: 0.428181
    [027/030] 27.57 sec(s) Train Acc: 0.760491 Loss: 1.347846 | Dev Acc: 0.677551 loss: 0.477506
    [028/030] 27.71 sec(s) Train Acc: 0.760795 Loss: 0.677938 | Dev Acc: 0.679592 loss: 0.491113
    [029/030] 28.23 sec(s) Train Acc: 0.761403 Loss: 1.153831 | Dev Acc: 0.676676 loss: 0.475061
    [030/030] 28.26 sec(s) Train Acc: 0.763734 Loss: 0.431732 | Dev Acc: 0.678426 loss: 0.441988

## Plot Curve

绘制 train 和 validation 的 acc 和 loss

```python
def plot_curve(train_loss, dev_loss, train_acc, dev_acc):
    
    import matplotlib.pyplot as plt

    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()

plot_curve(train_loss, dev_loss, train_acc, dev_acc)
```

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/caf145ac6be39d95a197d177e0a5606eaeb490b9d05d62ac18212913309db413.png" />

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/a84ef117dab52355ad4c9c064c8119b249a4dfcdc3e815774283878497992419.png" />

## Test

将 test 的结果写入 predict.csv 当中.

```python
test_set = ImgDataset(test_x, transforms=test_transform)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
```

```python
def test(model, test_loader, args):

    predictions = np.array([], dtype='int64')
    model.eval()
    for idx, X in enumerate(test_loader):
        X = X.cuda()
        logits = model(X)
        pred = logits.argmax(1).cpu().numpy()
        predictions = np.append(predictions, pred)

    return predictions


# test
predictions = test(model, test_loader, args)

with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(predictions):
        f.write('{},{}\n'.format(i, y))
```
