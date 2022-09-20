# PyTorch Tutorial

> 整理一下 Pytorch Tutorial

**reference**:

- [ML 2020 Spring - PyTorch_Introduction](https://colab.research.google.com/drive/1Xed5YSpLsLfkn66OhhyNzr05VE89enng#scrollTo=Xi_QP1bmMThC)
- [Learning the Basics - PyTorch Tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)

我当初是直接过了 [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，掌握了一些基本内容，这里在此基础上继续补充一下。

同时遇到问题一是查文档，注意 PyTorch 的版本。[PyTorch stable version documentation](https://pytorch.org/docs/stable/index.html)。

二是直接在 Google 上面搜，找 GitHub issues 或是 StackOverflow，或者 CSDN 和 cnblogs 偶尔也能给出靠谱的解决办法。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

torch.manual_seed(2375)
np.random.seed(2375)
```

## Tensor

Tensor 和 vector，matrix 几乎可以说没有什么区别。只不过 tensor 一是可以在 gpu 运行，比如矩阵乘法会加速很多。二是可以作为函数变量储存梯度信息，自动求梯度。PyTorch 的 Tensor 和 numpy 的 array 特别像，请看：

```python
# Initializing

x_numpy_data = np.array([[0, 1, 2], [3, 4, 5]])
x_torch_data = torch.tensor([[0, 1, 2], [3, 4, 5]])

print(x_numpy_data, x_torch_data)
print('*'*50)

x_numpy_ones = np.ones((2, 3))  # 必须传入 array like 的 shape，而不是 np.ones(2, 3)
x_torch_ones = torch.ones((2, 3)) # torch.ones(2, 3) 也 work

print(x_numpy_ones, x_torch_ones)
print('*'*50)

x_numpy_zeros = np.zeros((2, 3)) # 同 x_numpy_ones
x_torch_zeros = torch.zeros((2, 3)) # 同 x_torch_zeros

print(x_numpy_zeros, x_torch_zeros)
print('*'*50)

x_numpy_rand = np.random.rand(2, 3, 4) # numpy 比较怪，这里生成 0~1 的随机数，又变成传一系列 dimensions 了，array like shape 不能 work
x_torch_rand = torch.rand(2, 3, 4) # torch.rand((2, 3, 4)) 也 ok

print(x_numpy_rand, x_torch_rand)
print('*'*50)
```

    [[0 1 2]
     [3 4 5]] tensor([[0, 1, 2],
            [3, 4, 5]])
    **************************************************
    [[1. 1. 1.]
     [1. 1. 1.]] tensor([[1., 1., 1.],
            [1., 1., 1.]])
    **************************************************
    [[0. 0. 0.]
     [0. 0. 0.]] tensor([[0., 0., 0.],
            [0., 0., 0.]])
    **************************************************
    [[[0.43749531 0.00136684 0.20813947 0.2688284 ]
      [0.0613601  0.36398268 0.80690013 0.17517012]
      [0.4845313  0.87873553 0.36479806 0.66866733]]
    
     [[0.3276178  0.79793325 0.3168611  0.73746649]
      [0.42691957 0.97367206 0.51520666 0.53565325]
      [0.45002277 0.58673459 0.38951537 0.51184156]]] tensor([[[0.9988, 0.8845, 0.3499, 0.7021],
             [0.2837, 0.1307, 0.8201, 0.6429],
             [0.7082, 0.5319, 0.1796, 0.1864]],
    
            [[0.5664, 0.4796, 0.8436, 0.4824],
             [0.0400, 0.4130, 0.9563, 0.4429],
             [0.3883, 0.0825, 0.1788, 0.1438]]])
    **************************************************

```python
# Operation

# 大部分运算符包括 @ 都是一样的
# 就是 numpy 中像 mean 函数需要设置 axis，在 torch 中改名为 dim，但是本质一样，沿着轴（我第一遍也搞反了）

print(np.mean(x_numpy_data, axis=0))
print(torch.mean(x_torch_data.float(), dim=0))  # torch 的 mean 必须用 float 类型

print(np.mean(x_numpy_data, axis=1))
print(torch.mean(x_torch_data.float(), dim=1))
```

    [1.5 2.5 3.5]
    tensor([1.5000, 2.5000, 3.5000])
    [1. 4.]
    tensor([1., 4.])

```python
# Torch and numpy

print(torch.from_numpy(x_numpy_data))   # numpy to torch
print(x_torch_data.numpy())             # torch to numpy
```

    tensor([[0, 1, 2],
            [3, 4, 5]])
    [[0 1 2]
     [3 4 5]]

```python
# Tensor on GPU

x_gpu = None
if torch.cuda.is_available():
    x_gpu = x_torch_data.cuda()

print(x_gpu)
```

    tensor([[0, 1, 2],
            [3, 4, 5]], device='cuda:0')

## Dataset & Dataloaders

其实我觉得数据集处理应该是比较重要的部分，数据集处理好了后面模型什么的就一目了然。数据集处理就像个拦路虎，一旦没有别人 implement 的现成代码，我甚至都没有动力去用那个数据集。不得不用怎么办？只能学了。

### Custom Dataset

Dataset 是一个 class，用来存和读入数据集，如果想要 custom，就需要继承 Dataset 类。自定义的类必须覆写以下三个函数（方法）：

- `__init__`
- `__len__`
- `__getitem__`

这里我们自定义一个 FashionMNIST 数据集：

```python
from torch.utils.data import Dataset

import os
import pandas as pd
from torchvision.io import read_image

class FashionMNISTDataset(Dataset):
    """Load FashionMNIST image dataset"""

    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        """ 初始化自定义
        
        __init__ 函数是我们首先需要自定义的，你需要在声明类的时候传入所有需要用到的东西

        Arguments:
            annotation_file: 数据集 csv 格式，存图片文件名和标签
            img_dir: 存图片路径
            transform, targer_transform: 两个函数

        """

        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """ 重定义方法，返回数据集的 size"""

        return len(self.img_labels)

    def __getitem__(self, index):
        """ 重定义方法，用于返回第 index 个数据
            这里 read_image 只是一个函数，transform 也是函数，你自定义输出什么
        """

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label        
```

### Use `DataLoader` to enumerate dataset

Dataloader 比 for 循环牛逼在以下方面：

- 可以设置 batch size 和处理最后一个 batch 的方法
- 可以 shuffle
- 可以设置 batch 数据集是否需要做额外处理
- 可以设置多进程 load dataset

`DataLoader` 长这个样子

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```

- dataset: 由你自定义的 `Dataset` 创建的一个对象
- batch_size: batch size
- shuffle: 是否在每轮 epoch 打乱数据集
- num_workers: 多少个进程同时 load 数据集，cuda 下吃显存
- collate_fn: batch 的时候对数据集额外操作的函数
- drop_last: 最后一个不足 batch size 的 batch 留不留

```python
# 这个代码跑不了，需要自己下载数据集

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

annotation_file = 'Your csv file path here'
img_dir = 'Your image file directory path here'
transform = ToTensor()
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

dataset = FashionMNISTDataset(annotation_file, img_dir, transform=transform, target_transform=target_transform)
dataloader = DataLoader(dataset)

for idx, example in enumerate(dataloader):
    pass
```

## Transforms

有的 CustomDataset 类会给你各种 transform 的参数，这是什么意思呢？

是说让你把 dataset 和 label 通过两个 transform 函数转换成某种形式，一般 data 需要变成 tensor，而 label 需要由数字变成 one-hot tensor。

`ToTensor()` 方法让你把 ndarray 或者 PIL image 类型的数据变成 tensor，有点像 `torch.from_numpy()`

`Lambda()` 里面传入 lambda 函数，上面 `target_transform` 的代码做了这么几件事：

- 声明一个一维，大小为 10 的 0 tensor
- 下面用到了 `torch.scatter_` 函数，详见[torch.scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)
  - 这个 document 有点小问题，第一个参数 dim，第二个 index，第三个 src。但是第三个 src 也可以是 scalar，这时需要用 `value=`，文档没有
  - 关于 scatter 的理解：[This blog](https://yuyangyy.medium.com/understand-torch-scatter-b0fd6275331c)
  - 所以这里就是对第 y 个元素赋值
  - torch 的 tensor 赋值用 scatter，因为 cuda tensor 没有 fancy indexing，for 循环又太慢，这个才是一种 indexing 方法

## Neural Network

接下来就来开始写神经网络了，我决定重新用 PyTorch 写作业 2 的 classification，叠三层网络就够了，看看效果如何。

```python
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import numpy as np

class Args(object):
    """ Define arguments, just like hw2

    Attributes:
        dev_ratio: dev set ratio, default = 0.1
        max_iter: max training iteration steps, default = 10
        batch_size: batch size. default = 8
        learning_rate: learning rate, default = 0.2
        seed: random seed, default = 0
    """

    def __init__(self):

        self.dev_ratio = 0.1
        self.max_iter = 10
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.seed = 0
        self.input_size = 510
        self.hidden_size = 256


class ProcessedDataset(Dataset):

    def __init__(self, X_data_path, Y_data_path=None, train=True, norm=False, X_mean=None, X_std=None):

        # Parse csv files to numpy array
        with open(X_data_path) as f:
            next(f)
            X_data = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
        if train == True:
            assert Y_data_path is not None
            with open(Y_data_path) as f:
                next(f)
                Y_data = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

        
        # normalization
        if norm == True:
            # Training normalization
            if train == True:
                X_mean = np.mean(X_data, axis=0)
                X_std = np.std(X_data, axis=0)
                X_data = (X_data - X_mean) / (X_std + 1e-8)
            
            # Testing normalization
            else:
                assert X_mean is not None
                assert X_std is not None
                X_data = (X_data - X_mean) / (X_std + 1e-8)

        # convert ndarray to tensor
        X_data = torch.from_numpy(X_data).float()                   # 这里必须是 float，forward 需要传 float，要么你 model.double()
        if train == True: Y_data = torch.from_numpy(Y_data).flatten().long()    # 这里需要 long，而且是一维的要 flatten 一下

        if train == True:
            self.item = {
                'data': X_data,
                'label': Y_data,
                'mean': X_mean,
                'std': X_std,
            }
        else:
            self.item = {
                'data': X_data,
            }
        self.train = train

    
    def __len__(self):

        return self.item['data'].shape[0]


    def __getitem__(self, index):

        if self.train == True:
            return self.item['data'][index], self.item['label'][index]
        else:
            return self.item['data'][index]


X_train_fpath = './hw2_data/X_train'
Y_train_fpath = './hw2_data/Y_train'
X_test_fpath = './hw2_data/X_test'
output_fpath = './output_{}.csv'

# get arguments
args = Args()
np.random.seed(args.seed)

# get dataset
train_set = ProcessedDataset(X_train_fpath, Y_train_fpath, train=True, norm=True)
X_mean, X_std = train_set.item['mean'], train_set.item['std']
test_set = ProcessedDataset(X_test_fpath, train=False, norm=True, X_mean=X_mean, X_std=X_std)

# train dev split
dev_size = int(len(train_set) * args.dev_ratio)
train_size = len(train_set) - dev_size
train_set, dev_set = random_split(train_set, [train_size, dev_size], generator=torch.Generator().manual_seed(args.seed))

print('Size of training set: {}'.format(len(train_set)))
print('Size of development set: {}'.format(len(dev_set)))
print('Size of testing set: {}'.format(len(test_set)))

# generate dataloader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=args.batch_size)
test_loader = DataLoader(test_set, batch_size=args.batch_size)
```

    Size of training set: 48831
    Size of development set: 5425
    Size of testing set: 27622

`super.__init__()` 的作用在[这里](https://blog.csdn.net/qq_38787214/article/details/87902291)，它可以使子类调用父类的属性。

```python
from torch import nn

class HW2Network(nn.Module):

    def __init__(self, args):

        super(HW2Network, self).__init__()
        self.flatten = nn.Flatten()
        self.logistic_regression = nn.Sequential(
            nn.Linear(args.input_size, 1),
            nn.Sigmoid(),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(args.input_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 2), # 不需要 sigmoid 或者 softmax，如果使用 ce 的话
        )


    def forward(self, x):

        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits   # CE
        # return logits.flatten() # BCE


def train(model, train_loader, dev_loader, args):

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()

    # optimization method
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.99)  # SGDM
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.max_iter*0.3), int(args.max_iter*0.7)])

    train_loss = []
    train_acc = []
    dev_acc = []
    correct, total = 0, 0

    # Training
    for epoch in range(args.max_iter):
        model.train()
        # Training
        for idx, (X, Y) in enumerate(train_loader):

            X, Y = X.cuda(), Y.cuda()
            # forward
            logits = model(X)
            # loss = loss_fn(logits, Y.float()) # BCE
            loss = loss_fn(logits, Y)    # CE

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute correct examples
            # correct += (logits.round() == Y).type(torch.float).sum().item()     # BCE
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
                # correct += (logits.round() == Y).type(torch.float).sum().item() # BCE
                correct += (logits.argmax(1) == Y).type(torch.float).sum().item()    # CE
                total += torch.ones_like(Y).sum().item()

            dev_acc.append(correct/total)
            correct, total = 0, 0

    print('Training loss: {}'.format(train_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    return train_loss, train_acc, dev_acc


# model
model = HW2Network(args)
model.cuda()

# train
train_loss, train_acc, dev_acc = train(model, train_loader, dev_loader, args)
```

    Training loss: 0.2635279595851898
    Training accuracy: 0.9147877372980279
    Development accuracy: 0.8833179723502305

```python
def plot_loss_curve(train_loss, dev_loss, train_acc, dev_acc):

    import matplotlib.pyplot as plt
        
    # Loss curve
    plt.plot(train_loss)
    plt.title('Loss')
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()

    
# plot loss curve
plot_loss_curve(train_loss, dev_loss, train_acc, dev_acc)
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

with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
```

几点注意:

- `nn.CrossEntropyLoss` 和一般的 loss function 不太一样, 用于多类别分类问题, 一是不需要单独写 softmax 函数, 二是不需要网络输出 (minibatch size, C) 的 shape. (在 test 的时候当然也不需要写 softmax 了hhh)
- 我最开始发现模型准确率只有 56 左右, 以为自己哪里写错了, 后来发现准确率应该是小于 1 的, 我并没有用百分数表示... 那为什么算出了 56 大的离谱的准确率呢? 因为我分母部分算的是 minibatch 的数量, 而不是 example 的数量... 除以 batch size (64) 就正常了...
- validate 时, `with torch.no_grad()` 用于停掉梯度计算, 减少资源占用, 加速. 而 `model.eval()` 用于通知模型进入测试阶段, 不再 dropout 等等. 最好都用上.
- scheduler 用于对学习率进行某种调整策略, 比如 `MultiStepLR` 是先 warm up 再 annealing.

关于一个 NN 的大致框架:

```python
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

# 1. Define your dataset and dataloader
class YourDataset(Dataset):
    """ Your dataset here. """

    def __init__(self, *args, **kwargs):
        
        Your code about how you process the data


    def __len__(self):

        return the length of your dataset


    def __getitem__(self, index):

        return the corresponding example given the index


your_dataset = YourDataset(*args, **kwargs)
your_dataloader = DataLoader(your_dataset, *args, **kwargs)


# 2. Define your neural network
class YourNetwork(nn.Module):
    """ Your network here. """

    def __init__(self, *args, **kwargs):
        
        super(YourNetwork, self).__init__()
        Your code here


    def forward(self, *args, **kwargs):

        Your code here


model = YourNetwork(*args, **kwargs)


# 3. Train and validate
def train(model, your_dataloader, *args, **kwargs):

    Some of your codes here

    # Choose a loss function
    loss_fn = None

    # Choose an optimization method
    optimizer = None
    scheduler = None

    Some of your codes here

    # Training
    for epoch in range(args.max_iter):
        model.train()
        # Training
        for idx, (X, Y) in enumerate(your_dataloader):

            some of your codes here

            X, Y = X.cuda(), Y.cuda()
            # forward
            logits = model(X)

            loss = compute loss with loss_fn

            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            some of your codes here
        scheduler.step()

        # Validate
        some params = eval(model, your_dataloader, *args, **kwargs)

    return something


def eval(model, your_dataloader, *args, **kwargs):

    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(dev_loader):

            X, Y = X.cuda(), Y.cuda()
            logits = model(X)

            some of your codes here

    return something
```

## Autograd

pytorch 比较神奇的一点就是可以自动为你求好梯度. 只需要 `loss.backward()` 反向传播计算梯度, 然后 `optimizer.step()` 更新参数. 怎么做到的呢? 其实就是链式法则.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/975018ba446be7ce4b8203e80696d11e1136a873493113aa3261239ae065aea7.png" style="zoom:50%;" />

定义一个网络, 其函数为 $y=\sin(x_5)=\sin(x_4x_2x_3)=\sin((x_1+x_2)x_2x_3)$, 其中 $x_1, x_2, x_3$ 为 input, $y$ 为 output, $x_4, x_5$ 为 hidden states. 设置 $x_1, x_2, x_3$ 为 0, 1, 2. 求 y 在此时对 $x_1, x_2, x_3$ 的偏导.

我们发现这就是高数中利用链式法则求偏导的过程, 高数中我们因为很少求具体数值, 所以直接反向从 $y=\sin(x_5)$ 开始利用我们知道的求导法则求出关于 $x_5$ 的偏导, 再求 $x_5$ 关于 $x_4$, 这时 $y$ 关于 $x_4$ 的就是 $y$ 关于 $x_5$ 的乘以 $x_5$ 关于 $x_4$ 的.

我们还知道比如求 $y$ 关于 $x_2$ 的偏导, 首先仍然是求 $y$ 关于 $x_5$, 然后这时出现两条路. 一个是从 $x_5$ 到 $x_2$. 另一种是 $x_5$ 到 $x_4$ 乘 $x_4$ 到 $x_2$. 所以总的偏导就是 $\frac{\partial y}{\partial x_5}(\frac{\partial x_5}{\partial x_2}+\frac{\partial x_5}{\partial x_4}\frac{\partial x_4}{\partial x_2})$

这时候人能写出个式子, 代数进去算, 求好了. 计算机不懂求导法则, 怎么办呢? 计算机只能在给 x, y 的情况下近似算出 y 对 x 的偏导, 用 $\frac{\Delta y}{\Delta x}$ 算. 这时就需要先 forward 求出网络中每一个节点的值, 比如说求出 $x_5=2$, $y=0.9093$. 再开始反向求一个近似的导 $\frac{\sin(2+\epsilon)-\sin(2)}{\epsilon}$, 一点点反向求回去.

pytorch 还有个细节就是每个节点有个 `is_leaf` 属性和 `requires_grad` 属性. 叶子节点是用户创建的, 比如这里的 input, NN 中需要更新的权值. 非叶子节点是中间量, 依赖叶子节点. 非叶子节点的梯度求完之后会被释放掉. 除非设置 `retain_graph=True`. 而 `requires_grad` 代表是否需要为这个 tensor 求梯度. **我 input 那里写错了, 应该是 `True` 的.**

详见[一文解释Pytorch求导相关 (backward, autograd.grad)](https://zhuanlan.zhihu.com/p/279758736).
