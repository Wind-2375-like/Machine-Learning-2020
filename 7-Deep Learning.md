# ML 2020: Deep Learning

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [神经网络训练中的梯度消失与梯度爆炸 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/25631496)
    1. [L2正则=Weight Decay？并不是这样 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/40814046)
    
3. **Extra videos:** None

---

## History

我猜每门 Machine Learning 都会讲一讲 history. 最初就是 perceptron (跟 Logistic Regression 一样, 只不过 sigmoid 是任意非线性函数), 发现解不了异或. 于是寄了.

后面 80 年代提出多层感知机, 加上 Hinton 的 back propagation, 基本就是现在 DNN 的样子. 但是算力不行没发展起来. 而且当时发现 1 hidden layer 就够了, 不需要 deep.

接着 06 年 Hinton 提出了 RBM initialization, 就跟石头汤里的石头一样激起了 DL 的研究. GPU 出现加速算力的发展. 之后几个比赛用这东西发现好用, 就火了.

想说什么呢? 现在的人争相进入 DL 给挖开的坑灌水, 灌完了别人也忘了, 但是能被铭记的只有最初挖坑的大佬. 往往坑不是一下子就能挖出来的, 需要默默耕耘几十年. 这就是大佬和普通人的区别啊.

<span class="heimu" title="=.=">当然我安心地做一只菜鸡就够了.</span>

## Three Steps for Deep Learning

deep learning, 也是三步完成.

<img alt="图 18" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ae5d20b6b3bb7b5c9c46d9d0fbad078eac20fc65c8c382ce09502a37123cf7a7.png" />

### Define a Neural Network

Neural Network (NN) 由一个个 neuron 所组成. neuron 就是 $f(wx+b)$. 一个 NN 所有 neuron 的 w 和 b 构成网络的参数 $\theta$.

**Structure**:

不同的组成方式, 构成不同的 structure, 定义了不同的 function set.

定义网络的 structure 目前仍然是 intuitive and experimental. lecture 有提到 Evolutionary Artificial Neural Networks 说是可以让 structure automatically determined, 但我没听过... <span class="heimu" title="=.=">可能是我菜.</span>

用的最多的还是 fully connected feedforward network.

<img alt="图 22" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b2413dc378ae739f3908e60748bb8185d37dd5ba2d135fb38d09230f90a5d0fd.png" />  

分为 input layer, hidden layers, 和 output layer. 每个 layer 的 neuron 数目是可以不一样的. 最后从 output layer 到 output 需要经过一个 softmax.

deep learning 就是有很多 hidden layers. 比如 Residual Net 152 层. 我家才住 25 层.

**Matrix Operation:**

这个看图就行了.

<img alt="图 20" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/bd104b9b1c37a0252aca3bd94f80e785473a971ed9b0eefa072d913035fd6744.png" />  

<img alt="图 21" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/19ce804a32a23d1947e9c6970a6a8e5be14148c27ffc8ee3823e8796d31037bb.png" />

这里 GPU 变成爹了.

### Goodness of Function

仍然是 cross entropy.

$$
L = \sum_{n}C_n = \sum_{n}{-\sum_{k}\hat{y}_k^n\ln y_k^{n}}
$$

### Pick the Best Function: Backpropagation

仍然需要梯度下降更新参数, 但是怎么求出关于每个 parameter 的 partial derivative 呢?

懒得讲, 直接一图说明:

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/28d37f7daf93cbcf3a8d09b34eafa5a95ceca51f1ec1ec1e5aaf3450b9ee2c27.png" />

求完导数之后, 就可以 gradient descent 更新 w 和 b 了.

## Tips for Training DNN

DNN 会了, 可是东西写完还可能是一坨屎. 不调怎么行呢? 但是各种调的方法不能随便用, 要看情况.

### Overview

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9da417656e9186e7d756989a334277686ab13a6e47c1647f4046f91daf9be091.png" />

### Different Approach

首先需要注意, 如果 training data 就很烂的话, 那根本就不是 overfitting.

**Activation Function**

sigmoid 函数在网络越深的时候表现越烂, 为什么呢?

因为消失梯度问题. sigmoid 的导数如下所示:

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9452a7e226e1c2b74e48ecbaa24bbe9c1765d5daa7ab5e0323f5d461d3997890.png" />  

最大才 1/4, 层数一多, 链式法则连乘, 梯度自然传到前面层就没了, 因此 lecture 吐槽:

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/3b86a8c770e7e20c092cebd47832296ededd68730db66a7fc0f04e543615f4bb.png" />

所以提出了一些新的 activation function

1. ReLU
   <img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9725adb598f414c0d2c5655d854f22b9b4f6b4cad6b385b46604e156488ecc7c.png" />

   不是连续函数怎么求导啊? 原来本质上像开关一样, 最终变成 thinner linear network. 不存在梯度消失.

   <img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5ce7e5aa018d65827f96c2021e3d7a1e8a1d73e96c022c6a0c7420052f48ef25.png" />

   ReLU 还存在一些变体:

   <img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e8631eebe3aa6bd48f7e2870d863c7795faa2a4160b39fcaf886b1a1f70539b9.png" />  
2. Maxout
   <img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9e9b88077faaeb2c308feffc244dacce7c87f31f72f328fcfa5ce7dae5384001.png" />

   ReLU 是 Maxout 的一个 special case

   <img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/f9c24a669b20741a32ab1ffcb40f9b96e53bef5dc285310e985cf4d9603a4d2b.png" />

   但是 Maxout 还能表示很多其他函数. 事实上, 它可以表示任何一个分段线性函数. 段数等于一个 group 中 elements 的个数.

   <img alt="图 11" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8aa1cabcd2c7b7b04ca407750c890db5795af96be3d38365aae1dfee543f5618.png" />  

   和 ReLU 一样, 虽然函数有不可导的点, 但是就跟开关一样, 会让网络变成 thin linear network:

   <img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5d65d6ce7373c0e4439819911c9f607c9f3052f29160b67fd8eabe6130fb7eff.png" />  

**Adaptive Learning Rate**

1. Adagrad
2. RMSProp
3. Momentum
4. Adam

详见 [Gradient Descent - New Optimiz~~ation~~ers for Deep Learning](./3-Gradient%20Descent.md##new-Optimiz~~ation~~ers-for-deep-learning).

**Early Stopping**

详见 [Gradient Descent - Early Stopping](./3-Gradient%20Descent.md###early-stopping).

Early Stopping is a beautiful free lunch.

<img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/96317c321c45b2f8406d27d58fc1bdde9f652558b138eddf510a22323b0efda9.png" />

**Regularization**

regularization 部分前面我没太理解. 这次终于搞懂了.

1. L2 Regularization
   <img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/be50a1c7d4fc07f4d7d6a03a2238bf5504de0354265bedfd5239191c61e847c7.png" />  

   是不加 bias 的, 因为 overfitting 只和 weight 有关.
2. Weight Decay
   在梯度更新时:
   $$w^{t+1} = (1 - \eta\lambda)w^t - \eta\frac{\partial L}{\partial w^t}$$

   对于 SGD, L2 Regularization 等价于 Weight Decay:

   <img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/0e95eda89a39c33ed5fb61073b519ce59539da8bd8c35fb44e3781e3526e3413.png" />  

   但是在 Adam 中两者不一样, 通常 Adam 就用 weight decay.

**Dropout**

**dropout 只给 hidden layers**

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/62d76ebb67a6c9c2c9302551a912ad01d7d9999f4fdaaecbc737743c346ebbbe.png" />  

这时网络结构会改变, 变得 thin

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b9f8e7e71f464de34999fa4e09d78ccc42817dae5aa33d639572d69a4da0d3fc.png" />  

在测试的时候不要 dropout, **而且所有的 weight 都要乘以 1-p%**.

为什么呢? 首先 dropout 的作用就像这样, 好像看起来还能说得通:

<img alt="图 18" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/676c666350f8dbafb9aa4f6b185043f8b946a8db4ede2e0a9dc7ad56f63667d7.png" />

但是 1-p% 这个我实在是说不通... 这玩意儿好像没啥理论证明, 或者 lecturer 并没有给. deep learning 就是玄学.

或者 dropout 可能是一种 ensemble, 把一堆东西糅一起. 假如有 $M$ 个 neurons, 那么就会有 $2^M$ 个 networks. 他们就像平行宇宙每个都会 train 出一个参数. 这些参数中会 share. 在测试的时候, 所有模型糅在一起, 正好相当于 weight * (1-p%).

dropout 的 activation function 说是用 ReLU 和 maxout 会好?

## Why Deep Learning?

为什么神经网络一定要 deep, 要有很多层呢?

**Universality Theorem**

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/25e11283c8490a9946f732b8eb465033afd273a084bd34f9edf3368b02965263.png" />

明明一层的 fat & short network 就可以表示所有的 function, 为什么一定要用很多层的 thin & tall 呢?

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/25ed864b0e9cb9f3aad0bb55de2061086adc20e506d41275b09820c1509390b7.png" />  

同 parameter 数量的情况下, fat & short 的 error 要大. lecturer 从 Modularization 的角度进行了解释.

**Modularization**

假如我们要训练一个四分类器, 长发男, 长发女, 短发男, 短发女. 我们都知道实际中长发男是很少见的, 如果我们只用 0 hidden layer 的 network 训练会这样:

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/4d0b6c6afebfacddc2432dcc29b3dde21aeeaa5d7e65988f63c5850cd2c8acd4.png" />  

长发男因为数据量少, train 的效果不好. 但是我们如果用 1 hidden layer, 这一层只是判断长发/短发和男生/女生, 发现这一层的任务比较简单, 因为每个任务都有足够而且对等的训练数据.

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/27bd06f6a90663ae080d98ee7031c9de8e0e8efc209ee45de88968743e661021.png" />  

再接着把 hidden layer 的结果送入到 output layer, 这时效果就不错.

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b1ffbcc9501f89352aa3c53678f402d01643d81cdc812e96c5ae9c7282434be9.png" />  

lecturer 管这个叫 modularization. 其实我感觉就是第一层只处理 basic, 第二层利用 basic 的知识学一些稍微难一点的, 第三层再利用第二层学更难的... 最后就能学到很难的 function. 而且不需要太多 data. 这就是和 fat & short 相比的优势. 这就跟 CNN 一样.

然后 deep learning 还可以取代 end2end 中之前需要人工设计的 blocks, 改为 NN. 还可以实现 complex task. 主要是算力提上去了. 这样大家都用 deep 了.