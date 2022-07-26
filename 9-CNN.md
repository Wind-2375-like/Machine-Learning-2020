# ML 2020: Deep Learning

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [Multidimensional discrete convolution - Wikipedia](https://en.wikipedia.org/wiki/Multidimensional_discrete_convolution)

3. **Extra videos:** None

---

CNN, 图像分类, 非常经典的网络. 如果在图像分类中我们用 fully connected 的神经网络, input 为所有 pixels 展平得到的向量, 假如是 500\*500, 再乘以 rgb 3 通道, 就是 750000, 假如第一层 hidden layer 有 256 维, weights 的数量就是 750000\*256, 这也太多了.

所以 network 需要简化.

## Three Properties of Images

**Convolution:**

- 识别 pattern 并不需要 neuron 看到整张图片. (只看局部图片, 参数变少)
- pattern 可以出现在不同的地方. (识别不同地方的同一 pattern, 可以用同一组 parameters)

**Maxpooling**

- subsampling 不影响 object 的分类. (减少参数)

前两个 properties 需要用到 convolution, 后面的需要 maxpooling.

| <img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/00195bef40702db627284acf40565c81837491bf2a33090662eafd3ec36f298b.png" style="zoom:50%;" /> | <img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9d58960f335aace90e4136c44dd185595bda448fd659cbe428d4b9a38e1819a8.png" style="zoom:50%;" /> | <img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/847754d2a39e37d86199a85d241d257482336925da009ed691aa640a1f5a0fa8.png" style="zoom:50%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

## The Whole CNN

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/735646ce8079c12c9d56fb497250e75757be414d74f47025460b7f34a235bbfa.png" />

### Convolution

卷积的定义:

$$
\begin{aligned}
 \displaystyle y(n_{1},n_{2},...,n_{M})&=x(n_{1},n_{2},...,n_{M})*{\overset {M}{\cdots }}*h(n_{1},n_{2},...,n_{M})\\
 &=\displaystyle \sum _{k_{1}=-\infty }^{\infty }\sum _{k_{2}=-\infty }^{\infty }...\sum _{k_{M}=-\infty }^{\infty }h(k_{1},k_{2},...,k_{M})x(n_{1}-k_{1},n_{2}-k_{2},...,n_{M}-k_{M})
\end{aligned}
$$

而 CNN 中是这样实现的:

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/30c721d08ded9a4d0bdd089d394085172c992746d86c486a97b41f43f5183299.png" />  

先把第一个 filter 贴在 image 的左上角, 对应元素相乘后累加算出一个数代表左上角的值 $\begin{bmatrix}
    1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1
\end{bmatrix} \odot \begin{bmatrix}
    1 & -1 & -1 \\ -1 & 1 & -1 \\ -1 & -1 & 1
\end{bmatrix} = 3$. 再按照 stride 向右, 向下移动, 再对应元素相乘, 生成左上角右面, 下面的值. 最后算出来的矩阵比原来的小.

如果 stride 超出 image 的范围, 应该是补零吧?

其实按照上面的定义, 这并不是卷积, 卷积应该把 filter 上下翻转 180°, 再左右翻转 180°, 再按照上面的方式计算. 其实 CNN 的"卷积"是互相关函数. 但是如果 filter 中心对称, 或者我们就当真正的 filter 是中心对称镜像翻转之后的, 就没有区别了.

跟 filter 越像的值越高, 代表 pattern 越和当前区域契合. 但是有人提出问题, 说图像 scale 不同, 图像的 pattern 的大小也不一定一样, 比如图像的鸟嘴超大, filter 很小, 匹配不上? 所以肯定不能喂入原始的图片, 必然要经过处理.

**Convolution 是简化了的 fully connected 网络**

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/26389ffc930c96c52fd4b3139bf1fb5cf8735f702c7472189790484f9b81f649.png" />  

这张图画的太好了. 把一个 image 分成 36 个 pixels, 展成 vector. 第一次卷积相当于处理了第 1, 2, 3, 7, 8, 9, 13, 14, 15 元素. weight 颜色对应 filter 1 不同颜色圈住数字的大小. 然后 filter 1 移动, 处理第 2, 3, 4, 8, 9, 10, 14, 15, 16 元素, weight 颜色对应 filter 1 不同颜色圈住数字的大小. 它不仅去掉了一些权值 (比如第 4, 5, 6, 10, ... 元素没有和圆圈 3 连接), 而且共享参数. filter 有多少个数, 就有多少参数. 极大节省参数个数.

### Maxpooling

maxpooling 就是先 group, 再取最大.

<img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5a36d2a7370bf4a578f5f87f0f0e31b67d41cc5d789ef6662dc0ba16afd1c493.png" />  

这就是 maxout, 是可微的.

最终会得到更小的图片, channel 数量等于 filter 数量.

<img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/f8ed82122bb9356345fff891c712e92e00de85a6c5a02c62dc9f27dc32595046.png" />  

有一个问题是在下一层 convolution 时, 因为上一层算出多个 channel, 下一层 convolution 怎么算呢? 其实 filter 是有深度的, 比如上一层算出 4 channel (4 层) 的 image, 下一层 convolution 时 filter 也要广播叠 4 层, 直接和 4 channel 图片对应元素相乘.

### Flatten

<img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/593b750b8b460a0fbebc08ebcb69e82361516cccf22e1683d5701d49f84a58b6.png" />

## What Does CNN Learn?

我们需要强行解释 CNN 学到了什么. 一种可解释性方法是固定训练好网络的参数, 把 input 当作变量. 研究网络哪个部分, 就算出什么 input 会使那一部分最大.

比如我们想研究不同 filter 的作用, 就可以固定网络参数, 更新 input, 求出使 filter output 值最大的 input, 数学语言表示如下:

$$
\text{k-th activation: }a^k = \sum_{i}\sum_{j}a_{ij}^k
$$

接着定义目标函数, 用 gradient descent 求出最优 input.

$$
x^{*}_{k} = \argmax_{x_{k}}a^k
$$

发现不同 filter 确实算出不同 pattern.

<img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/a1f549d32d6505a5ab33fa48e2331a781befe9a085efddd1c21981c8688a5c3b.png" />  

而且浅层学的是局部 pattern, 深层学的是全局 pattern, 由简到难, 跟 modularization 一样.

<img alt="图 11" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/74e49ea70b3f03695c2dc1a4ccd679528eea1935af5ab960d77f6b6c7309fb8d.png" />  

当然也可以研究 fully connected 网络的 neuron 的作用, 目标函数变成:

$$
x^{*}_{k} = \argmax_{x_{k}}a^j
$$

<img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6d84c62f18f93c4a47e2e36e340f5209cba2e8c0690fc3ec4bc9d056b7562193.png" style="zoom:50%;" />  

我们干脆试试研究 output 层:

<img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/940c23489d5c5b3949f0e860da2109530b3768c32acfa6f7b23ae4f571cd350d.png" />  

怎么跟坏掉的电视一样. Machine Learning 学了个 P.

有人试着稍微加 constraints,若白色太多 (255), 就不靠谱, 稍微好了一点(?)

<img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/c6533c428d1a3b4c4f3e1067f1ec194095a3525723fdfd341c108f3963df46e7.png" />  

## Other Application

**Deep Dream:**

这东西是从上面的可解释性来的.

<img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8de7038ad6132701761d1ec10cb551ab3af0ba50d7c71d960c6d7affd3c25bb0.png" />  

给 train 好的 CNN 喂入图片, 新的目标函数是把 filter 算出的结果正的更正, 负的更负, 再反推最优的 input. 然后就 san 值狂掉了.

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/51901ad61d77bf183b4df5e81d444bb96fdaaaf78194cc0f74699a03cbc5acdf.png" />  

**Deep Style:**

这个就像开滤镜, 给一张图片赋予另一个风格.

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e958c0b5c040a68ac3a02eb1292295a41aa5b593c75e64dd91961cdff1e0fa15.png" />

image 送入训练好的 CNN 中提取到 content. 然后把呐喊这个图片送入 CNN 中计算 output 的 correlation 代表 style. 然后对这个 CNN, 希望 content 接近 image, style (correlation) 接近呐喊, 反求 input.

**Playing Go:**

<img alt="图 18" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/4be027866c5593a280cc6c40e28bce2335aee02002df0a25c34b843d1ae82cc2.png" />  

把棋盘视作图片进行 supervised learning. 为什么可以用 CNN 呢? 比如 Alpha Go 就用的 CNN (但是是 RL)

为什么 CNN 能 work? 因为围棋具有 CNN 的两点性质:

<img alt="图 19" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/0ebb4119341d0a4cb9d142738e65647c89ed9775d81199e6d75fdf6a13f25903.png" />

但是第三条...?

<img alt="图 20" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b331403a3a2c1b1c6dc29c82b6a9212edae3ed7eb98344ba64a60aef00d91593.png" />  

原来 Alpha Go 没有用 maxpooling...

**Speech:**

<img alt="图 21" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5ba203be90e69ff23c1d26f139176deb24a4bc60f83024fa863ecff8c4d5e32e.png" />

为什么 filter 不沿着 time 呢? 因为 CNN 好像不太能处理好动态的东西. 所以还是借助 seq2seq 模型后续处理吧.

**Text:**

<img alt="图 22" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/584b3ae14b76e0df77a3c6577e9e285e1fe4dbd99b7f86701c8443bb0b59895f.png" />

CNN 还能处理 text 吗... 但肯定不如 seq2seq.
