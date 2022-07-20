# ML 2020: Classification

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:** None
3. **Extra videos:** None

---

Classification, 就是让 machine 学一个输出为离散的函数 $f(x) = n$.

## Some Attempts

怎么找函数呢, 最开始有两种尝试. regression 和 ideal model.

### Regression

可不可以把分类问题变为回归问题. 比如二分类, 正类 $f(x)=1$, 负类为 $-1$. 然后训练出一条直线. 在预测的时候以 $0$ 为分界.

但是这个方法会因为离群点造成 error 的增大.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/058d413acbbada67c1f0d861c3efe185aa7606d6ae818355179c7e763f72232b.png" style="zoom:50%;" />

而且我们用人为的方式定义了 class 的远近, 但是实际上人为定义的可能是错的.

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/87e7803f041431f69c64c7ecce6f17cebe31879e91345fcab42a1f8a373b1087.png" style="zoom:50%;" />

### Ideal Model

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e819c9e91f2ac59d572f17908f25740f2ff1e3cb175ebe38c1155afe95236e4a.png" />

## Generative Model

这个 model 的名字之前令我有点陌生. 但是 Latent Dirichlet Allocation (LDA) 用的大概就是这么个东西. generative 的 model 是确定的, 可以直接求出来.

### Three Steps

**Step 1**

第一步找一个函数, 本质上就是找一个概率分布. 数据 x 属于 class $C_i$ 的概率. 用到了贝叶斯公式:

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1b75ef0991e524c5c8b5d08cc89cb4a336cd26151c89fc7413883994689ffc07.png" />  

$P(C_i) = N_i / \sum N_i$

$P(x|C_i)=f_{(\mu, \Sigma)_{C_i}}(x)\cdot\Delta x$, Δx 可以分式上下消掉, 这样我们可以用概率密度函数表示概率. 这里我们假设 $C_i$ 类下数据点的分布为高斯分布:

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/7838ad9a8535e9804d8bf39d3dd722d7f17bacaad4d18049d967097cd5d50dcc.png" />

这样我们就找到了函数.

**Step 2 and 3**

这里函数的 $\mu, \Sigma$ 是需要优化的参数. 我们需要找到最好的, 怎么定义"好"呢? 需要用 Likelihood.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/788b73e38f15e9b9d59cd949b86b5db904ba8381e3350441435f372bf3befe2d.png" />  

这里 x1 到 x79 都是 $C_1$ 类的数据点. 我们在求 $C_1$ 类的高斯分布是什么样子的. 然后我们发现求出来的 $\mu$ 和 $\Sigma$ 就是 the best 了.

然而后面发现这样很烂. 于是假设不同类的 $\mu$ 不同, 但是 $\Sigma$ 一样, 于是有:

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/17c4248c6888126303e0ab4b0fcf3fcab584f680c6605514ba7bb4e1527fde3a.png" />

这样一搞 boundary 变成 linear 了, acc 也上去了.

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/d4a257efe6c07aaa7e9caa2fb4e71669b3be6a2bec728cf0689a5287607409d4.png" />

### Probabilistic Distribution

这个 distribution 不一定非要用 Gaussian 的.

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/2c745b5e658b3c65189e272dc2c62a1bb3c3ea75fb6437a73ff91a56a7209291.png" />

### Modifying Posterior Probability

我们对后验概率一通变换, 发现变成这样了:

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/79f14cb8f51c2b2aa8a6252ed9ba8fd3bc1db1e677199fc8510c1131572794f8.png" />  

这个 z 等于什么呢? 一通变换发现变成这样:

<img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/623a669634824bc13b170de845b8292953e3cbd2719b652004fc872aeb4c4b11.png" />  

所以现在 $P(C_1|\bm{x})=\sigma(\bm{w}^{\mathrm{T}}\bm{x}+\bm{b})$ 了. $\bm{w}, \bm{b}$ 可以直接通过求 $\mu, \Sigma, N$ 算出来. 同时这里也解释了为什么我们求出一个线性边界的原因. 我们把它化成 sigmoid 套一个线性函数了.

## Logistic Regression

等等, 还有必要直接求吗? 这不是跟 linear regression 很像吗? 用损失函数, 再梯度下降, 迭代出来 w 和 b 不是也挺好吗. 而且 generative 的方法各种矩阵求逆, 多慢啊. 所以 logistic regression 就出来了. 下面介绍 3 steps.

**Step 1**

<img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/cdb14c06ac4a61fc5d6e37b55df7081d879b2add1cc91f76ed895b2a40296adb.png" />

**Step 2**

同样用 likelihood. 进行一下变换.

<img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6f5e89cbc75e114be3ac9d76249dccb38c63b54bb5d0606901893968e936b041.png" />

变成交叉熵 cross entropy 了.

<img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5e22276cb341f8d46603925c43074aa4a2d4f8a045be89fc0584c228f87054d2.png" />  

在变成一个个 batch 的时候, 乘法变成 `np.dot`, 是内积.

**Step 3**

$$
\sigma'(z) = \sigma(z)(1-\sigma(z))
$$

再用上链式法则, 得:

$$
\frac{\partial L(f)}{\partial w_i} = \sum_{n}-(\hat{y}^n-f(\bm{x}^n))x_i^n
$$

$$
\frac{\partial L(f)}{\partial b} = \sum_{n}-(\hat{y}^n-f(\bm{x}^n))
$$

$\hat{y}^n-f(\bm{x}^n)$ 为 error, error 越大, 更新越大.

## Comparison with Linear Regression

<img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/89b2bf03355a824364505c37ab1dbfc2815e5cd8ab732b4c7345ed466bf5eb99.png" />  

发现 step 3 logistic regression 的梯度下降函数和 linear 的完全一样. step 1 差了个 sigmoid. 但是 step 2, 为什么 logistic regression 不用 square error 呢?

<img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b0af36e86902fd7c7db79b69fdf60e0915439b03eca4763106251b6e5e91585a.png" />  

发现用 square error 无论 close to target 还是 far from target, 都没有梯度, 那这函数更新不了了啊, 就寄了.

## Comparison with Generative Model

虽然他们是同一个 function set, 但是 Generative Model 做了 Gaussian 的假设, 而 Logistic Regression 完全是从数据中学习, 没有假设. 但是这个假设不一定是准确的, 所以一般情况 Generative Model 比较烂.

但是 Generative Model 还有如下好处:

- Less training data
- Robust to noise
- Priors and class-dependent probabilities can be estimated from different sources.

关于第三条, lecture 中解释, 比如语音识别. 求语音 $x$ 被识别为某个字 $C_i$ 的概率 $P(C_i|x)$. 我们需要知道字 $C_i$ 中语音 $x$ 出现的 prior probability $P(x|C_i)$ 和 class-dependent probability $P(C_i)$. 前者可以按照上面的方式从数据中求出来. 但是后者, 因为语音数据量少, $P(C_i)$ 估测不准, 于是用文本数据中汉字的出现概率作为 $P(C_i)$. 这时便使用 generative model.

## Multi-class Classification

有时我们需要多类别分类, 比如 $K$ 类别分类, 怎么办呢? 这时我们使用 softmax 函数. 首先

$$
z_i = \bm{w_i}^{\mathrm{T}}\bm{x}+b_i
$$

注意这里没有 sigmoid. 接着用 softmax 公式:

$$
y_i = \frac{e^{z_i}}{\sum_{k}^{K}e^{z_k}}
$$

这时 cross entropy 变为:

$$
\sum_{n}{-\sum_{k}\hat{y}_k^n\ln y_k^{n}}
$$

这里 $\hat{y}$ 是一个 one-hot vector

<img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/120e3a21be71915c669d71ae53ac92d2ab45a859e0cd9e61a27eeb061c5b19bb.png" />

对于二分类, softmax 和 sigmoid 本质上一样. 一个是 $\displaystyle\frac{1}{1+e^{-(\bm{w}^{\mathrm{T}}\bm{x}+\bm{b})}}$, 另一个是 $\displaystyle \frac{1}{1+e^{-(z_1-z_2)}} = \frac{1}{1+e^{-(\Delta\bm{w}^{\mathrm{T}}\bm{x}+\Delta\bm{b})}}$. 没差别.

## Limitation and Neural Networks

logistic regression 只能给出一个 linear boundary, 这是它最大的缺陷. 比如异或, 在 (0, 0) 和 (1, 1) 为 0, 在 (0, 1) 和 (1, 0) 为 1.

<img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/24e31d29c4c1c4b79387da660ed04bec7fa56f36dce36d9dc3e5aaa8c898d290.png" style="zoom:50%;" />  

怎么办呢, 一种方法是 feature transformation, 但是这时人找的, 并不算机器学习, 人得累死.

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/142f84d7f41c32eb0fe9eb72ee266901ca8ed8a173658e1e2c998f05a93003a8.png" />  

我们可以再叠一层, 像这样:

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/c81f6f19fc50ae86fa7b6be0d989a1a9ad3c4ec394da77371ec6f4bdcc47df3c.png" />  

这样子最后发现效果跟上面是一样的.

这个东西就是就是 Neural Network. 就是一堆 $f(wx+b)$ 叠在一起. 所以接下来就可以到 deep learning 了.
