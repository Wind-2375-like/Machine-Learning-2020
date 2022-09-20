# Recurrent Neural Network

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [Recurrent neural network - Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network#Elman_networks_and_Jordan_networks)
    2. [Partial recurrent neural networks Fully recurrent NN have each neuron... | Download Scientific Diagram (researchgate.net)](https://www.researchgate.net/figure/Partial-recurrent-neural-networks-Fully-recurrent-NN-have-each-neuron-connected-to-all_fig2_228344631)
    3. [What is LSTM , peephole LSTM and GRU? | by Jaimin Mungalpara | Nerd For Tech | Medium](https://medium.com/nerd-for-tech/what-is-lstm-peephole-lstm-and-gru-77470d84954b)
3. **Extra videos:** None

---

## Why We Need RNN: Slot Filling Example

在 slot filling 的任务中, 我们可以仅仅使用 NN 做到输入句子中一个单词的 embedding, 输出 slot 各类型的概率分布.

但是, 这种方式无法考虑到上下文的因素. 比如 arrive Taipei 和 leave Taipei 中的 Taipei 虽然都是同一个词, 通过 NN 只会输出相同的结果, 但是一个是 destination, 一个是 place of departure. 因此我们必须考虑到上下文 arrive 和 leave 两个词.

至少我们希望, Taipei 这个词的输出和前面的词有关, 这一时刻的输出取决于前一时刻, 这时, 我们就需要 neurons 存在 memory. 这也就是为什么 RNN 被提出来.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ae53512e84f6df6239f14d6025576dd09a7c107a0b97617a3111f32ea3b468a0.png" />  

## RNN Architectures

RNN 有很多变种.

**Elman Network**

这个是比较常见的, 出现在各个 RNN 的例子中, 我们便以这个详细举例.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/85badedf72d6e6489dfa3ba5832829a8c29bda9f3a49ba41970c1b6ef369486c.png" />  

网络分成 input, hidden, 和 output 三部分 layer, 一般只用三层网络. Elman Network 就是把 **hidden layer** 中 neurons 的值存储到 memory 当中, 再在下一次输入 (新的 x1, x2, ...) 时, 将新的输入和存储在 memory 当中的向量同时经过线性变换送到 hidden layer 中.

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/99f7cf179849a7d9c9b4536e73a52e0cc552dba3b6cf9e07d1ed0e4097ad3153.png" />  

可以用 RNN 生成一个word、sentence、图片等等。

用公式表述的话就是这样:

$$
{\displaystyle {\begin{aligned}h_{t}&=\sigma _{h}(W_{h}x_{t}+U_{h}h_{t-1}+b_{h})\\y_{t}&=\sigma _{y}(W_{y}h_{t}+b_{y})\end{aligned}}}
$$

- $x_{t}$: input vector
- $h_{t}$: hidden layer vector
- $y_{t}$: output vector
- $\displaystyle W, {\displaystyle U}$ and ${\displaystyle b}$: parameter matrices and vector
- ${\displaystyle \sigma _{h}}$ and ${\displaystyle \sigma_{y}}$: [Activation functions](https://en.wikipedia.org/wiki/Activation_function)

**Jordan Network**

Jordan 网络和 Elman 网络几乎一样, 只是把 **output layer** 存到了 memory, 然后在下一次输入的时候同 input 一起送到 **hidden layer** 当中.

$$
{\displaystyle {\begin{aligned}h_{t}&=\sigma _{h}(W_{h}x_{t}+U_{h}y_{t-1}+b_{h})\\y_{t}&=\sigma _{y}(W_{y}h_{t}+b_{y})\end{aligned}}}
$$

**Fully RNN**

Fully RNN, 就是把所有 neurons 的输出都存到 memory 当中, 再线性变换送回到所有 neurons 的输入.

**Deep RNN**

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8ff68d55814bb345c88fc853c53f983377994c641ce3195ab01a8346ef65341d.png" />  

**Bi-directional RNN**

双向 RNN 也好, LSTM 也好, 并不是说让 cell 内部可以同时传入前一时刻和后一时刻的输入, 而是把两个 RNN 拼在了一起, 一个前向, 一个反向罢了.

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/96d32f683a01afc8482ed40328c25e6d9dcc7e7adef9d20ff997294d5f48ae82.png" />  

## Most Famous (?): LSTM

RNN 中最有名的变体当属 LSTM 了. RNN 中只有一个输入, 里面装了一个 memory 罢了. 但是 LSTM 需要 四个输入, 除了正常的输入外, 还需要三个门控信号, 分别控制 input, "forget"(memorize), 和 output.

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/00dbdeb1c17bd2bbea7b2f813386088a7a2d0fc73d00a4aec453bc7f8a45feed.png" />  

### Structure and Formulation

<img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e441a52f5424cb89ae4b3b44d60d1926075b48bcf4769e3495f1b4036568fb1e.png" />  

这个便是 LSTM 介绍中最常用的图片了, 虽然很紧凑, 但是我在第一次学的时候只能照着这个图死记硬背顺下来他在干什么, 个人感觉没有李宏毅的清晰.

<img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e3ab8d10e30f7b79822fb606aa4f3a6bb2533038defbda429319dc549b6fa085.png" />  

这里面的 $z$, $z_i$, $z_f$, $z_o$ 分别是原始输入 $x$ 分别进行四个线性变换 $z=wx+b$, $z_i=w_ix+b_i$, $z_f=w_fx+b_f$, $z_o=w_ox+b_o$ 得到的. 这张图中 memory c/c' 的传递代表了上面那张图的 cell state 的传递. 然后这张图少了 output 和下一时刻 input concatenate 的过程, 在此补上.

<img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/d839c4d09797a5f46908d41e24afd3846c7981f1fca61cbd8179d4613ec004e6.png" />  

我们从这张图中可以看出, 一个 LSTM 的 neuron 还是输入了一个 x 的向量, 输出一个 output 的数字. 只不过, 这个 neuron 比较复杂, 里面有各种门控开关, 有 memory, 输出和输入在内部之间也连了起来. 而且一个输入被复制成四份进行了四个线性变换, 代表了参数量的增加.

### A Toy Example

我强烈建议过一下这个人体 LSTM, 我感觉没人会具体手算一个 LSTM 并且说明其作用的... 不得不佩服台大授课的敬业态度...

| <img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/3beee445714fb0c045d4bdfa8342d4f302fdda45d79c28eb4b85a0778d639ab3.png" style="zoom:50%;" /> | <img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/3de7b21ba2ace1a2c98bf81d13d72e9fa32574c3a97da571295a032c24205f00.png" style="zoom:50%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

左面这张图代表了 Lecturer 希望 LSTM 做到的事, 右面的网络代表了他如何设定 weight 使 LSTM 完成他想要做到的事情. 这样我们对 LSTM 在干什么一目了然.

### RNN, LSTM, and NN Cells

<img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/167622073aa2cae716117e368bca6dbd34a66aadf2843661a78d9434adaf521b.png" style="zoom:50%;" />  

这是一个 NN 的基本结构. 每一个 neuron takes vector input, 经过线性变换和 activation function, 输出一个数字, 一层中排多个 neurons 就能输出一整个向量.

RNN 每一个 neuron 仍然输入一个 input vector, 输出一个数字, 但是 neuron 内部的结构不太一样, 多了 memory. 这样这一时刻的输入得到的结果取决于前面的时刻了, 不再是时不变(原谅我用了信号与系统的术语). 管他什么结构, 我们仍然可以用这个图表示 RNN 的 cell.

LSTM 则是一个 neuron 要处理四个经过线性变换的 input, 里面还有 memory, 更复杂了, 参数也更多了 (忽略输出和 input 的 concatenation)

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/2265e6700b5aa6b9b08e8aa25706af58130aeb1e6482759ddfa652d191af710f.png" />

### Full LSTM (Peephole and Deep LSTM)

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ef37752deed4c5ecb2635b21ef45be837382c5f1bdc4a5099b166b99ace0fef7.png" />  

整个 LSTM 就是这样, 这里还有一个叫 peephole 的东西, 就是把 cell state 也同输出, 与下一时刻的输入进行 concatenate.

然后你还可以 deep, 叠很多层, 然后你就会头大晕掉. 这时候调包 pytorch 就行了...

<img alt="图 18" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ac63b446c57925a0d3ce4f8b7b5a3591de20fa75232d4418d4f60f60d5760064.png" />  

## Training

training 这里没有详细说明. 仍然是用 output 和 label 之间计算 loss, 然后 backprop, 不过 LSTM 和 RNN 还有时间这个参量, 需要用到 Backpropagation through time (BPTT)

## RNN's Rough Error Surface

为啥要用 LSTM 的一个原因就是 RNN 具有一点问题. 梯度在有的地方很平, 所以 param 几乎不更新, 而有的时候很陡峭, 所以更新飞了.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6064aca0a605a246727b4e938c6f46f308cbfd0b34cab13566c3b92ccbbf5a71.png" />  

为什么呢, 因为 RNN 的输出相当于次数很高的幂函数, 导致函数在 0 到 1 之间几乎都是 0, 梯度几乎为零 (gradient vanishing), 1 之后特别大 (gradient explode), 在 1 前后梯度变化十分剧烈. 导致根本没有办法正常地更新参数.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/476c99b5a65502543424c8088d9ea9ecf88b21008b13a83ef60c778ac7ccb231.png" />  

怎么办呢? 针对 gradient explode, 一种方式是 clipping, 让梯度大于一个值的时候就不再继续大下去, 截断了.

针对 gradient vanishing, LSTM 就出现了. LSTM 保证 cell state 只要在 forget 门开的时候就能加进去, 不会连乘什么参数, 导致只要 forget gate 不关, influence 总是能存在, 是相加的关系.

然后 LSTM 不是参数太多了吗, GRU少了 1/3 的参数, 性能却相差无几, 这里也没有介绍. 后面还有 Clockwise RNN, Structurally Constrained RN, Vanilla RNN initialized with Identity matrix + ReLU activation function. 管他呢反正都是调包, 而且现在好像全都上 transformers 了...

## Applications

application 一笔带过.

**Many to One:**

输入一个 sequence, 输入一个值. 这个就在最后的输出分类就行了

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ea8197170ce3f94e7943dd6a5a637d293ea4c6c70d5dc3cd2a74bca1a1dcb5be.png" />  

**Many/One to Many:**

输入 sequence 或者值, 输出一个 sequence, 长度可以不一样. 首先输入的 sequence 所得到的输出并没有用, 直到最后一个元素输入进去, output 目标 sequence 的第一个元素. 将这个元素作为下一时刻的输入送进去. 直到输出一个 EOS 符号 (end of speech, 人为定义) 为止.

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9dd34793b4d887fc868ecfdd171a9ac973dcdcef11c5aa6e3abfa6e86e6e3d86.png" />  

然后 RNN 类模型和 seq2seq 模型有很深的联系, 尤其是 NLP. 懒得整理了.

## Deep Structured Model

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5986551c5f4fddfb43721eb5351213b1ee0c58b6174984907975ba9b792d0ebf.png" />  

就很多人现在都觉得尽管 RNN, LSTM 等 NN 模型在 efficiency, error, explanability 等方面就是玄学, 相比之下传统模型就 make sense. 但是 RNN 他们可以 deep, 就能学出很复杂的东西, 不需要太多的假设. 所以现在不少模型都会把他们结合在一起. 就是用 deep 的模型干一些脏活, 学出原始的 embeddings, 然后送入到更 make sense 的 structured model 当中, 得到更好的 performance.

<span class="heimu" title="=.=">大概是这样吧, 我不确定.</span>
