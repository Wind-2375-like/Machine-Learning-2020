# Attention Is All You Need

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [浅析Transformer训练时并行问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/368592551)
    1. [Transformer家族4 -- 通用性优化（Universal-Transformer） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/178710534)
    1. [nlp - What is the positional encoding in the transformer model? - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)
    1. [deep learning - Can the decoder in a transformer model be parallelized like the encoder? - Artificial Intelligence Stack Exchange](https://ai.stackexchange.com/questions/12490/can-the-decoder-in-a-transformer-model-be-parallelized-like-the-encoder#:~:text=on this post.-,Can the decoder in a transformer model be parallelized like,sequences during a testing phase).)

---

看完 Hung-Yi Lee 的 RNN, LSTM, Transformer 的 ppt, 一目了然. 我就想问, 国内能用心做 ppt 的老师有几个呢? <span class="heimu" title="=.=">(如果你愿意, 那就大陆吧)</span>

## What is Transformer?

看了这个题目, 都知道说的是 Transformer, 他是什么呢?

Transformer 是一个 seq2seq 模型. 有 encoder 和 decoder.

特别之处是具有 **self-attention layer**. 他的作用是:

- 可以**并行化**处理, 像 CNN 一样, 比 RNN 要好
- 可以照顾到一段 seq 的**全局信息**, 像 RNN 一样, 比 CNN 要好 (虽然 CNN 也能, 但是需要很多层, 太庞大了)
  - <img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/62fcc98de82cbfab2a89dc7ec77bdf73f4bdc27da5a4fc7d1fec8ea01a637ea8.png" style="zoom:33%;" />

这个神奇的 self-attention layer 是怎么做到的呢?

## Self-attention Layer

第一步, 对序列中每个 token 的 input 生成的 embedding 计算 query q, key k, 和 value v.

<img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/60df5aa6b128c1bc3adb776f534f44f7110bab2e1f84338d1d15f9f008b1cfd5.png" style="zoom:33%;" />  

第二步, 每个 token 的 q 跟所有 token 的 k 做 Scaled Dot-Product Attention (包括自己对自己), 再 softmax, 算出这个 token 对所有 token 的权重.

<img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/d1fcfbb142dd880bc508b402fbb04271bb35a253e7a8b22db23c342c6b78400b.png" style="zoom:33%;" />  

为什么要 scale? d 若太大, 内积太大, softmax 接近 1, gradient 小 (跟 sigmoid 一样), 收敛慢.

第三步, 用 这个 token 对所有 token 的权重 对所有 token v 进行加权求和, 算出这个 token 的 embedding. 此时这个 token 的 embedding 包含了整个 sequence 对其施加的影响.

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/d24749453503b81de9c14acca32f2bdb5761f3da8193ce66088ef4378a98f971.png" style="zoom:33%;" />  

如法炮制, 算出第二个, 第三个, 第四个 token 的 embedding b2, b3, b4.

## Multi-head Self Attention

Transformer 需要多头注意力, 这是什么呢? 原来就是对每个 token 的一组 qkv, 拆成了多组. (每组的维度更小了). 只有同一组的 qkv 之间才会互相 attention, 一个 token 得到多组 b. 最后用一个 $W^o$ 再变回一个 b.

| <img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/d77abb924f0a2025ecd8bf243b3db0f06314f71d0693d6443d1f425bf4738738.png" /><br>第一个 head 计算 | <img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b9c4d4b8099c40d319f667b474b97ff51bfb1e6e9d2cbcc947ae0e17a51c4a10.png" /><br> 第二个 head 计算 | <img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/e9bf80ed8839f9fde1bd1c69942e1eebf20a665319d5fbe18b90f85b721aa2ba.png" /><br> 两个 head 通过线性变换拼在一起 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

为什么要多头? 有的 head 管 local, 有的 head 管 global, 各司其职.

## Parallelization

transformer 可以并行. 也就是同时输入 a1, a2, a3, a4, 可以同时得到 b1, b2, b3, b4. 而不是输 a1 得 b1, 再把 a2, b1 输入得到 b2 这种.

这就要用到神奇的矩阵乘法. ppt 写的特别清楚.

首先 qkv 的计算可以同时完成.

<img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6de13b5974ddbca32110f8d83191872a10e54e55e504208767c34d4a9ecfd55c.png" style="zoom: 33%;" />

其次, 不同 token 的 k q 之间 attention 也可以用一个矩阵乘法完成.

| <img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/19513689c440a4875578df87ee39cba75ad01cd74f6004b2f5eb641555403a0d.png" style="zoom:33%;" /><br>第一个 token 的 q 和所有的 k 计算权重 | <img alt="图 11" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1942598f62469a8970aefb34425fdd9ce90cb9be13ba6f348887fd02a0954c87.png" style="zoom:33%;" /><br> 所有 token 的 q 和所有 token 的 k 计算权重, 然后 softmax |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

最后一步加权求和也是一步矩阵乘法.

<img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ad8a929915fcbb566bb0b82423859d8c20e88f7266694a0a6692f42e8faf1a22.png" style="zoom: 33%;" />

Multi-head 呢? 第一步 qkv 的计算一模一样, 只不过计算完之后, 对 qkv 的 feature 维度拆开, 让每个 d 维 vector 拆成 H 个 d/H 维 vector.

第二步, 就要用到张量了. 原来, 是 (4, d) 维度的 $K^{\mathrm{T}}$ 和 (d, 4) 维度的 $Q$ 进行矩阵乘法 (这里 4 是图中的 token number). 张量就是把不同 head 的 K Q 叠在一起成一个立方块. 维度 (H, 4, d/H) 和 (H, d/H, 4) 之间进行张量乘法, 得到 (H, 4, 4) 的张量. 每一层是 (4, 4) 的矩阵, 代表每一个 head 的 attention score. 计算复杂度. 矩阵乘法的复杂度是 $O(4^{2}d)$, 张量乘法的复杂度是 $O(H*4^{2}*\frac{d}{H}) = O(4^{2}d)$, 复杂度不变.

第三步也是一步计算的张量乘法, 得到一个 (H, d/H, 4) 维度的张量 O. 把他 reshape 成 (d, 4) 维度的矩阵 (不同层沿着 feature 的方向拼在一起), 经过一个矩阵乘法 ($W^o$) 得到 b1, b2, b3, b4.

论文里面的公式和 ppt 不一样, 因为向量, 维度的表示不同. 但本质完全一样. 看看论文, 按照论文的公式再重新推一遍矩阵化公式.

用了矩阵, 就可以并行计算, 而且可以用 GPU 加速.

## Positional Encoding

self attention 现在这样有顺序相关的信息吗? 发现没有. 需要给 self attention 的每个 token 加上一个顺序的信息. 注意是**相加**.

<img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/61f0e5dfaee4738770d6b7c4673f8e29f04a0d89ea9e41d339c1e75a6f9f1da5.png" />

这里 a 用来计算 qkv. 前面的 x 是这个 token 的 one-hot 表示. 先通过 embedding 变成一个低维向量. 然后对 a 相加一个 position encoding e, 这个 e 在每一个 position 有一个唯一的 encoding.

为什么相加而不是 concat? 换一种方式理解就是 concat. 我们对 one-hot 的 x 和 one-hot 的 p 进行 concat, 再 embedding. 就得到了 a 和 e 的相加.

<img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/2989815de8c3ebe2534a4899faf9e39807af34692032a0da3cff8d08eaad72f8.png" style="zoom:33%;" />

这个 encoding 是可以 learn 出来的还是手设的? 都试过, 决定还是手设. 怎么手设呢? 论文中给了公式:

$$
\text{PE}(pos,2i)=sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\\
\text{PE}(pos,2i+1)=cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

## Transformer Structure

seq2seq 的 encoder-decoder 结构别忘了, transformer 就是这种结构, 只不过里面的 RNN LSTM 换成了 self-attention layer.

<img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/79dd8e72e1e1063e84b47e4b6743a6ffd1f25092fa738320cd0870570a88ec83.png" style="zoom:50%;" />

先看 encoder. one-hot inputs 经过 input embedding, 和 position encoding 相加, 经过 multi-head attention 得到 output. 这个 output 和送入 attention 之前的 embedding 进行相加和 **layer norm**. 接着我们经过一个 **feed forward** 层得到输出, 输出和进入 feed forward 层之前的输入再相加和 layer norm. 这是一层. encoder 是 deep 的, 叠了 N 层相同的结构.

什么是 layer norm? 先看 batch norm, 就是对一个 batch 里面同一 feature 不同数据之间进行 norm. 而 layer norm 是对同一数据不同 feature 之间 norm.

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/74d152f2a415fe11c9273a203780a1e368855eb6c8f0fd2820bbd5e926f2607d.png" style="zoom:33%;" />  

为什么要用 layer norm? 众说纷纭. 李沐说是稳定 $\mu, \sigma$.

feed forward 就是个普通 NN, 起了个 position wise 就是只针对一个个词设计 NN, Time-Invariant. 名字比较 fancy.

接着我们看 decoder. outputs (shifted right) 是什么意思? 想想 decoder 怎么 parallel? encoder 自然可以, decoder 每一步的输入是上一时刻的输出啊, 不能 parallel, 所以 transformer 用到了 teacher forcing, 把 ground truth shift right 送入 decoder. 当然 test 的过程就不能 parallelize 了...

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/b53fb01cd3c129b280224fd5b3f43717b0d37c206446fdd95543242dded927f2.png" style="zoom:50%;" />  

接着经过了 masked self-attention. 这是啥? 想一想 decoder 还是一步一步输进去的, 所以当前步只能看见前面步的信息, 所以当前 token 只能和前面的 token 进行 attention. decoder 不能超前偷看啊. 具体实现呢? 就先正常按照 self-attention 的步骤算 qkv, 算出 α, 得到矩阵 A. 接着把左下角变成 -inf 再 softmax. 这样所有 token 对后面 token 的 attention score 都会变成 0.

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ccf0742f3c9f0e730d7a307d4f8cc54d30c5902e75b3d4f7c763b68fafde9a1f.png" style="zoom:50%;" />

接下来通过另一个 multi-head attention 接 add and norm. 这里这个 multi-head attention 的 q 跟 k 是 encoder 同一层对应 token 的 q 跟 k. 而不是用前面 masked multi-head attention 的 output 生成的. 不然 encoder 和 decoder 不就完全独立了, encoder 的信息就传不到 decoder 了.

最后还是 feed-forward 接 add & norm. 然后叠 n 层. decoder 最终的输出经过线性变换+softmax就可以 classify 了. 这就是 transformer 的结构.

论文还提了 embedding (input, output, softmax) 是共用的, 而且算完乘以 $\sqrt{d}$, 查了半天也查不出什么道理, 也没想出来, 这就是玄学.

## Attention Visualization

Attention 本身很适合可解释性. Transformer 也一样.

| <img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/0ff01c4dc7a4dcc2d06e1baf283498a4619457adce6f70f7f6142768f44c9058.png" /><br>it 指代 animal | <img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8412b1d7fc11f3892eed8d6fe140d1577d79c7478be9e77c5b30a28ca46fb7c7.png" /><br>it 指代 street |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

再来一个例子, 下面是不同 head 关注到的情况.

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/87a29bdc63ee7ade8aa2999c8baf8a45d2b605ce83c77239b94b2b727ec2d447.png" style="zoom:50%;" />

上面的 head 关注远一点的信息, 下面的 head 关注 local 的信息.

## More Transformers

这一部分是助教课程的一部分. 他的 slide 标题是 New Architecture.

有的人是乱叠乱 tune model 的, 这一部分贡献了很多的表情包, 大家可以自行收藏.

助教给的建议是: 多看 SOTA 论文别自己乱想. 那我觉得现在 SOTA 都得小心. 其实这还是一个没有答案的问题.

助教还给了 new architecture 的几个方向:

- improve performance
- extract better feature
- generalization
- less parameters
- explanation
- ...

### Sandwich Transformer

更换 feed forward (简称 f) 和 self-attention (简称 s) 的顺序怎么样? 不行.

那我们试试别 sfsfsfsf..., 而是 sss...sfsfsf...fff, 发现居然可以.

我们管 sss... 和 ...fff 这部分是 sandwiching coefficient. 就像面包片一样. 然后中间的部分是肉.

<img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/123087e7bba92c8fb745fa10b1776bb2c8446e38918467ca9ed15140161abb55.png" />

测试了一下相同 parameter 的情况 (s 是 f $\frac{1}{2}$ 的参数量), 发现全是面包片不行, 全是肉也不行.

### Universal Transformer

这里参考了一下 [Universal Transformers原理解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/44655133).

transformer 在 NLP task 好, 但是 algorithm task 表现比较差.

Neural Turing Machine 在 algorithm task 好, 但是 NLP task 差.

同时满足? => 原班人马提出的 Universal Transformer.

改动:

1. 不再是叠 n 层 (有 n 个网络), 而是只有 1 层但是**循环递归** n 次, 引入 **time step**.
2. feed forward 改成了 **transition function**: separable convolution 和 fully-connected neural network. 共享参数.
3. 每一个 token 有位置编码, token 的每一"层" (time step) 有**时间编码**, 长得和原版比较像.
4. 引入 **Adaptive Computation Time (ACT)**, 助教管他叫 Dynamic Halting. 是说不同的 token 的"深度"(循环递归次数)不同, 模型自己预测要不要停, 停了后面的 time step 直接复制.

然后我看了看试验还是 Neural GPU 比较好的样子... Universal Transformer 差很多, 怪不得我没听到怎么宣传...

![img](https://pic4.zhimg.com/80/v2-f6d26c24a811b6f7c3c5c614f0f4f733_1440w.jpg)

而且其实这个就是通用性的优化, 也没啥实际作用 (
