# Word Embeddings (Word2vec)

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [词向量Word Embedding原理及生成方法 (getui.com)](https://www.getui.com/college/2021053176)
    2. [Skip-Gram: NLP context words prediction algorithm | by Sanket Doshi | Towards Data Science](https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c#:~:text=Skip%2Dgram%20is%20one%20of,while%20context%20words%20are%20output.)
    3. [万物皆可Embedding之Word Embedding - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/384452959)
3. **Extra videos:**
    1. [Word2Vec - Skipgram and CBOW - YouTube](https://www.youtube.com/watch?v=UqRCEmrv1gQ)

---

词向量是 unsupervised learning 的一个应用. 我们先看看之前有哪些 embedding 的方法.

## One-hot Encoding and Word Class

one-hot 向量就是只有一个值为 1, 其他全为 0 的向量. 假如向量有 V 维, 就有 V 个 one-hot 向量. 我们让 V-1 代表词典的大小, 每一个词由 V-1 个 one-hot 向量中的一个来表示. 比如 a 是 [1 0 0 ...], abandon 是 [0 1 0 ...]. 最后剩下的一个向量 [0 ... 0 1] 表示 unknown word.

但是这样的问题是:

- 向量维度太大. 单词如果有几百万个, 那么维度也要几百万... 矩阵乘法累死.
- 不同单词之间内积为 0, 没有相似度, 都是正交的. 无法得知不同单词之间的关系.

但是 one-hot 至少可以唯一地表示出每个单词, 而且也蛮重要的... 下面这个 word class 方法让我觉得就很扯. word class 人为将意思相似的词划分到一类, 用 class embedding 表示这一 class 所有 word 的 embedding. 先不说人为划分的关系到底靠不靠谱, 难道一个词只能属于某一类吗?

比如我把 "鸟, 猪, 人" 划分为动物类, "啄, 吃, 唱" 划分为用嘴的动作. 这两类用两种 embedding 来表示. 但是只有 "鸟" 才会 "啄" 吧, 所以 "鸟" 和 "啄" 又具有某种关系, 应该划到鸟类. 所以一个词可以有很多类, 关系很复杂, 不能简单地 cluster.

## How Word Embedding Works?

那么词向量怎么运作呢? 他让 machine 阅读大量文章, 仅仅观察文章中词的出现频率, 不需要人为标注, 进行 unsupervised learning 即可.

他的中心思想是: understand words by context. 比如;

- 我 **今天** 吃了 炸 **鸡块**
- 我 **明天** 吃了 炸 **猪排**

这两句话, 我, 吃了, 炸, 是一样的词. 唯一不同的是 今天/明天 和 鸡块/猪排, word embedding 认为 context 就是 word 前后的词. 如果两个词有相似的 context, 那么两个词就应该有相似的意思和相似的 embedding. 比如 今天/明天 都是时间状语, 鸡块/猪排 都是用来炸的食物.

基于这种思路, 人们提出两种 word embedding 的方式. count based 和 prediction based.

## Count-based: GloVe

其实现在 GloVe 已经是预处理的词向量了, 其他什么 CBOW, skip-gram 都不知道哪里去了. 但是 lecture 中没有仔细介绍, 想必是比较复杂, 之后还要重新学啊.

GloVe 的想法就是统计词 $w_i, w_j$ 共同出现的频率. 如果出现频繁, 那么 $V(w_i)$ 就会接近 $V(w_j)$. 这里 $V(\cdot)$ 代表 embedding 的函数.

也就是说 $V(w_i)\cdot V(w_j) \Leftrightarrow N_{i, j}$

## Prediction-based: CBOW and Skip-gram

prediction-based 的想法就是预测附近会出现什么词.

<img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/efaf207f527a50f261f63f1ee5f330698186ff9d41540654e079f737f110c3d6.png" />  

输入一个 one-hot 向量, 首先经过**线性变换**获得隐层表示 z (没有激活函数). z 再经过线性变化和 softmax 算出各个单词作为后一个词的概率. 和真的 $w_i$ 求 loss 进行梯度下降, 更新线性变换的矩阵参数.

这个任务看起来很奇怪, 他能真的预测出来吗? 但他就像命运石之门里的 G-Back 任务一样, 目的并不在于预测, 而是获得隐层表示 z. 如果想要预测出相同的 "宣誓就职" 一词, 那么 "蔡英文" 或者 "马英九" 所应该得到的隐层表示 z 应该是相似的, 哪怕输入的 one-hot encoding 并不一样. 词向量的最终目的, 是为了得到每个 one-hot encoding 所得到的隐层表示 z 作为词向量.

这里输入的 one-hot 是 V 维, 线性变换矩阵为 N×V. V 代表 one-hot 的维度, 很大. N 代表隐层维度, 比较小, 可能就几百维. 矩阵乘向量代表矩阵列向量的线性组合. 如果乘以 one-hot, 就相当于选择 N×V 矩阵的某一列. 也就是说隐层表示被保存在了线性变换的矩阵当中. 最终 train 完这个任务的矩阵就是我们真正想要的东西.

接下来简单介绍一下 CBOW 和 Skip-gram.

### CBOW

<img alt="图 13" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1c9f959f05d4e661bca219291d5e2f57178a0785eb61a4c370b00bb5f8988dad.png" />

CBOW 用周围的 context 来预测中心词是什么. 我们发现周围词用的线性变换矩阵是一样的, share parameters. 因为:

- W 的每一列/行存储了 one-hot 对应位置的一个词向量. 如果每个 context 对应的 W 都不一样, 那么相同列/行的词向量会不一样, 岂不是说明一个词有好几个词向量吗? 所以只能有一个 W.
- 降低维度, 减少 memory. 存储 $W_{CV \times N}$ 明显比 $W_{V \times N}$ 占用更大空间.

### Skip-gram

<img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1809ee2a540d456f601d6a0e60dddf82c9097931f03e9b6defd3891f15d0384a.png" />  

反着来, 最后还是要前面线性变换的矩阵. 但是这个用中间词预测周围词.

看起来并不 deep, 因为 word embedding 通常只是 NLP 中预处理的操作, 后面还要用更复杂的网络模型. Mikolov 用了很多 tips 优化, 不用 deep 就可以得到很好的 embedding. 现在 word embedding 通常都是直接调包用人家训练好的作为初始输入.

## Results

<img alt="图 15" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5e5d1cd120d61fccc5bec7ac0b5bccca93bcb8d6658e3ce4c1f269b6a6f4f752.png" />  

词向量看起来有很多奇妙的特性. 比如 中国-北京 ≈ 日本-东京, 代表他们都是国家-首都的关系. 就像 罗马之于意大利就像柏林之于___ 一样.

## Drawback

似乎 word embedding 对于一词多义的现象不太好处理, 因为每个词只能用一个 embedding 来表示.

## Application

word2vec 引起了一波 embedding 热潮. 比如 multi-domain, document, semantic, ... 不再赘述.
