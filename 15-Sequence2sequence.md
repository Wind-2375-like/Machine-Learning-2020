# Seq2seq and Attention

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Extra reading:**
    1. [李宏毅机器学习课程笔记-14.1 Seq2Seq：Conditional Generation - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/373459411)
    2. [李宏毅机器学习课程笔记-14.2 Seq2Seq：Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/373733143)
    3. [记忆网络之Memory Networks - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/29590286)
    4. [Teacher forcing是什么？ - MissHsu - 博客园 (cnblogs.com)](https://www.cnblogs.com/dangui/p/14690919.html)

---

## Seq2seq: Conditional Generation

可以用 RNN 生成一个 word、sentence、图片等等。

一个word有多个character组成，因此RNN每次生成一个character。一个sentence由多个word组成，因此RNN每次生成一个word。一张图片由多个像素组成，因此RNN每次生成一个像素。

<img alt="图 1" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/87d6a1e357f740fb80cbd8b2a7237fc8de68d3d98786f20fa0240583f34a09e1.png" style="zoom:50%;" />

特地讲了 image generation: 从上到下、从左到右逐个生成像素这种方法(上图右上角)并没有充分考虑pixel之间的位置关系，有一种更充分地考虑了pixel之间位置关系的方法(上图右下角)叫做PixelRNN，PixelRNN根据多个邻居生成一个像素，这可以通过3维LSTM单元(上图左上角)实现。如上图左上角所示，3维LSTM单元有3组输入和3组输出，将几层(每层9个)3维LSTM单元排列在一起，就可以生成一张3×3的图片。

以上来源: [李宏毅机器学习课程笔记-14.1 Seq2Seq：Conditional Generation - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/373459411)

但是实际上不想只是生成些随机的东西. 比如:

- image caption generation: 输入图片, 生成文字.
- machine translation: 输入原文, 生成译文.
- chat-bot: 输入我的对话和对话历史记录, 生成机器人的回话.

我们不光 generate 一个 sequence, 而是在 input 一个 "sequence" 的前提下 generate. 这就是 seq2seq, conditional generation.

具体分析任务:

- image caption generation:
  - <img alt="图 2" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/528dac762b0fbef1c494fbaae0c2993e49102991549cc1061675a0b1eb9178a6.png" style="zoom: 33%;" />  
- machine translation / chat bot:
  - <img alt="图 3" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6b0c2333addb3f4010c2d79f6626f791745c6994fcefda713120347862a8b6d3.png" style="zoom:33%;" />
  - 这里用到 encoder-decoder 结构
    - <img src="https://www.researchgate.net/profile/Chandan-Reddy-2/publication/329464533/figure/fig2/AS:701043021213696@1544153089687/The-basic-seq2seq-model-SOS-and-EOS-represent-the-start-and-end-of-a-sequence.ppm" alt="The basic seq2seq model. SOS and EOS represent the start and end of a sequence, respectively." style="zoom: 50%;" />
    - **jointly train**: encoder 和 decoder 的模型参数可同可变
  - 但是 Chat bot 需要考虑 context, 比如如果我在前面说"我叫 xxx", bot 之后如果再问"你叫什么名字"就太蠢了. 我认为目前 context 仍是个值得研究的方向.

## Attention: Dynamic Conditional

我认为 Attention 就是加权, 对某个 sequence 各个部分向量进行的加权求和, 得到具有侧重点的向量. 比如"我爱你"强调每个字, 侧重点都会有变化. 或者把 machine learning 翻译成 机器学习. 那么翻译 "机器" 时 "machine" 的权重比 "learning" 大, 翻译 "学习" 时 "learning" 的权重就大了.

**Attention 的 encoder-decoder 架构:**

<img alt="图 4" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/58bd49c08f10a26bf36461885f3b64012b57c5c07d944344d8b017ef4142ddbf.png" style="zoom:50%;" />  

权重的计算方式 (match 函数):

- Cosine similarity of z and h
- Small NN whose input is z and h, output a scalar
- $\alpha = h^{\mathrm{T}}Wz$

不要局限思维, 比如 softmax 一定从 hidden layer 取吗? 一定要用 softmax 吗? ...

**Image caption generation**

encoder 也可以是整个 CNN. Attention不只可以做单张Image的Caption，还可以做多张图片/视频的Caption，相关论文如《Describing Videos by Exploiting Temporal Structure》。(摘自[李宏毅机器学习课程笔记-14.2 Seq2Seq：Attention - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/373733143))

<img alt="图 5" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1460767553b33b2b3ca5f17ad588b9fd11f17edc353d863199740418bb339ffe.png" />  

**==Attention 具有比较好的可解释性.==**

| <img alt="图 6" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/3456bf71bac6f33eb359ed4584318425a9370c2fcd95b43587a2f1eec7328adb.png" style="zoom: 50%;" /><br>正确的判断 | <img alt="图 7" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/1e052f6ec539f3a5c4f128e9d37afe0d3ee1474032f98090582f617dba77bf35.png" style="zoom:50%;" /><br>错误的判断 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

我们可以知道为什么 machine 做出正确/错误的判断.

### Memory Network

有关这一部分, 可以参考[记忆网络之Memory Networks - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/29590286).

| <img alt="图 8" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/be7cd48e187a1f5a2423f91619bec00c91940106d49c7595955254facccfb874.png" style="zoom:33%;" /> | <img alt="图 9" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/f90f2592445ed4a059d34ae2c73a1a51abdbd695f08bc31e3b94090d820a0a11.png" style="zoom:33%;" /> |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

首先, 输入的 Document 编码成 N 个向量, 作为 memory.

然后 Question 经过编码得到 q, 对 memory 的内容进行 attention (match), 将记忆按照与 Question 的相关程度 (attention score) 进行组合得到输出向量.

更高级一点的是 Document jointly 编码出和 x 不同的向量 h, 用 h 和 attention score 加权得到输出向量.

接着我们将输出向量同 Question 再次合成新的向量 q, Document 编码成新的 N (或者 2N) 个向量作为新的 memory, q 和 x 进行 attention, attention score 和 h 加权得到 output.

这个过程可以叠很多层, 最后一层的 output 通过 DNN 生成 answer.

为什么叫 memory 呢, 我表示不太像 memory, 可能某些部分理解的还不够. 在参考[记忆网络之Memory Networks - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/29590286)发现 memory 的生成比我想象的复杂.

<img alt="图 10" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/50ae5e913d5008e441300a47bb718d4cb7a53b419bba576cd6d108b283ceddf4.png" />  

原来这里 N 不等于 Document 的句子个数. input 将 document 的每句话变成 sentence 个数的 vector, vector 经过 generalization (是另一个 seq2seq?) 变成 N 个新的 vector, 写入 memory slots, 具体怎么写我不太清楚, 难道是直接覆盖吗 (这样能长期留住吗)? 留作未解决问题. 反正这个看起来像个 memory.

### Neural Turing Machine

这么看上面的 memory network 也可以改 memory, 但是看着好 low 啊. neural Turing machine 不止可以读 memory, 还能改 memory, memory 可以长期存储.

| <img alt="图 12" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/978841d5427f42a3f9732584f97393c1e0f4e43425cd0395d30ffc0b7a7b985c.png" style="zoom: 50%;" /><br>$r^0=\sum{\hat{\alpha_0^i}m_0^i}$<br>$k^1, e^1, a^1, h^1, y^1=f(x^1, r^0, h^0)$<br>$\hat{\alpha_1^i}=\text{softmax}(\text{sim}(m_0^i, k^1)) $ | <img alt="图 14" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/29caa30ff86833cd7b897add7433accf5c91c621f4766f3fffe45d2bc29778f2.png" style="zoom: 50%;" /><br>$m_0^i:=m_0^i-\hat{\alpha_1^i}e^1\odot m_0^i+\hat{\alpha_1^i}a^1$ |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

左图, 初始化有个 memory m 和 attention score α, 加权组成 r0, 同输入 x1, 隐层 h0, 一同算出 k1, e1, a1, h1, y1.

k 用来和 memory 计算更新 attention score

e, a 用来更新 memory. e 控制以往 memory 的删除, a 控制新的信息的写入, α 控制力度.

然后再继续迭代下去:

<img alt="图 16" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8a2fa64c17ac7133d2de53f70a0936b829010958b681e12be65a78c4710117e9.png" style="zoom:50%;" />  

我比较好奇这里计算 r0 的 α 和写入 memory 的 α 居然是共用的. 也就是说前一时刻 memory 比较重要的信息传入 f 得到隐层和输出, 结果 memory 对应的重要信息还要被覆写掉, 进入下一个时刻. 所以重要的信息用过就马上更换吗? 不太理解.

### Pointer Network

pointer network 最初想要解决演算法问题: 给一堆点坐标和编号, 输出一个节点编号集合, 选中的节点可以包含所有的点.

<img alt="图 17" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/9c2272bda221a7473d90aa1bf985498344061d2a94d54a37f0e00809eba0c1db.png" style="zoom:33%;" />

这个就是凸包问题 (Convex Hull), 信竞这已经是很高级的玩意儿了. 结果现在有了 NN 之后有人想用 seq2seq 硬 train 一发:

<img alt="图 18" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/11297f107c53172791aba8e46be16972ecdfbbc366cb7130c668596e511edba7.png" style="zoom:50%;" />  

encoder 依次输入坐标. decoder 一步步输出点的序号, 直到 END 标记. 但是这样并不 work.

假如最多有 100 个点, 但是训练只练 50 个点. encoder 可以处理变长序列, 一个个 encode 就行. decoder 进行 softmax 之前的 output 是 100 维 (代表编号 1-100 的概率). 训练的时候强行让后 50 个变成 0 再 softmax, 训练之后 model 或许可以学会找到 50 个点的凸包. 但是测试集 100 个点, machine 在训练时从来没输出过 test 需要的标号, test performance 怎可能好?

于是就有了 pointer network. What decoder can output **DEPENDS** on the input.

<img alt="图 20" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/96102a962d519bbd76400e60ad327c3439ef6eca93149416d474f8904f446333.png" />

encoder 得到隐层向量 h 之后 (这里最开始多了个 END 代表的 embedding), decoder 的 hidden state (也可以是别的), 和 encoder 每个序列的 h 计算 attention score, softmax, argmax, point 出最大概率的 output. decoder 下一时刻的 input 是 pointer 的 output 对应的 embedding (在这里可能是坐标之类的), 直到 point 出序号 0 (对应 END) 结束.

pointer network 使 decoder 的输出直接从 input 的各个坐标, word, ... (token) 里面取, 而非自己 generate. 这样比如在 text summarization, text2SQL, ... 中有的输出词语 (人名, 地名, ...) machine 之前从没见过, 那 decode 的时候 machine 就 output 不了他们, 称为 OOD (out of domain) 词语. 用 pointer network 就能输出他们.

另一方面, 我们还希望 machine 能自己 generate 一些东西, 而非完全摘抄原文.

因此一种常见方法为:

- 以 $p_{\text{gen}}$ 的概率 generate 一些词语.
- 以 $1-p_{\text{gen}}$ 的概率使用 pointer network 摘抄一些词语.

<img alt="图 21" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/427bf8ae91ecff19d10e876b8c42c006728699cd421a69cf74aa13a1ba715927.png" style="zoom:50%;" />  

## Tips for Seq2seq

接下来讲一些 tips.

### Attention regularization

假如要做video的caption generation，某视频有4个frame，即有4张图片。

<img alt="图 22" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/a36b5d8cde919899f4c95bcbc8ea3ecf9487fcd2081dd79a5d086c9b83e10599.png" />  

用 $\alpha_t^i$ 表示attention weight，其上标表示frame的索引、下标表示时刻的索引。在第1个时刻，对上面四张图片产生attention $\alpha_1^1, \alpha_1^2, \alpha_1^3, \alpha_1^4$，生成第1个word w1；在第2个时刻，以此类推……

现在我们关注每张图片在各个时刻的 attention. 发现有这样一种情况: 第二张有个女人的图片在每个时刻的 attention 都比较大, 而其他图片每个时刻都没什么 attention. 结果 w1, w2, w3, w4 全都生成 woman... 这就是**bad attention**。

**good attention**需要关注到输入中的每个frame，对每个frame的关注度不能太多也不能太少并且应该是同等级的。那如何实现这种好的attention呢？比如使用正则项 $\sum_i(\tau-\sum_t\alpha_t^i)$ 使得每个frame在所有时刻的attention weight之和都接近τ，这个τ是通过学习得到的，详见《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》。

以上内容摘改自[李宏毅机器学习课程笔记-14.3 Seq2Seq：Tips for Generation - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/374340958).

### Teacher Forcing

以下部分改自[Teacher forcing是什么？ - MissHsu - 博客园 (cnblogs.com)](https://www.cnblogs.com/dangui/p/14690919.html).

RNN 存在两种训练模式（mode）：

1. free-running mode: 上一个state的输出作为下一个state的输入。
2. teacher-forcing mode: 使用来自先验时间步长的输出作为输入。

常见的训练RNN网络的方式是free-running mode，即将上一个时间步的输出作为下一个时间步的输入。可能导致的问题：

- Slow convergence.
- Model instability.
- Poor skill.

训练迭代过程早期的RNN预测能力非常弱，几乎不能给出好的生成结果。如果某一个unit产生了垃圾结果，必然会影响后面一片unit的学习。错误结果会导致后续的学习都受到不好的影响，导致学习速度变慢，难以收敛。teacher forcing最初的motivation就是解决这个问题的。

<img alt="图 23" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/5f89cd16bb32b5d317ec296c10462296f42d9a39208e715a639508d2a62087b2.png" />

使用teacher-forcing，在训练过程中，模型会有较好的效果，但是在测试的时候因为不能得到ground truth的支持，存在 exposure bias, 训练测试存在偏差, 模型会变得脆弱。

<img alt="图 24" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ce32e0671839737699b75840744bf9230d72cbf1b5e90a9f612247030774e8eb.png" style="zoom:50%;" />  

训练的时候, 第一步错了, 后面还能对. 测试的时候, 第一步错了, machine 从来没学过错误的路怎么走, 就会步步错... 那有没有解决这个限制的办法呢? 这里给出 scheduled sampling 和 beam search.

#### Curriculum Learning (Scheduled Sampling)

Curriculum Learning是Teacher Forcing的一个变种：一开始老师带着学，后面慢慢放手让学生自主学。

Curriculum Learning即有计划地学习：

<img alt="图 25" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/7011d3237aa27f70bf7ab2e56f7ef5661b896ebb222bced3c3aa8b93819ee745.png" style="zoom:50%;" />  

- 使用一个概率 $p$ 去选择使用ground truth的输出 $y(t)$ 还是前一个时间步骤模型生成的输出 $h(t)$ 作为当前时间步骤的输入 $x(t+1)$。
- 这个概率 $p$ 会随着时间的推移而改变，称为**计划抽样(scheduled sampling)**。
- 训练过程会从force learning开始，慢慢地降低在训练阶段输入ground truth的频率。(exponential, inverse sigmoid, linear, ...)

#### Beam Search

beam search 在 search tree 希望能够搜到一条概率最大的 path.

<img alt="图 26" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/52b0e900eea0777f51de7b8a6e2d23702f023bd7fa4f5eb37425617f09405820.png" />  

greedy search 会选左边, 因为第一个分支概率大. 但是全局概率是右面的高. 0.4\*0.9\*0.9 > 0.6\*0.6\*0.6.

但是暴搜是不可能的. n 层二叉树有 $2^n$ 个 path...

beam search 第一层从 m 叉树 greedy 选 k 个最优叉, 接着后面每一层从 k 个叉的 km 个子 path 再选 k 个最优叉, 以此类推, 直到最后一层从 k 个 path 中选出最优的一个.

<img alt="图 27" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/cfccbe9d92988f40cbf699118e45a37ab643d2dfa4519d049efc0bd4f86bbb4f.png" style="zoom:50%;" />  

这里 k 是 beam size. k=1 就是 greedy search. k=m 就是 brute search. 一般 k 不会取太大.

这里我觉得应该训练和测试都用 beam search 输出 output 吧 (可能不对). 如果这样的话确实可以解决 decoder 一步错步步错的问题. 因为 beam search 可以搜索多条候选路径, 哪怕第一步错了, 但是后续 path 的得分小, 也不会选, 就没事了.

我查了一下 beam search 的道理在哪里. 然后发现了这篇[你一直在用的Beam Search，是否真的有效？ - 腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/1757936), 是 ETHz Ryan NLP 巨佬组发的一篇论文 [If beam search is the answer, what was the question? - ACL Anthology](https://aclanthology.org/2020.emnlp-main.170/).

这个 blog 对论文的解读如下: 序列生成模型中，增大beam search的搜索宽度反而会导致生成文本质量==下降==，**这说明beam search的成功并不是因为它能够很好地逼近exact search，而是因为beam search算法本身有良好的inductive bias，恰好抵消了模型本身的误差。**作者通过探索解码目标MAP的==正则项==，将beam search隐含的归纳偏差与认知科学中的==均匀信息密度(UID)==假说联系起来，通过实验证明了UID假说与文本质量的强相关性，以及beam search隐含的归纳偏差使得模型能够生成更符合UID假设的文本，恰好弥补了模型本身的误差。

blog 认为:

- 需要注意的是，这篇论文只解释了beam search为什么好，但是没有解释exact search为什么这么糟糕，而后者实际上是更值得关注的问题，因为这表明模型本身是有问题的，也许是设计的目标函数和人类的认知不一致，也许用链式法则来建模句子是错的，也许是训练和测试的不一致导致的，而总是做次优决策的解码方式弥补了模型的不足。另外，我们也不能仅仅通过对解码的过程加正则项来优化决策，因为此时的得分已经不是单纯的似然概率了，增加的正则项并不是模型所训练的目标。
- UID假设本身还是有一些启发式的感觉，UID假设本身是否站得住脚也是需要商榷的，毕竟人类的语言不能用一条假设就完全概括。实际上有一些研究认为UID虽然对句法归约比较有效，但在其他的语言现象中(如词序变换、句法变换)存在一些明显与UID假设相悖的结论。

### Use Output Probability as Next Input?

这个一看就感觉不行. 我不觉得这个能起到保存 search tree 每条 path 的概率的作用. 可以 end2end backprop 也没觉得怎样.

而且 lecturer 提到这个反例:

<img alt="图 28" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/ee7f75c285841c872c983883be41404e193e5d9e44bf7b5a8f464b6cde398e43.png" style="zoom:50%;" />  

送进 distribution 如果 高兴 和 难过 概率差不多, 那么下一步 想笑 和 想哭 概率也差不多. 所以就会自由组合. 可能出现 高兴想哭 或 难过想笑 这种不太合逻辑的东西...

### Object Level: Reinforcement Learning

<img alt="图 29" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/059f7fd98e24f8b6a1f217f39f9bde730cb34b25bfa894706d5a0c46347fb221.png" style="zoom:50%;" />  

假如我们要生成一个sentence，那我们就应该关注整个sentence(Object Level)而不仅仅是每个word(Component Level)。

如果是按照Component Level，那使用Cross Entropy计算损失的话，训练前期loss会下降得很快，但后期loss会下降得很慢（“The dog is is fast”和"The dog is running fast"的loss的差距很小）。

那有没有一个损失函数可以基于Object Level衡量两个句子间的差异呢？确实可以整一个. 那可以 Gradient Descent 吗? 不行. 因为模型输出的分布是离散的，如果微小改变模型参数, 模型输出的句子可能还是相同，那损失函数的输出就是一样的，即微小扰动并没有对loss产生影响。

RL 可以解决不可微的问题. 没有好好学 RL, 就贴一张图吧.

<img alt="图 30" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/06a9bd69cdc14d7b2f9f09c6a14beb455cbcbd82fc485e54480768f776f71d9e.png" />  

以上摘改自[李宏毅机器学习课程笔记-14.3 Seq2Seq：Tips for Generation - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/374340958)

## Recursive Network

recursive network 是 RNN 的泛化版本.

- Recurrent: 顺序的
- Recursive: 只要是可复用的结构就可以

recursive network 的 neuron 的输出, 可以作为输入的一部分返回到这个 neuron 当中.

<img alt="图 31" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/6d842bda8a43243e770ae89eeaeee6e71241c2bd811dbfc36c59cc8acf3f1790.png" style="zoom:50%;" />  

比如这里 h 和 x 维度相同, 我们就可以像搭积木一样把这些 f 叠起来, 这些 f 应该是同一个, 共用参数. 只不过组合次序不再是线性的了.

RNN 也是一种特殊的 Recursive Network. 组合顺序是线性的. f 的一个输出 h 可以接到下一时刻 f 的输入当中, 就跟叠积木一样.

<img alt="图 32" src="https://cdn.jsdelivr.net/gh/Wind2375like/I-m_Ghost/img/8de5d8419f4f04c582bf8f621f2edf02b699887cdc3414e7979a484cc0743a35.png" />

与Recursive Network相关的模型有：Recursive Nerual Tensor Network、Matrix-Vector Recursive Network、Tree LSTM。  
