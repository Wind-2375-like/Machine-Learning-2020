# ML 2020: Introduction

> reference的内容为唯一教程，接下来的内容仅为本人的课后感悟，对他人或无法起到任何指导作用。

## Reference

1. **Course website and video:** [ML 2020 Spring (ntu.edu.tw)](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.php)
2. **Notes:** [LeeML-Notes (datawhalechina.github.io)](https://datawhalechina.github.io/leeml-notes/#/)
3. **Extra reading:** None
4. **Extra videos:** None

---

![img](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205091605935.png)

## Concept

Machine Learning, 就是让机器自己去找输入 $x$ 和输出 $y$ 的一个映射/函数. $y=f(x)$.

学习和本能不同. 本能是先天就有的, 通过一连串预设的 if 来执行. 这样子并不能叫做智能:

![img](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205091605237.png)

怎么找函数呢? 一共有三步, 就跟把大象放进冰箱里面一样:

- 定义一组限制在一个范围的函数 $f_1, f_2, \cdots \in M$, 这个 M 叫做 model set. 比如线性模型集, 或者各种非线性模型集 (NN), 具体的某个函数叫做 model
- 定义选择函数的好坏.
- 选择最佳的函数 $f^*$.

![img](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202205091605024.png)

## Categories

具体怎么找函数呢?

- 监督学习: 给 training data, 带有正确答案 label, 机器通过某种模型输出他的答案 prediction, 和 label 比较计算 loss, 找 loss 最低的模型.
- 半监督学习: data 标 label 很麻烦, 能不能用大量无 label 的数据和少量带 label 的数据训练出结果?
- 无监督学习: data 干脆没有 label, 能学到些什么东西. 比如给一堆二次元角色图片送进机器, 机器能不能自己创造新的?
- 强化学习: data 同样没有 label, 通过策略和奖励来驱动机器寻找一个最优函数.

## Cutting Edge

- 可解释性: 模型/数据的什么 features 会影响预测?
- 异常检测: 机器能不能知道自己"不知道"? (猫和狗的图片中突然冒出一个光头)
- 元学习: 学习如何学习
- 对抗学习: 捣乱, 对模型/数据增加噪声
- 迁移学习: 训练时的任务是对猫/狗进行分类, 有 label. 测试时的任务变成的对凉宫春日/实玖瑠进行二分类 or 训练时语音识别是在安静的环境, 测试时变成噪杂的大教室, 有回音
- 终身学习/增量学习: 每多学一个任务, 就能解决所有学过任务的问题

关于 classification 只是 ML 的一小部分, 而 generation (生成 structured data) 才是终极目标: 个人看法是目前的 generation 不过是更高级一点的 classification(?
