dataset 自定义

一般都是重载 `datasets.GeneratorBasedBuilder`: [HuggingFace Datasets来写一个数据加载脚本_名字填充中的博客-CSDN博客](https://blog.csdn.net/qq_42388742/article/details/114293746)

- `__info()`: 有什么列名 (属性)
- `__split_generator()`: 怎么分 train dev
- `__generate_examples()`: 比较重要, 具体怎么生成 examples

[Loading a Dataset — datasets 1.2.1 documentation (huggingface.co)](https://huggingface.co/docs/datasets/v1.2.1/loading_datasets.html)
