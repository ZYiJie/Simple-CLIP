# Simple-CLIP

利用开源的文本和视觉预训练模型微调来实现一个简单的CLIP图文对比检索模型（pytorch实现）

## 特点

- 简单易拓展，可以自由组合不同规模、类型的文本和视觉预训练模型，仅需在text_ptm、img_ptm中修改参数即可

- 使用 openAI 的 *Learning Transferable Visual Models From Natural Language Supervision* 论文原版模型

- 支持在test数据集中进行img2text、text2img检索验证

- 目前仅支持单卡训练


## 数据集

- Wukong中文开源数据集，相关数据处理参考[这个](https://github.com/ZYiJie/text2img)项目

- 参考data/example.tsv的格式可自由替换成自己的数据

## References & Resource

- CLIP：https://openai.com/blog/clip/

- 预训练模型：https://huggingface.co/models

- wukong数据集：https://wukong-dataset.github.io/wukong-dataset/benchmark.html