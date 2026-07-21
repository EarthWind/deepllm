# DeepLLM

<p align="center">
  面向大语言模型学习与实践的中文知识库：从基础原理、经典论文到模型部署与微调。
</p>

<p align="center">
  <a href="https://github.com/EarthWind/deepllm/stargazers"><img src="https://img.shields.io/github/stars/EarthWind/deepllm?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/EarthWind/deepllm/network/members"><img src="https://img.shields.io/github/forks/EarthWind/deepllm?style=flat-square" alt="GitHub Forks"></a>
  <a href="https://github.com/EarthWind/deepllm/commits"><img src="https://img.shields.io/github/last-commit/EarthWind/deepllm?style=flat-square" alt="Last Commit"></a>
</p>

DeepLLM 汇集了大模型相关的学习笔记、论文解读、课程资料、代码实现和模型实践教程。内容以中文为主，适合希望系统理解 LLM 技术路线，或需要查阅部署、评测和微调示例的学习者与开发者。

> 本仓库是资料与示例集合，不是一个具有统一依赖和启动入口的应用。运行代码前，请以对应子目录中的 `README.md`、`requirements.txt` 或 Notebook 说明为准。

## 内容导航

| 目录 | 内容 | 适合场景 |
| --- | --- | --- |
| [`concept/`](concept/) | RoPE、MoE、Mamba、GQA、MLA、稀疏注意力、Cross-Layer Attention 等核心概念 | 快速理解现代 LLM 架构组件 |
| [`papers/`](papers/) | Transformer 至 DeepSeek-R1 的论文导读、原理拆解与阅读路线 | 建立论文脉络，进行专题精读 |
| [`course/`](course/) | 深度学习、NLP、预训练模型、强化学习、模型压缩与部署等课程资料 | 系统补齐基础知识 |
| [`code/`](code/) | Transformer、BERT、Attention、MLP、Normalization 等实现 | 对照代码理解算法细节 |
| [`models/`](models/) | 主流模型的部署、LangChain 接入、Web Demo、评测和 LoRA/GRPO 微调教程 | 复现模型实践流程 |
| [`books/`](books/) | 《从零构建大语言模型》《动手学大模型》相关中文资料与 Notebook | 按书籍章节学习与练习 |
| [`blogs/`](blogs/) | AHEAD OF AI、Hugging Face、Maarten Grootendorst 等技术文章归档 | 扩展阅读与专题参考 |

## 重点内容

### 论文阅读路线

[`papers/to-2026/`](papers/to-2026/) 收录 31 篇基础与代表性论文的中文原理解读，并按以下主题组织阅读顺序：

- 基础架构：Transformer、BERT、GPT、T5
- 规模化训练：Scaling Laws、Chinchilla、PaLM
- 对齐与微调：FLAN、InstructGPT、LoRA、QLoRA、DPO
- 推理与 Agent：Chain-of-Thought、Self-Consistency、ReAct、Tree of Thoughts、DeepSeek-R1
- 高效架构：RoPE、FlashAttention、Switch Transformer、Mixtral、Mamba
- 检索与多模态：RAG、Toolformer、Gemini

完整索引与建议顺序请参阅[论文原理文档索引](papers/to-2026/README.md)。

### 模型实践

[`models/`](models/) 覆盖 DeepSeek、Qwen、GLM、Llama、Gemma、InternLM、Kimi、MiniCPM、MiniMax 等模型家族，包含不同组合的实践记录：

- FastAPI、vLLM、SGLang、Ollama 与 Web Demo 部署
- Transformers 与 LangChain 接入
- LoRA、GRPO 等微调示例
- EvalScope 等评测实践
- GPU、Docker 及 AMD 环境配置

### 从原理到代码

如果希望结合实现理解 Transformer，可以从以下内容开始：

1. 阅读 [`concept/`](concept/) 中的注意力、位置编码与前馈网络专题。
2. 按 [`papers/to-2026/README.md`](papers/to-2026/README.md) 的路线阅读基础论文。
3. 对照 [`code/annotated-transformer/`](code/annotated-transformer/) 或 [`code/transformer/`](code/transformer/) 查看实现。
4. 进入 [`models/`](models/) 选择具体模型完成部署、评测或微调实践。

## 获取仓库

```bash
git clone https://github.com/EarthWind/deepllm.git
cd deepllm
```

Markdown 文档可以直接在 GitHub 中阅读。运行 Python 脚本或 Notebook 时，建议为目标子项目创建独立虚拟环境，再安装该目录声明的依赖；不同示例对 Python、PyTorch、CUDA 和显存的要求可能不同。

课程文档使用 Sphinx 组织，可在安装 [`course/requirements.txt`](course/requirements.txt) 中的依赖后进入 `course/` 目录构建：

```bash
python -m pip install -r course/requirements.txt
make -C course html
```

构建结果位于 `course/_build/html/`。

## 参与贡献

欢迎通过 [Issue](https://github.com/EarthWind/deepllm/issues) 或 Pull Request 补充与改进内容，例如：

- 修正文档中的事实、公式、链接或代码问题
- 补充新论文、新架构与新模型的原理解读
- 增加可复现的部署、评测或微调教程
- 改善目录索引、阅读路线和示例说明

提交内容时，请尽量注明资料来源、软硬件环境、依赖版本和复现步骤，并避免提交模型权重、密钥或其他大体积/敏感文件。

## 资料来源与版权

仓库中包含原创笔记、开源项目代码、中文翻译以及第三方公开资料的学习归档。相关内容的版权与许可归原作者或对应项目所有；引用、转载或使用前，请查看具体子目录中的来源说明与许可证，并遵守原始条款。

本仓库仅用于学习、研究与技术交流。根目录目前未声明统一的开源许可证，因此不要将未单独标注许可的内容视为已获得任意复制、修改或商业使用授权。
