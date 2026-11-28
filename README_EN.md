<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo.svg" alt="DeepLLM" width="400">
  </picture>
</p>

<p align="center">
  <a href="README.md">简体中文</a> | <b>English</b>
</p>

<p align="center">
  A Chinese-language knowledge base for learning and practicing large language models: from core concepts and landmark papers to deployment and fine-tuning.
</p>

<p align="center">
  <a href="https://github.com/EarthWind/deepllm/stargazers"><img src="https://img.shields.io/github/stars/EarthWind/deepllm?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/EarthWind/deepllm/network/members"><img src="https://img.shields.io/github/forks/EarthWind/deepllm?style=flat-square" alt="GitHub Forks"></a>
  <a href="https://github.com/EarthWind/deepllm/commits"><img src="https://img.shields.io/github/last-commit/EarthWind/deepllm?style=flat-square" alt="Last Commit"></a>
</p>

DeepLLM collects study notes, paper explainers, course materials, code implementations and hands-on model tutorials about large language models. It is aimed at learners and developers who want a structured view of how LLMs work, or who need working examples of deployment, evaluation and fine-tuning.

> **Language note:** most content in this repository is written in Chinese. File and directory names, code, and many diagrams are in English, and the papers themselves are linked in their original language.

> This repository is a collection of materials and examples, not an application with a single set of dependencies or one entry point. Before running any code, follow the `README.md`, `requirements.txt` or notebook instructions in the relevant subdirectory.

## Contents

| Directory | What it contains | Use it to |
| --- | --- | --- |
| [`concept/`](concept/) | Core concepts such as RoPE, MoE, Mamba, GQA, MLA, sparse attention and Cross-Layer Attention | Understand the building blocks of modern LLMs |
| [`papers/`](papers/) | Explainers and reading paths for papers from Transformer to DeepSeek-R1 | Follow how the field developed and study topics in depth |
| [`course/`](course/) | Course materials on deep learning, NLP, pretrained models, reinforcement learning, compression and deployment | Fill in the fundamentals |
| [`code/`](code/) | Implementations of Transformer, BERT, attention, MLP, normalization and more | Read the code alongside the algorithm |
| [`models/`](models/) | Deployment, LangChain integration, web demos, evaluation and LoRA/GRPO fine-tuning tutorials for popular models | Reproduce practical workflows |
| [`model-architectures/`](model-architectures/) | Architecture diagrams of major open-weight models (draw.io sources + PNG), one folder per model family | Compare design choices across models |
| [`books/`](books/) | Chinese materials and notebooks for *Build a Large Language Model (From Scratch)*, *Hands-On Large Language Models* and related books | Study and practice chapter by chapter |
| [`blogs/`](blogs/) | Archived technical articles from Ahead of AI, Hugging Face, Maarten Grootendorst and others | Further reading on specific topics |
| [`llm-releases/`](llm-releases/) | Timeline of major LLM releases with specs, pricing, open-weight status and source checks | Keep up with new models |

## Highlights

### End-to-end LLM tutorial

[`concept/LLM全流程/`](concept/LLM全流程/README.md) walks through the full pipeline in eight chapters: data and tokenization, model architecture, pretraining, post-training, inference and deployment, RAG and tool use, and evaluation. It comes with four original SVG flow diagrams. Start here for the big picture before diving into individual architecture topics.

### Paper reading path

[`papers/to-2026/`](papers/to-2026/) contains explainers for 31 foundational and representative papers, organized into a suggested reading order:

- Foundations: Transformer, BERT, GPT, T5
- Scaling: Scaling Laws, Chinchilla, PaLM
- Alignment and fine-tuning: FLAN, InstructGPT, LoRA, QLoRA, DPO
- Reasoning and agents: Chain-of-Thought, Self-Consistency, ReAct, Tree of Thoughts, DeepSeek-R1
- Efficient architectures: RoPE, FlashAttention, Switch Transformer, Mixtral, Mamba
- Retrieval and multimodality: RAG, Toolformer, Gemini

See the [paper index](papers/to-2026/README.md) for the full list and recommended order.

### Model practice

[`models/`](models/) covers model families including DeepSeek, Qwen, GLM, Llama, Gemma, InternLM, Kimi, MiniCPM and MiniMax, with notes on various combinations of:

- Deployment with FastAPI, vLLM, SGLang, Ollama and web demos
- Integration with Transformers and LangChain
- Fine-tuning examples such as LoRA and GRPO
- Evaluation with tools such as EvalScope
- GPU, Docker and AMD environment setup

### From theory to code

To understand the Transformer through its implementation, try this order:

1. Read the attention, positional encoding and feed-forward topics in [`concept/`](concept/).
2. Follow the reading path in [`papers/to-2026/README.md`](papers/to-2026/README.md) for the foundational papers.
3. Compare with the implementations in [`code/annotated-transformer/`](code/annotated-transformer/) or [`code/transformer/`](code/transformer/).
4. Pick a model in [`models/`](models/) and work through deployment, evaluation or fine-tuning.

## Getting the repository

```bash
git clone https://github.com/EarthWind/deepllm.git
cd deepllm
```

Markdown documents can be read directly on GitHub. To run Python scripts or notebooks, create a separate virtual environment for the subproject you want and install the dependencies it declares. Examples differ in their Python, PyTorch, CUDA and GPU memory requirements.

The course documents are built with Sphinx. Install the dependencies in [`course/requirements.txt`](course/requirements.txt), then build from the `course/` directory:

```bash
python -m pip install -r course/requirements.txt
make -C course html
```

The output is written to `course/_build/html/`.

## Contributing

Issues and pull requests are welcome via [GitHub Issues](https://github.com/EarthWind/deepllm/issues), for example to:

- Fix facts, formulas, links or code in the documents
- Add explainers for new papers, architectures and models
- Add reproducible deployment, evaluation or fine-tuning tutorials
- Improve the directory indexes, reading paths and example instructions

When contributing, please note your sources, hardware and software environment, dependency versions and reproduction steps. Do not commit model weights, secrets or other large or sensitive files.

## Sources and copyright

This repository contains original notes, code from open-source projects, Chinese translations, and archived copies of publicly available third-party materials kept for study. Copyright and licensing remain with the original authors or projects. Before quoting, reposting or reusing any content, check the source notes and licenses in the relevant subdirectory and follow the original terms.

This repository is for learning, research and technical exchange only. The root directory does not currently declare a single open-source license, so content without its own license should not be treated as licensed for copying, modification or commercial use.
