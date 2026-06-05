![gemma4](./images/gemma4_banner.png)
# Gemma4 模型列表与选型参考

## 1. 先说结论

Gemma4 这一代可以先按 4 类来理解：

- **E2B / E4B**：面向手机、边缘设备、浏览器和轻量本地部署的 effective 小模型；
- **12B Unified**：中等体量的统一多模态模型，不再依赖独立视觉/音频编码器；
- **26B A4B**：MoE（Mixture of Experts）路线，强调高吞吐和更强推理能力；
- **31B Dense**：大号 dense 模型，适合更充足的本地工作站或服务端资源。

如果只是快速选型，可以先这样记：

- **想上手机/端侧**：优先看 E2B、E4B
- **想要统一多模态、结构更简洁**：优先看 12B Unified
- **想要更高吞吐的强推理能力**：优先看 26B A4B
- **想要最强 dense 上限**：优先看 31B Dense

---

## 2. 参考图对应的四款核心型号

下面这张表按你提供的参考图整理，并结合 Gemma4 官方 overview / model card 中可核对的信息补充了中文说明。

| 属性 | E2B | E4B | 12B Unified | 31B Dense |
| --- | --- | --- | --- | --- |
| 架构定位 | Effective 小模型 | Effective 小模型 | Unified 统一多模态模型 | Dense 大模型 |
| 总参数量 | 2.3B effective（含 embeddings 约 5.1B） | 4.5B effective（含 embeddings 约 8B） | 11.95B | 30.7B |
| 层数 | 35 | 42 | 48 | 60 |
| Sliding Window | 512 tokens | 512 tokens | 1024 tokens | 1024 tokens |
| Context Length | 128K tokens | 128K tokens | 256K tokens | 256K tokens |
| Vocabulary Size | 262K | 262K | 262K | 262K |
| 支持模态 | Text, Image, Audio | Text, Image, Audio | Text, Image, Audio | Text, Image |
| Vision Encoder 参数 | ~150M | ~150M | - | ~550M |
| Audio Encoder 参数 | ~300M | ~300M | - | No Audio |
| 典型定位 | 超轻量端侧、多模态低延迟 | 更强端侧、多模态平衡点 | 统一多模态本地执行 | 更高能力的 dense 旗舰 |

### 2.1 怎么理解 E2B / E4B 的 effective

这里最容易误解的是 “2B / 4B”。

- 它不是说总静态权重真的只有 2B / 4B；
- 它更接近“主干有效计算规模”；
- E2B / E4B 之所以加载内存比字面直觉更高，是因为它们带有大量 **Per-Layer Embeddings（PLE）** 权重；
- 这些额外权重主要是查表型静态参数，增加内存多于增加 FLOPs。

因此：

- **E2B = 更激进的端侧优先**
- **E4B = 端侧与能力之间更均衡**

### 2.2 怎么理解 12B Unified

12B Unified 的关键不是“只比 E4B 大一些”，而是：

- 它把多模态能力做成了 **统一模型路径**；
- 不再像 E2B / E4B / 31B Dense 那样依赖独立视觉编码器或音频编码器参数表；
- 从部署角度看，结构更统一，适合希望减少多模态拼装复杂度的场景。

### 2.3 怎么理解 31B Dense

31B Dense 可以理解为 Gemma4 里的大号 dense 主力：

- 参数规模更大；
- 上下文窗口达到 256K；
- 保留图像能力；
- 更适合本地工作站或服务端，而不是强调手机或浏览器端部署。

---

## 3. 官方发布但参考图未覆盖的 26B A4B

严格来说，只列上面四个型号还不够完整。  
Gemma4 官方 overview 明确把 **26B A4B** 列为正式型号之一，它代表的是 **MoE 路线**。

| 属性 | 26B A4B |
| --- | --- |
| 架构定位 | Mixture of Experts（MoE） |
| 总参数量 | 26B 总参数 |
| 每 token 激活参数 | 约 4B |
| Context Length | 256K tokens |
| 主要特点 | 高吞吐、强推理能力、效率优先 |
| 官方定位 | advanced reasoning / high-throughput |
| 内存参考（BF16 / SFP8 / Q4_0） | 48 GB / 25 GB / 15.6 GB |

这一型和 E2B / E4B 的“effective”不是一回事：

- **E2B / E4B**：很多额外参数是 lookup-heavy 的 PLE 权重；
- **26B A4B**：是 MoE，只在生成时激活一部分专家；
- 但为了高效路由和推理，**全部 26B 权重仍然要加载进内存**。

所以实际工程选型时：

- 如果你关心 **端侧和轻量本地运行**，优先看 E2B / E4B；
- 如果你关心 **更高能力但仍讲究效率**，26B A4B 是很重要的一档；
- 如果你明确偏好 **dense 路线**，再看 31B Dense。

---

## 4. 适合什么场景

### E2B

- 手机、平板、浏览器、WebGPU、本地轻量推理
- 对音频输入有需求的端侧场景
- 希望在较低显存/内存下跑起来

### E4B

- 仍然希望端侧部署，但要比 E2B 更强的综合能力
- 本地笔记本 GPU、小型工作站
- 多模态助手、OCR、轻量 agent

### 12B Unified

- 想要统一多模态路径，不想维护额外 encoder
- 希望模型结构更简单、调用链更统一
- 本地高配设备或中等服务器

### 26B A4B

- 更强推理、代码、agent 工作流
- 高吞吐在线服务
- 能接受较高显存，但希望效率优于同级 dense

### 31B Dense

- 更看重 dense 模型上限
- 更适合高质量本地工作站或服务端部署
- 适合复杂 reasoning、长上下文、多步骤任务

---

## 5. 常见 Hugging Face 模型名

Gemma4 在 Hugging Face 上常见的主模型命名可以先记这一组：

- `google/gemma-4-E2B`
- `google/gemma-4-E2B-it`
- `google/gemma-4-E4B`
- `google/gemma-4-E4B-it`
- `google/gemma-4-12B`
- `google/gemma-4-12B-it`
- `google/gemma-4-26B-A4B`
- `google/gemma-4-26B-A4B-it`
- `google/gemma-4-31B`
- `google/gemma-4-31B-it`

另外，Gemma4 还提供了和 **Multi-Token Prediction（MTP）** 相关的 draft / assistant 变体，主要用于 speculative decoding 加速，不应和主模型列表混为一谈。

---

## 6. 一个实用的选型顺序

如果你是按硬件和需求来倒推，可以直接按下面顺序看：

1. **只有端侧预算**：先看 E2B  
2. **端侧但想更强**：看 E4B  
3. **需要统一多模态结构**：看 12B Unified  
4. **需要更高吞吐的强推理能力**：看 26B A4B  
5. **明确追求 dense 大模型能力**：看 31B Dense

---

## 7. 参考资料

1. Google AI for Developers, Gemma 4 model overview  
   https://ai.google.dev/gemma/docs/core
2. Google AI for Developers, Gemma 4 model card  
   https://ai.google.dev/gemma/docs/core/model_card_4
3. Hugging Face, Gemma 4 collection  
   https://huggingface.co/collections/google/gemma-4
4. Hugging Face, Gemma 4 12B model card  
   https://huggingface.co/google/gemma-4-12B
