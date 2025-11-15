# R1-Omni 复现指南：基于强化学习的可解释全模态情感识别

[![论文](https://img.shields.io/badge/论文-PDF-red)]()
[![GitHub](https://img.shields.io/badge/官方仓库-blue)](https://github.com/HumanMLLM/R1-Omni)

基于论文《R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning》的完整复现指南。

## 📖 项目概述

R1-Omni 是首个将**可验证奖励强化学习（RLVR）**应用于全模态大语言模型的情感识别研究，显著提升：
- 🧠 **推理能力**：清晰分析视觉和音频模态贡献
- 🎯 **情感识别准确率**：在分布内数据表现优异  
- 🔄 **泛化能力**：在分布外数据集具有鲁棒性

## 🛠️ 环境搭建

### 方案一：原始环境搭建（推荐）
```bash
# 克隆 R1-V 项目（依赖项）
git clone git@github.com:StarsfieldAI/R1-V.git
cd R1-V

# 创建并激活环境
conda create -n r1-v python=3.11
conda activate r1-v

# 安装基础依赖
bash setup.sh
pip install timm imageio decord ipdb  
pip install "moviepy<2.0.0"
```

### 方案二：简易环境搭建
```bash
# 方法1：使用环境配置文件
conda env create -f environment.yml

# 方法2：使用conda需求文件
conda create -n r1-v python=3.11
conda activate r1-v
conda install -r conda_requirement.txt

# 方法3：使用pip需求文件  
conda create -n r1-v python=3.11
conda activate r1-v
pip install -r pip_requirement.txt
```

## 📥 模型下载
模型huggingface地址
Whisper音频模型: https://huggingface.co/openai/whisper-large-v3

BERT文本模型: https://huggingface.co/google-bert/bert-base-uncased

SigLIP视觉模型: https://huggingface.co/google/siglip-base-patch16-224

R1-Omni主模型: https://huggingface.co/StarJiaxing/R1-Omni-0.5B/tree/main

推荐使用以下命令下载所需模型（请替换为你的本地路径）：

```bash
# 先建好python或conda环境
pip install -U huggingface_hub # 安装依赖
# Linux
export HF_ENDPOINT=https://hf-mirror.com
#windows
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载所有必要模型
huggingface-cli download --resume-download openai/whisper-large-v3 --local-dir /hy-tmp/openai/whisper-large-v3 --local-dir-use-symlinks False
huggingface-cli download --resume-download StarJiaxing/R1-Omni-0.5B --local-dir /hy-tmp/StarJiaxing/R1-Omni-0.5B --local-dir-use-symlinks False
huggingface-cli download --resume-download google/siglip-base-patch16-224 --local-dir /hy-tmp/google/siglip-base-patch16-224 --local-dir-use-symlinks False
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir /hy-tmp/google-bert/bert-base-uncased --local-dir-use-symlinks False
```

> **提示**：`--local-dir-use-symlinks False` 参数取消文件软连接，新手推荐使用。

## ⚙️ 路径配置

下载完成后，需要修改以下文件中的模型路径：

### 1. 修改模型配置文件
在 R1-Omni 模型目录的 `config.json` 中修改：
```json
{
  "mm_audio_tower": "/hy-tmp/openai/whisper-large-v3",
  "mm_vision_tower": "/hy-tmp/google/siglip-base-patch16-224"
}
```

### 2. 修改代码中的路径
**文件1：** `R1-Omni/inference.py`
```python
bert_model = "/hy-tmp/google-bert/bert-base-uncased"
```

**文件2：** `R1-Omni/humanomni/model/humanomni_arch.py` (第83行)
```python
bert_model = "/hy-tmp/google-bert/bert-base-uncased"
```

**文件3：** `R1-Omni/src/r1-v/src/open_r1/trainer/humanOmni_grpo_trainer.py` (第297行)
```python
bert_model = "/hy-tmp/google-bert/bert-base-uncased"
```

## 🚀 运行推理

使用以下命令进行情感识别推理：

```bash
python inference.py --modal video_audio \
  --model_path /hy-tmp/StarJiaxing/R1-Omni-0.5B \
  --video_path angry.mp4 \
  --instruct "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
```

https://github.com/user-attachments/assets/8c73cbe6-5f24-49a9-bef9-bff6c50e4580
### 预期输出格式
```
<think>In the video, a man in a brown jacket stands in front of a vibrant mural. He is wearing a pink shirt underneath his brown jacket, and his hair is dark and curly. His facial expression is complex, with wide eyes, slightly open mouth, raised eyebrows, and furrowed brows, revealing surprise and anger. His body language suggests he is facing an urgent situation, possibly communicating with others or confronting an authority figure. Overall, this man displays a strong emotional reaction, primarily anger, triggered by some unexpected event.</think>
<answer>angry</answer>
【在视频中，一名穿着棕色夹克的男子站在鲜艳的壁画前。他的面部表情复杂，睁大的眼睛、微张的嘴巴、扬起的眉毛和皱起的眉头，显示出惊讶和愤怒。他的身体语言表明他正面对紧急情况。在音频中，语调高亢且情绪化，语速很快。总体来看，该男子表现出强烈的情绪反应，主要是愤怒。】

```

https://github.com/user-attachments/assets/30a51132-a25c-4d8a-ab00-799e0b98a3a2

### 预期输出格式
```
<think>The video is set in a minimalist room with white walls and a blackboard filled with diagrams and notes, creating an atmosphere of concentration or classroom learning. Two male characters are dressed in uniform blue uniforms; the man on the left exhibits exaggerated facial expressions, laughing heartily and waving his arms, displaying extreme joy and excitement. In contrast, the man on the right maintains a more reserved demeanor, occasionally showing confusion or curiosity through his gestures, yet he never breaks from the formality of his speech.</think>
<answer>happy</answer>
【<think>视频背景设定在一间极简风格的房间内，墙壁洁白，黑板布满图表和笔记，营造出一种专注或课堂学习的氛围。两名男性角色身着蓝色制服；左侧男子面部表情夸张，开怀大笑并挥舞着双臂，展现出极度的喜悦和兴奋。相比之下，右侧男子则保持着更为内敛的姿态，偶尔通过手势流露出困惑或好奇，但他的言辞始终保持着正式。</think>
<答案>开心</答案>】

```

## 📊 性能表现

| 模型 | DFEW (UAR/WAR) | MAFW (UAR/WAR) | RAVDESS (UAR/WAR) |
|-------|----------------|----------------|-------------------|
| HumanOmni-0.5B | 19.44%/22.64% | 13.52%/20.18% | 9.38%/7.33% |
| EMER-SFT | 35.31%/38.66% | 28.02%/38.39% | 27.19%/29.00% |
| MAFW-DFEW-SFT | 44.39%/60.23% | 30.39%/50.44% | 30.75%/29.33% |
| **R1-Omni** | **56.27%/65.83%** | **40.04%/57.68%** | **44.69%/43.00%** |

## ❗ 常见问题

1. **路径配置错误**：确保所有模型路径都正确指向下载的本地目录
2. **环境依赖冲突**：如果方案二不行建议使用方案一的原始环境搭建方式


## 📝 引用

如果使用本工作，请引用：
```bibtex
@article{zhao2025r1omni,
  title={R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning},
  author={Zhao, Jiaxing and Wei, Xihan and Bo, Liefeng},
  journal={arXiv preprint},
  year={2025}
}
```

## 🙏 致谢
- 基于 [HumanOmni](https://github.com/HumanMLLM/HumanOmni) 构建


---


