# Cosmos Anime 0.8B

这是一个面向单张约 22 GB 显存、从零训练动漫文生图模型的第一版工程。它不是把现成模型改个输入层，而是把已经公开验证过的 Cosmos-Predict2 小型 T2I 结构，改造成适配 Wan2.2 高压缩 latent 和小型双向文本编码器的静态图模型。

当前代码已经包含：

- 约 0.805B 的 Cosmos-Predict2-style DiT 主干；
- 约 26M 的两层文本 conditioning adapter；
- Wan2.2 TI2V-5B `f16c48` VAE 静态图包装；
- 7 个固定比例的 512、768、1024 三档 resolution buckets；
- Danbooru 分类 tag 的动态 shuffle/dropout；
- 离线 latent 缓存、同 bucket batch、rectified-flow 训练；
- raw WebDataset 滚动编码、下一 shard 异步预取与 source cursor 恢复；
- 本地原子 checkpoint 与 Google Drive 后台镜像；
- BF16、SDPA、activation checkpointing、梯度累积和可选 8-bit AdamW。

## 最终结构

| 部件 | 选型 | 规格 |
|---|---|---|
| VAE | Wan2.2 TI2V-5B VAE | 空间压缩 16 倍，48 latent channels，静态图 `T=1` |
| Text encoder | T5Gemma-2 270M encoder only | 18 层，hidden 640，冻结、在线运行 |
| Text adapter | 自训练 transformer adapter | `640 → 1024`，2 层、16 heads，约 25.8M |
| DiT | Cosmos-Predict2 MiniTrainDIT-style | width 1280，27 blocks，20 heads，head dim 64 |
| 每个 DiT block | Cosmos 顺序 | self-attn → cross-attn → GELU FFN |
| Conditioning | Cosmos AdaLN-LoRA | rank 256，shared modulation + block-local low rank |
| Position | 静态图 2D axial RoPE | learnable frequencies，长宽均支持外推 |
| 目标 | Rectified flow | `x_t=(1-t)x₀+tε`，预测 `ε-x₀` |

`configs/cosmos_08b_anime.yaml` 的精确可训练参数量为：

```text
DiT（不含 text adapter）  804,748,048
text adapter               25,842,944
合计                       830,590,992
```

NVIDIA 没有公开 Cosmos 0.8B。官方最接近的是 Cosmos-Predict2-0.6B-Text2Image：1280 width、20 blocks、20 heads。这里保持其宽度、head geometry、MLP ratio 和 AdaLN-LoRA，只把深度扩到 27 层。Cosmos3 最小公开模型是 16B，把其双流 MoT 硬缩到 0.8B 没有经过验证，因此不作为第一版主干。

这份实现是静态图适配版，改了 latent channels、文本编码器和位置编码，不能直接加载 NVIDIA 的 0.6B checkpoint。

## 为什么这样处理图片和文本

结论是：**图片离线缓存 VAE latent；文本 hidden states 在线计算。**

图片侧不要在每个训练 batch 里运行 VAE。Wan VAE checkpoint 的 FP32 文件约 2.8 GB，训练时重复 encode 会浪费显存和大量算力。离线把最终 crop 编成 FP16 latent：

```text
768×768 图片：约 216 KiB / latent
1024×1024 图片：约 384 KiB / latent
```

训练时完全不加载 VAE。不要缓存裁好的 FP16 RGB tensor：它更大，而且仍然需要运行 VAE。

文本侧不要只缓存一份 encoder hidden state。Danbooru tag 最有用的正则化正是每轮 shuffle、局部 dropout 和不同 caption 配方；整句 embedding 一旦缓存，这些变化就被锁死。270M encoder 的 BF16 常驻量约 0.5 GB，冻结并在 `inference_mode` 下在线运行更合适。

数据量很大后，可以提前缓存 4–8 组 caption variant 的 token IDs，但仍然在线跑 encoder。不要先缓存单 tag embedding 再拼接：双向 encoder 中每个 token 的结果依赖完整上下文。

缓存 latent 后也不要直接 `flip(latent)`。Wan VAE 并不严格满足 `flip(encode(x)) == encode(flip(x))`。如果确实要增强，应该把翻转后的 RGB 作为另一个样本重新 encode；动漫数据默认不翻转更稳，文字、签名、服装和人物不对称细节都可能受损。

## 分辨率与比例

模型总像素步长是 32：Wan VAE 压缩 16 倍，DiT 再 patch 2。因此所有宽高均为 32 的倍数。

第一阶段建议从约 512² 像素预算开始：

```text
512×512
448×576  / 576×448
416×608  / 608×416
384×672  / 672×384
```

主训练阶段用约 768²：

```text
768×768
672×896  / 896×672
640×960  / 960×640
576×1024 / 1024×576
```

最后才做约 1024² 的短程 high-resolution fine-tune：

```text
1024×1024
864×1152 / 1152×864
832×1248 / 1248×832
736×1312 / 1312×736
```

在 22 GB 卡上，建议：

1. 用 512 桶完成架构和数据清洗验证，并承担大部分早期训练；
2. 切换 768 桶做主质量阶段，batch size 1、gradient accumulation 16；
3. 模型已经稳定后，再用 1024 桶做少量精修。

768 桶经过 VAE 和 patch 后只有约 576–600 个图像 token；1024 桶约 943–1024 token。不同阶段需要分别从原图生成 latent，不能直接把低分辨率 latent 放大。

## Danbooru 数据建议

manifest 会保留 `artist`、`character`、`copyright`、`general`、`meta`、`rating` 和 `quality` 等字段。推荐 caption 采样混合：

```text
65% 纯 Danbooru tags
20% tags + 自然语言 caption
15% 纯自然语言 caption
```

第一版默认规则：

- general tag 独立 dropout 10%；
- character tag dropout 2%；
- general tag 每次采样重新排序；
- 最多 128 个 tag；
- 输入文本时将普通 tag 的下划线替换为空格；
- CFG 整句 dropout 15%。

正式训练前至少处理：

- exact hash 和感知近重复去重；
- 按重复簇切 train/validation，不能逐图随机切；
- alias/deprecated tag 归一化；
- 删除 `tagme`、`source_request`、`duplicate` 等非视觉 meta；
- 单独保存 rating，不要让安全等级静默混合；
- 记录来源和授权状态；数据包可下载不代表其中每幅图都允许任意用途；
- 严格过滤违法内容，并遵守训练和发布所在地的法律。

建议先用约 2,000 张有代表性的动漫图做 Wan VAE 重建验收，重点看小脸、眼睛、线稿、文字、高饱和色块和横竖构图。通过后再缓存完整数据。

当前缓存格式是“一图一个 safetensors”，适合先跑通和中等数据集。达到百万图规模时，应改成按 bucket 分组的无压缩 WebDataset tar shards，避免大量小文件随机 I/O；模型和 manifest 接口可以保持不变。

## 安装

建议 Linux 或 WSL2；Windows 原生也能运行核心 PyTorch 代码，但 8-bit optimizer 和高效 attention 的安装通常没有 Linux 稳定。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

训练环境应安装训练依赖：

```powershell
pip install -e ".[train]"
```

获取 Wan2.2 源码和 VAE：

```powershell
git clone https://github.com/Wan-Video/Wan2.2 external/Wan2.2
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir models/Wan2.2-TI2V-5B
```

T5Gemma-2 使用 Gemma license，需要先在其 Hugging Face 页面接受条款并登录：

```powershell
hf auth login
```

## Colab 滚动流式训练

Colab 部署入口是 [`notebooks/colab_rolling_train.ipynb`](notebooks/colab_rolling_train.ipynb)，完整说明见 [`docs/COLAB.md`](docs/COLAB.md)。notebook 会挂载 Drive、安装项目、登录 Hugging Face、准备 Wan/T5、生成 AnimeTimm shard list、运行环境预检，然后以 `--resume auto` 启动 rolling raw 训练。

L4 24GB + 外部 FlashAttention-2 的专用配置是
`configs/colab_l4_fa2_24gb.yaml`，安装、FA2/SDPA 对照 benchmark 和
Drive 恢复说明见 [`docs/L4_FA2.md`](docs/L4_FA2.md)。该配置将本地
checkpoint 原子写入 `/content/checkpoints_l4_fa2`，再后台镜像到
`/content/drive/MyDrive/cosmos`。

默认 notebook 是隔离的单 shard smoke run，只训练 8 个 optimizer step。验证通过后再将 `SMOKE_RUN=False`、`SHARD_LIMIT=None` 并切换到正式数据集。不要在同一个正式 checkpoint 任务中途改变 shard list。

## 数据准备

图片可以有同名 `.json` 或 `.txt` sidecar。`.txt` 会作为 general tags；`.json` 可以使用：

```json
{
  "general_tags": ["1girl", "solo", "blue_hair"],
  "character_tags": ["hatsune_miku"],
  "copyright_tags": ["vocaloid"],
  "artist_tags": ["example_artist"],
  "rating": "safe",
  "quality": "high"
}
```

先生成 512 桶 manifest：

```powershell
python scripts/prepare_manifest.py `
  --images D:\datasets\anime `
  --output data/source_512.jsonl `
  --resolution-stage 512
```

然后离线缓存 Wan latent：

```powershell
python scripts/cache_latents.py `
  --manifest data/source_512.jsonl `
  --output-dir data/cache_512 `
  --wan-repo external/Wan2.2 `
  --vae-checkpoint models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

把配置中的 `data.manifest` 指向 `data/cache_512/manifest.jsonl` 后训练：

```powershell
python scripts/train.py --config configs/cosmos_08b_anime.yaml
```

查看精确参数量：

```powershell
python scripts/inspect_model.py --config configs/cosmos_08b_anime.yaml
```

## 22 GB 显存设置

默认配置以以下组合为目标：

- BF16 model 和 activation；
- T4/V100 使用 FP32 主权重 + FP16 autocast/GradScaler，避免 FP16 参数无法 unscale；
- PyTorch SDPA；
- 每层 activation checkpoint；
- micro-batch 1；
- gradient accumulation 16；
- 离线 latent 模式不加载 VAE；rolling raw 模式只在 block 边界换入 Wan encoder；
- T5Gemma encoder 冻结；
- 优先使用 8-bit AdamW。

本地配置可以显式允许在 `bitsandbytes` 不可用时退回 PyTorch AdamW，但所有 Colab 配置默认禁止 fallback，避免额外约 5–7 GB optimizer state 直接导致 OOM。显存不足时依次减压：

1. 先用 512 桶；
2. 将 text encoder 放到 CPU（会明显变慢）；
3. 降低文本最大长度，从 192 改到 128；
4. 先用官方 0.6B 深度，即把 `depth` 从 27 改成 20。

从零训练 0.8B 的时间成本远大于“能塞进显存”。先在 1,000–10,000 张小数据上验证 loss、过拟合样本和解码质量，再投入完整 Danbooru 数据。

## 已验证

仓库的 CPU smoke tests 覆盖：

- 方形和非方形 latent 前向形状；
- 文本 padding mask；
- activation checkpoint 反向；
- rectified-flow 两端点；
- bucket 选择和 Danbooru caption 规则；
- 27 层正式配置的 meta-device 参数精算。
- rolling raw block、HTTP Range 恢复、原子 checkpoint mirror 和 Colab preflight。

运行：

```powershell
pytest
```

本机 CUDA 性能 smoke：

```powershell
python scripts/gpu_smoke.py `
  --config configs/cosmos_08b_anime.yaml `
  --pixel-size 768x768 `
  --precision float16 `
  --parameter-precision float32 `
  --text-length 192 `
  --sdpa-backend efficient `
  --gradient-checkpointing `
  --optimizer adamw8bit
```

Tesla V100-SXM2-16GB 实测完整 0.83B 主干：768² 约 1.05 秒/step、峰值 7.94 GiB；1024² 约 1.17 秒/step、峰值 7.97 GiB。该结果包含 backward 与 8-bit optimizer step，不包含 T5/Wan 编码。V100 不适合 FA2/FA3，默认 PyTorch memory-efficient SDPA 即为本项目在该卡上的推荐后端。

## 主要参考

- [NVIDIA Cosmos-Predict2 官方代码](https://github.com/nvidia-cosmos/cosmos-predict2)
- [Cosmos-Predict2-0.6B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image)
- [Wan2.2 官方代码](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [Google T5Gemma-2-270M-270M](https://huggingface.co/google/t5gemma-2-270m-270m)
