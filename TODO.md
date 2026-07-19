# 项目交接 / TODO

最后更新：2026-07-19

## 当前结论

目标是用单张 Colab L4（实际可用约 22 GiB）从零训练动漫文生图模型。
当前正式方案：

- 主干：Cosmos-Predict2 MiniTrainDIT 风格，830,590,992 个可训练参数。
- VAE：Wan2.2 TI2V-5B `f16c48`，静态图片作为 `T=1`。
- Text encoder：T5Gemma-2 270M encoder-only，hidden size 640，冻结。
- Text adapter：2 层、`640 → 1024`，随 DiT 训练。
- 训练目标：rectified flow，DeepGHS L4 配置使用 logit-normal timestep。
- 分辨率：512/768/1024 各 7 个固定 ratio bucket；正式配置当前为 768。
- 数据：`deepghs/danbooru2024-webp-4Mpixel`，约 805 万张。
- L4：BF16、FlashAttention-2、AdamW8bit、关闭 gradient checkpointing。
- DiT：同 ratio microbatch 4 × gradient accumulation 4 = 有效 batch 16。
- Wan：encode batch 从 16 开始，OOM 后二分回退并记住安全值。

NVIDIA 没有发布官方 0.8B Cosmos checkpoint。这里保持公开 0.6B 的
width/head 几何并把深度扩到 27 层，所以是自研 Cosmos-style 0.8B，不与
官方 checkpoint 兼容。

## 已完成

### 模型与训练

- 非方形 2D axial RoPE、QK RMSNorm、AdaLN-LoRA、self/cross attention。
- SDPA 与外部 FlashAttention-2 packed-QKV self-attention 后端。
- BF16/FP16/FP32 精度路径、8-bit AdamW、cosine LR、梯度裁剪。
- rectified-flow batch/loss、uniform 与 logit-normal timestep。
- activation checkpoint 可配置；L4 正式配置明确关闭。
- loss 使用整个日志窗口均值；记录 gradient norm、samples/s、
  input-wait ratio、CUDA allocated/reserved。
- W&B 在线日志；run ID 持久化到 checkpoint mirror，重启续同一 run。

### 数据与流式流水线

- DeepGHS 1000 个 tar URL 自动生成，不会预先下载全部数据。
- 当前 tar 消费时，后台异步下载下一个 tar；预取深度固定为 1。
- 下载百分比、字节数、速度、耗时、重试和 cache hit 日志。
- tar 索引/图片扫描总数、通过/跳过数、速度、ETA、显存日志。
- CPU 有界预解码与 GPU Wan batch encode 重叠。
- 每 256 张形成滚动 latent block；不永久保存 8M latent。
- Wan 阶段：DiT 和 optimizer state 换到 CPU。
- DiT 阶段：Wan 换到 CPU，DiT 和 optimizer state恢复 GPU。
- Wan OOM 回退已避免在异常作用域内递归，从而不会保留失败激活。
- 七个 ratio 按形状组成 microbatch 4；断点使用保守游标，允许重启后
  少量重复，但不会越过尚未训练的分桶样本。
- caption 动态 tag shuffle/dropout 与确定性 CFG dropout。

### DeepGHS metadata

- metadata 改为图片仓库自身的 `metadata.parquet`，按 `id % 1000` 建索引。
- `_index_manifest.json` 标记 v2、来源和分区数。
- 自动识别旧的 17125 分区/异源索引并重建约 1000 个同源分区。
- 修复把一个 bucket 内多个 Parquet 文件误算成多个 bucket 的校验错误；
  已完成的 `deepghs_metadata_build` 会在重跑时直接复用。
- preflight 拒绝无 manifest、来源错误或分区数异常的索引。
- 每个 tar 解码前计算图片 ID/metadata 实际匹配率；低于 70% 立即报错，
  避免下载/编码数小时后才发现标签不匹配。

### checkpoint 与恢复

- checkpoint 先写本地隐藏临时目录，完成后原子 rename。
- model、optimizer、scheduler、scaler、RNG、epoch/micro-step/data cursor
  全部保存。
- 后台镜像 Google Drive，镜像完成后原子更新 `latest.txt`。
- `--resume auto` 同时检查本地和 Drive，只选择完整 checkpoint。
- bootstrap 现在始终以 `--resume auto` 启动。
- DeepGHS 配置每 250 optimizer step 保存，即每 4000 样本；本地和 Drive
  各保留最近 2 个。

### 推理闭环

- `scripts/sample.py` 可从 checkpoint 目录或 Drive checkpoint 根目录采样。
- 支持 Euler/Heun flow ODE、positive/negative CFG、固定 seed 和任意
  32 倍数宽高。
- T5 → DiT → Wan decoder 依次占用 GPU，不要求三个模型同时常驻。
- DeepGHS 正式训练每 1000 step 通过同一采样入口生成固定 seed 的低成本
  预览并上传 W&B；采样前训练状态换到 CPU，失败不会中断训练。

### 已修复的关键线上错误

- Wan import 不再执行依赖很重的 `wan/__init__.py`，避免
  `easydict`/`diffusers` 等无关导入错误。
- Hugging Face gated 下载使用认证 client，403 会给出明确权限提示。
- T5/Wan/text-window 从 `inference_mode` 改为 `no_grad`；训练入口同时
  clone 输入，修复 `Inference tensors cannot be saved for backward`。
- FP16 主权重与 GradScaler 不兼容路径、T5 vision tower 误保留、
  VAE decoder GPU 峰值、optimizer CPU/GPU 状态换入均已修复。

## 已验证

当前完整逻辑测试：

```text
55 passed
```

覆盖模型形状/反向、flow、caption、bucket、raw/latent tar、断点游标、
metadata 匹配、下载恢复、checkpoint mirror、preflight、采样器和配置继承。

本机 CUDA（Tesla V100、PyTorch 2.6、memory-efficient SDPA）：

```text
完整 27 层 / 830,590,992 参数
896×640，microbatch 4
无 gradient checkpoint
FP16 参数/计算，AdamW8bit
预热后约 0.408 s/iteration
peak allocated 10.375 GiB
peak reserved 11.291 GiB
```

V100 不支持本项目外部 FA2，因此 FA2 必须由 L4 preflight 和
`scripts/benchmark_l4_attention.py` 在 Colab 验证。用户的真实 Colab
日志已经验证了：

- L4 识别和 FA2 加载成功。
- DiT/optimizer → CPU，Wan → GPU 成功。
- Wan batch 16 编码 256 张成功，CUDA reserved 曾达到约 20.6 GiB。
- Wan → CPU，DiT/optimizer → GPU，T5 256 条编码成功。
- 长跑已到约 9,800 optimizer step；截取的 1,704 step 中 loss 均值
  0.3606、grad norm 均值 0.342，没有出现发散。
- 长跑均值约 6.58 秒/step；纯 DiT optimizer step 约 1.9 秒，但每约
  15 step 需要约 73 秒滚动解码/编码下一 latent block，截取日志中
  `input_wait` 占总 wall time 约 70.9%，当前吞吐瓶颈是在线 Wan 阶段。
- 早先的 inference tensor 反向阻断已修复并有回归测试。

## Colab 下一次运行

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

```bash
%cd /content/my_cosmos
!git pull
!uv sync --extra train --extra fa2
!uv run python scripts/bootstrap_deepghs_colab.py
```

第一次拉取本修复后，bootstrap 会发现旧 metadata 索引不合格，下载
DeepGHS 自带的 `metadata.parquet` 并重建 v2 索引。这是一次性工作，不能
跳过；否则旧索引会造成大量图片无标签。之后会自动 resume。

先做两 shard、8 step 的隔离集成 smoke：

```bash
!uv run python scripts/bootstrap_deepghs_colab.py --smoke-shards 2
```

smoke 会自动写到 Drive 的 `cosmos/smoke`，正式训练写到 `cosmos` 根目录，
不会串游标。正式训练本身仍不要中途改变 shard list。

采样：

```bash
!uv run python scripts/sample.py \
  --config configs/colab_l4_fa2_deepghs.yaml \
  --checkpoint /content/drive/MyDrive/cosmos \
  --prompt "1girl, solo, detailed eyes, best quality" \
  --negative-prompt "low quality, blurry" \
  --width 768 --height 768 --steps 28 --solver heun \
  --guidance-scale 5 \
  --output /content/drive/MyDrive/cosmos/sample.png
```

## 尚未完成 / 后续优先级

这些不是当前训练启动的阻断项：

1. 在 L4 拉取最新代码并恢复训练后，确认下一个 1000-step 边界能生成首张
   `preview/generated` W&B 图片，并记录采样耗时和峰值显存。
2. 用 `scripts/benchmark_l4_attention.py` 记录该 Colab runtime 的 FA2 对
   SDPA 实测；不同 PyTorch/driver 下结果会变化。
3. 将当前单张固定 prompt 预览扩展为多维 validation grid，并评估是否值得
   使用独立 GPU/进程做真正不暂停训练的异步采样。
4. 增加数据 quality/aesthetic scorer、fuzzy dedup、AI-generated 策略和
   分类 tag 映射；DeepGHS 自带 `tag_string` 当前可训练，但类别前缀较弱。
5. 512 预训练 → 768 主训练 → 1024 少量精修的真实长期曲线仍需实验决定。
6. EMA/post-hoc EMA、checkpoint 发布转换、多 GPU/FSDP 尚未实现。
7. 可选 shard SHA256 校验和真实大文件断网恢复压测尚未完成。

## 风险

- 从零训练 0.83B 的时间成本远高于“能否塞入 22 GiB”；单 L4 完成基础
  预训练会非常慢。
- DeepGHS license 为 `other` 且包含敏感内容；训练、发布和商业使用前需
  自行核对数据集条款与图片权利。
- 当前是自研架构，不能直接套用 Cosmos/Wan 生成主干 checkpoint。
- 超参数是工程起点，不是已经由长期训练验证的最终 recipe。
