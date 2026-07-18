# 项目交接 / TODO

最后整理：2026-07-18（本地 CUDA 性能与 optimizer smoke 完成后）

## 1. 项目目标

在单张约 22–24 GB 显存的显卡或 Colab L4/T4 上，从零训练一个偏动漫的文生图模型：

- 支持若干固定长宽比，不限定为 1:1。
- 主干采用 Cosmos-Predict2 小模型一派的 DiT。
- VAE 使用 Wan2.2 TI2V-5B 的 `f16c48` VAE。
- 文本编码器使用 T5Gemma-2 270M encoder-only。
- 数据主要来自 Danbooru。
- 图片离线或滚动编码成 latent；文本保留动态 tag shuffle/dropout。
- Colab 中异步下载下一 tar，同时训练当前 latent 块。

当前仓库是从空目录搭建的，现已初始化 Git，`main` 分支按可验证里程碑提交。

## 2. 已确定的模型方案

### DiT 主干

当前实现位于 `src/my_sd/models/cosmos_dit.py`。

- Cosmos-Predict2 MiniTrainDIT 风格。
- width：1280。
- depth：27。
- heads：20。
- head dim：64。
- MLP ratio：4。
- block 顺序：self-attention → cross-attention → FFN。
- QK RMSNorm。
- AdaLN-LoRA rank：256。
- 输入 latent channels：48。
- DiT patch：2×2。
- 静态图片使用 2D axial learnable RoPE。
- 支持非正方形 latent。
- 支持 activation checkpointing。
- 训练目标为 rectified flow / flow matching。

参数量：

```text
DiT（不含 text adapter） 804,748,048
text adapter              25,842,944
合计                      830,590,992
```

注意：NVIDIA 没有公开官方 Cosmos 0.8B。当前版本是在官方 0.6B 的 width/head/block 设计上把 20 层扩展到 27 层，因此是 Cosmos-style 0.8B，不能直接加载官方 0.6B checkpoint。

### Text Encoder

当前封装位于 `src/my_sd/encoders/text.py`，提取脚本位于 `scripts/extract_t5gemma_encoder.py`。

- 模型：`google/t5gemma-2-270m-270m`。
- 只保留约 270M 的 text encoder。
- hidden size：640。
- 最大长度默认 192；T4 配置为 128。
- encoder 冻结，训练自有的 2 层 text adapter：`640 → 1024`。
- 正式训练时在线批量编码文本，不固定缓存唯一 hidden state。
- 已修复一个重要问题：不能直接保留 `.get_encoder()` 返回的完整多模态模块；当前代码只取 `.get_encoder().text_model`，避免把 vision tower 一并常驻。

注意：

- T5Gemma 模型受 Gemma license 约束，Hugging Face 上需要先接受条款。
- `transformers` 对 T5Gemma2 的内部类和接口仍可能变化，真实环境必须固定版本并跑一次提取测试。

### VAE

当前封装位于 `src/my_sd/autoencoders/wan_vae.py`。

- 使用 Wan2.2 TI2V-5B 官方 `Wan2_2_VAE`。
- 静态图片作为 `T=1` 视频帧编码。
- 空间压缩 16 倍。
- latent channels：48。
- 训练主干时不需要 VAE decoder。
- 离线编码脚本支持删除 decoder，仅保留 encoder。

已处理：encoder-only 模式先在 CPU 加载完整 VAE，删除 decoder/conv2 后才搬到目标设备，避免完整 decoder 占用 GPU 峰值；`move_to()` / `offload_to_cpu()` 会同时移动 model 与 `vae.scale` tensors。

仍需处理：还没有在真实 Wan2.2 checkpoint 上做端到端 GPU 验证。

## 3. 已完成的分辨率 buckets

实现位于 `src/my_sd/data/buckets.py`。

所有宽高均为 32 的倍数，因为 Wan VAE 压缩 16 倍，DiT 再 patch 2。

### 512 阶段

```text
512×512
448×576  / 576×448
416×608  / 608×416
384×672  / 672×384
```

### 768 阶段

```text
768×768
672×896  / 896×672
640×960  / 960×640
576×1024 / 1024×576
```

### 1024 短程精修

```text
1024×1024
864×1152 / 1152×864
832×1248 / 1248×832
736×1312 / 1312×736
```

建议顺序：先 512 跑通和预训练，主要质量阶段用 768，模型稳定后才做少量 1024。

不同分辨率阶段必须重新从 RGB 编码 latent，不能放大低分辨率 latent。

## 4. 已完成的数据与 caption 功能

### Danbooru caption

实现位于 `src/my_sd/data/captions.py`。

- 支持 general、character、copyright、artist、meta、rating、quality。
- general tag dropout。
- character tag dropout。
- general tag 每次重新 shuffle。
- tag 去重、最大 tag 数限制、下划线转空格。
- CFG 整句 dropout 在训练循环中完成。

当前建议：

```text
general tag dropout    10%
character tag dropout   2%
CFG caption dropout    15%
```

文本侧应保持在线编码或缓存多份 token IDs，不要永久缓存唯一 text hidden state，否则 tag shuffle/dropout 会失效。

### 本地 manifest / latent

- `scripts/prepare_manifest.py`：从本地图片和 sidecar 生成 manifest。
- `scripts/cache_latents.py`：批量生成 Wan latent。
- `src/my_sd/data/latent_dataset.py`：读取 safetensors latent，按 bucket batch。
- 已禁止直接翻转缓存后的 latent；若需要翻转，应先翻 RGB，再作为独立样本编码。

### latent tar 流式训练

实现位于 `src/my_sd/data/tar_stream.py`。

- `LatentTarWriter`：写入无压缩 tar。
- 每个样本由：

```text
sample_id.latent.npy
sample_id.json
```

组成。

- `StreamingLatentDataset`：顺序读取 latent tar。
- 支持 shard shuffle、样本 buffer shuffle、动态 caption。
- 支持流式数据 cursor，用于 checkpoint 恢复。
- `AsyncShardPrefetcher`：后台线程下载后续 tar 到本地缓存目录。
- 支持本地路径、HTTP(S) 和：

```text
hf://datasets/OWNER/REPO/path/to/shard.tar
```

### Colab raw tar → latent tar producer

实现位于 `scripts/colab_encode_shards.py`。

- 从 shard list 异步预取 raw tar。
- 读取同 key 的图片和 `.json` / `.txt` sidecar。
- 根据比例选择 bucket。
- 同 bucket 批量跑 Wan VAE。
- OOM 时退回逐图编码。
- 输出永久 latent tar。
- 可上传到 Hugging Face dataset repo。
- 根据远端文件名跳过已经完成的 shard，可继续中断任务。

独立 producer 仍只在编码当前 tar 时预取下一 tar；正式的 `rolling_raw` 训练后端已经能在训练当前 latent block 时后台下载下一 raw tar。

## 5. Danbooru 流式数据调研结论

### 最容易立刻接入：AnimeTimm WebDataset

推荐先用：

- `animetimm/danbooru-wdtagger-v4-w640-ws-50k`：快速 smoke test。
- `animetimm/danbooru-wdtagger-v4-w640-ws-150k`：小规模试训。
- `animetimm/danbooru-wdtagger-v4-w640-ws-full`：正式 512 阶段。

Full 数据集约：

```text
总图片       5,914,596
train        5,321,713
train 大小   约 318 GB
全部大小     约 353.5 GB
单个 tar     约 1.6 GB
```

每个 WebDataset 样本已经包含：

```text
image.webp
image.json
```

JSON 有 `id`、`width`、`height`、`rating`、`general_tags`、`character_tags`。这和当前 `iter_raw_tar()` 的同 key sidecar 格式兼容，最省开发时间。

限制：

- 图片被缩到 `min(width, height) <= 640`，最适合 512 阶段。
- 不适合作为高质量 1024 阶段的唯一数据源。
- 默认数据不含完整 artist/copyright tag。
- gated、含敏感内容、license 为 `other`；必须在 HF 接受条款，并自行处理使用范围和过滤。

### 更高分辨率但更重：DeepGHS 2024

- 图片：`deepghs/danbooru2024-webp-4Mpixel`。
- 约 805 万张。
- `images/0000.tar` 等，每个约 1.7 GB。
- tar 内主要是 `Danbooru_ID.webp`，**没有同 key caption JSON**。
- 旁边的 `.json` 是文件 offset/size/hash 索引，不是训练标签。
- 标签可用 `p1atdev/danbooru-2024`，约 804 万行、35 个 Parquet，包含完整分类 tag。
- 必须按 Danbooru ID 联结图片和 metadata；当前代码尚未实现这层。

建议首次把 metadata 流式扫一遍，按图片 shard 或 `id % 1000` 生成紧凑 metadata 分片，不要训练每轮重复扫描完整 Parquet。

### 用户可能记得的 toolkit

- `CheeseChaser`：借助 `hfutils` 的 tar index 和 HTTP Range，只抓 tar 中指定图片。适合按角色/标签补采几千到几万张，不适合全量基础训练时发出数百万次 Range 请求。
- `waifuc.DanbooruSource`：直接迭代 `ImageItem(image, meta)`，适合站点定向采集和清洗，不适合百万级全量训练吞吐。
- `HakuBooru`：本地数据库筛选/导出，不是远程流式训练器。
- Pybooru/pydanbooru：API wrapper，不应用来压原站 API 做百万级训练。

主训练仍应使用：

```text
Hugging Face tar shard
→ 本地顺序读取
→ 异步预取下一 shard
→ 分块 VAE encode
→ 训练当前 latent 块
```

参考：

- https://huggingface.co/datasets/animetimm/danbooru-wdtagger-v4-w640-ws-full
- DeepGHS 4MP/约 805 万图数据源、外置 Danbooru 2024 Parquet 标签的
  1000 桶索引、异步 tar 下载与 rolling Wan 编码已接入；使用说明见
  `COLAB_DEEPGHS.md`。
- DeepGHS Colab 全自动 bootstrap 已完成：自动准备 Wan 源码/VAE、
  T5Gemma encoder、Drive 元数据索引、全量 shard 清单、preflight 和训练。
- DeepGHS L4 pipeline 已增加 CPU WebP 预解码、Wan batch 16 自适应探测、
  OOM 二分回退并记忆安全 batch、
  pinned non-blocking H2D、逐 shard 图片总进度、VAE/T5 进度，以及每个
  optimizer step 的输入等待比例和 CUDA 峰值日志。
- DeepGHS L4 正式配置已关闭全模型 gradient checkpointing，并接入 W&B
  在线指标（loss/LR/吞吐/输入等待/CUDA 峰值/数据游标）。
- https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel
- https://huggingface.co/datasets/p1atdev/danbooru-2024
- https://github.com/deepghs/cheesechaser
- https://deepghs.org/waifuc/main/tutorials/crawl_images/index.html
- https://huggingface.co/docs/hub/en/datasets-webdataset

## 6. 已完成的训练功能

训练入口：`scripts/train.py`。

- manifest latent 后端。
- streaming latent tar 后端。
- rectified-flow batch 和 loss。
- BF16 / FP16 autocast。
- 计算精度与参数精度分离：T4/V100 默认 FP32 主权重 + FP16 autocast/GradScaler；L4 使用 BF16 参数与计算。
- activation checkpointing。
- gradient accumulation。
- gradient clipping。
- cosine learning-rate schedule。
- 8-bit AdamW；本地配置可显式 fallback，Colab 配置要求失败时立即停止。
- 文本 window cache：
  - 一次读取 128/256 个 caption。
  - T5 以 16/32 的 batch 批量编码。
  - hidden state 暂存 CPU。
  - 可在 window 之间把 T5 offload。
- 保存 model、optimizer、scheduler、scaler、RNG 和数据 cursor。
- `--resume PATH` 和 `--resume auto`。

已经修复：

- 修复 T4/V100 直接用 FP16 参数时 GradScaler 在 optimizer step 报错；现在 FP16 计算默认保留 FP32 主权重。
- 数据模块缺少 `random` import。
- T5Gemma 错误保留 vision tower。
- meta device + `to_empty()` 导致非 module 参数未初始化；现改为直接在目标 device/dtype 构造。

## 7. 已验证内容

CPU smoke tests 覆盖：

- 方形/非方形模型 forward。
- activation checkpoint backward。
- text padding mask。
- flow matching 端点。
- bucket 选择。
- Danbooru caption 规则。
- latent tar round-trip。
- streaming cursor。
- text window cache。
- YAML 继承和正式配置参数量。

最后一次已记录结果（本机 CUDA PyTorch 环境）：

```text
39 passed
```

此外已在 Tesla V100-SXM2-16GB、PyTorch 2.6.0+cu124 上完成真实 0.83B DiT 的 forward/backward、GradScaler 与 AdamW8bit optimizer step；仍不是 Wan + T5 + DiT 的完整端到端验证。

### 本地 GPU 性能记录

测试入口：`scripts/gpu_smoke.py`。条件为完整 27 层、830,590,992 参数、batch 1、text length 192、activation checkpointing、FP32 主权重、FP16 autocast、bitsandbytes AdamW8bit：

```text
768×768：  1.052 s/step，0.950 step/s，峰值 allocated 7.941 GiB
1024×1024：1.174 s/step，0.852 step/s，峰值 allocated 7.973 GiB
```

数字只包含 DiT、loss、backward、GradScaler 和 optimizer step，不包含数据下载、T5 编码与 Wan 编码。rolling 模式会在 Wan 编码阶段暂停训练并换出 encoder，因此这些峰值不能直接相加。

attention 后端实测：

- Windows PyTorch 未编译 FlashAttention，且 V100 是 SM70，不适合安装 FA2/FA3。
- PyTorch memory-efficient SDPA 可用；在 width 1280 / depth 2 / 512×768 对照中约 `0.067 s/step`，强制 math 约 `0.090 s/step`，快约 26%。
- 默认 `auto` 已会选择可用的高效 kernel；无需在 V100 上强装 FlashAttention。
- 已增加外部 FA2 packed-QKV 图像 self-attention backend、L4 专用配置和
  `scripts/benchmark_l4_attention.py`；本机 V100 无法执行外部 FA2，
  真实加速比例必须在 L4 上运行对照脚本后填写。

## 8. Colab 配置

- `configs/colab_l4.yaml`
  - 27 层、约 0.83B 总参数。
  - BF16。
  - BF16 参数与计算，不启用 GradScaler。
  - text window 256 / encode batch 32。
  - gradient accumulation 16。

- `configs/colab_t4.yaml`
  - 20 层，接近官方 0.6B 规模。
  - FP32 主权重 + FP16 autocast/GradScaler。
  - text max length 128。
  - text window 128 / encode batch 16。
  - T5 window 之间 offload。
  - gradient accumulation 32。

- `configs/colab_l4_rolling.yaml` / `configs/colab_t4_rolling.yaml`
  - 直接消费 raw WebDataset，训练当前 latent block 时预取下一 tar。
  - 严格要求 8-bit AdamW，不允许静默退回 FP32 optimizer state。
  - `notebooks/colab_rolling_train.ipynb` 提供挂载、下载、预检、smoke 和 resume 入口。

## 9. P0：当前状态与下一优先级

### P0.1 单会话滚动流水线

**状态：首版已实现，待真实 GPU 集成验证。** `RollingWanDataset` 已在单进程内完成 raw tar 预取、分 bucket Wan 编码、encoder CPU/GPU 换入换出、按梯度累积倍数交付 latent block，以及基于 source cursor 的 checkpoint 恢复。滚动模式强制 `prefetch_shards=1`，DiT 与 optimizer 常驻 GPU；T5 在 Wan 编码阶段前已 offload。当前实现把滚动状态合并在原子 checkpoint 的 `data_cursor` 中，没有另设独立 rolling state JSON。

目标流程：

```text
首次下载 raw-0
→ Wan 编码 latent block-0
→ 训练 latent block-0，同时后台下载 raw-1
→ optimizer step 边界暂停
→ T5 下 CPU
→ Wan encoder 上 GPU，编码下一块
→ Wan 下 CPU
→ 删除已消费 raw/latent
→ 继续训练，同时下载再下一 tar
```

实现落点：

- `src/my_sd/data/raw_stream.py`：`RollingWanDataset`。
- `WanImageVAE.move_to()` / `offload_to_cpu()` 同时移动 model 与 scale tensors。
- block size 必须是 gradient accumulation 的倍数，VAE 只会在 optimizer step 边界换入。
- raw tar 在 shard 消费完成后删除；中断恢复时重读当前 shard，并按 checkpoint cursor 跳过已训练样本。
- 预取深度固定为 1，避免 `/content` 被多个 1.6 GB tar 和 checkpoint 撑满。

### P0.2 异步下载健壮性

**状态：核心功能已实现。** `AsyncShardPrefetcher` 当前已有：

- HTTP Range 断点续传与 `.part` 恢复。
- 自动重试和指数退避。
- Content-Length / Content-Range 大小校验。
- `minimum_free_gb` 与 `max_cache_gb` 磁盘预算。
- rolling raw 模式只保留一个预取 tar。
- rolling 与 latent tar 两种后端共用同一套 YAML 下载限制解析。

仍缺可选 SHA256 校验，以及真实 Hugging Face 大文件中断恢复压测。下载下一 raw tar 已发生在 DiT 消费当前 latent block 的阶段。

### P0.3 checkpoint 本地保存后镜像 Drive

**状态：首版已实现，待 Colab Drive 实测。** checkpoint 先写本地隐藏临时目录，再原子 rename；`AsyncCheckpointMirror` 单线程后台复制到 mirror，完成后原子更新 `latest.txt`。`--resume auto` 会同时搜索本地与 mirror，并忽略不完整 checkpoint；`keep_last_checkpoints` 同时控制两侧保留数量。

当前实现流程：

```text
/content/checkpoints_*
→ 完整写入临时目录
→ 原子 rename
→ 后台复制到 /content/drive/MyDrive/...
```

配置项：

- `checkpoint_mirror_dir`。
- `keep_last_checkpoints: 2`。
- `auto resume` 同时搜索本地和 Drive mirror。
- mirror 使用临时目录，完成后才更新 `latest.txt`。

真实 Colab 仍需验证 Drive 断开、重连和空间不足时，后台 mirror 的异常能否清晰上报。

### P0.4 真实 GPU 集成验证

本地检测到 16 GB Tesla V100。完整 27 层 DiT 已通过 768/1024、AdamW8bit optimizer step 和显存峰值验证；工作区没有 Wan2.2/T5 checkpoint，因此仍不能替代目标 L4/T4/22–24 GB 上的完整 encoder + rolling 集成验证。

至少完成：

1. Wan VAE 对 8–16 张不同长宽比图片编码。
2. 检查 latent 是 `[B, 48, H/16, W/16]`。
3. T5 encoder-only 输出 `[B, L, 640]`。
4. 已完成：完整 27 层 DiT forward/backward、GradScaler、gradient clipping 路径同类验证及 AdamW8bit step。
5. 已完成本地 V100 27 层峰值；仍需 L4 BF16 实机复测。
6. 20 层 T4 显存峰值测试。
7. 已完成：Colab 配置在 bitsandbytes 不可用时直接失败，不再静默退回 FP32 AdamW。
8. 分别测下载、VAE encode、T5 encode、DiT forward/backward 的吞吐。

## 10. P1：重要但可晚于首次跑通

以下项目已完成：AnimeTimm HF shard list CLI、默认拒绝缺失 metadata、rating 白名单、stream cursor 派生的确定性 caption/CFG dropout，以及最终 checkpoint 的 epoch/cursor 语义修复。

- 增加最小尺寸、score、banned/deleted、AI-generated、duplicate 等过滤。
- 对 DeepGHS 图片实现 Danbooru ID → metadata 分片联结。
- 将 `iter_raw_tar()` 改成按连续同 key 成员流式组装；当前使用 `getmembers()` 和整 tar grouped dict，会额外占 RAM。
- 处理 text window 预读和 checkpoint cursor 的严格一致性。
- 大规模 latent 不建议全部永久上传：512 的 48-channel FP16 latent 可能比 640px WebP 源图更大。先实际测单 shard 的输入/latent 大小，再决定永久缓存还是只做滚动临时缓存。
- 增加 validation 数据和固定 prompt 采样。
- 增加 loss、吞吐、显存和数据跳过率日志。

## 11. P2：尚未实现的完整模型能力

- 推理/采样脚本。
- ODE / flow sampler。
- CFG 推理。
- Wan decoder 输出图片。
- 固定验证 prompt/grid。
- EMA。
- checkpoint 转换/发布。
- FSDP/多 GPU。
- 数据 fuzzy dedup。
- aesthetic / quality scorer。
- 自然语言 caption 混合。
- 1024 高质量数据源和精修 recipe。

## 12. 已知风险

- 从零训练 0.8B 的时间成本远高于“能否塞进 22 GB 显存”；单卡完整基础训练可能需要非常长时间。
- 8-bit optimizer 是单卡可行性的关键之一。
- Wan latent 48 通道且空间压缩 16 倍；计算友好，但永久 latent 存储未必比压缩 WebP 小。
- AnimeTimm 640px 数据适合 512 预训练，不足以单独支撑 1024 质量。
- Danbooru 数据包含敏感内容和无法统一确认的图片权利。数据集仓库的 license 不等于其中每张图片都获得同等授权；训练、发布和商业使用前需自行评估。
- 当前模型结构是自研适配版，不能把 Cosmos/Wan 的生成主干 checkpoint 直接拼接进来。
- 还没有真实长时间训练曲线，所有超参数均是第一版工程起点。

## 13. 快速接手命令

安装：

```bash
pip install -e ".[train,test]"
```

检查参数量：

```bash
python scripts/inspect_model.py --config configs/cosmos_08b_anime.yaml
```

运行测试：

```bash
pytest
```

提取 T5Gemma encoder：

```bash
python scripts/extract_t5gemma_encoder.py \
  --model-id google/t5gemma-2-270m-270m \
  --output /content/models/t5gemma2-270m-encoder
```

编码 raw tar：

```bash
python scripts/colab_encode_shards.py \
  --shard-list /content/raw_shards.txt \
  --wan-repo /content/Wan2.2 \
  --vae-checkpoint /content/models/Wan2.2_VAE.pth \
  --resolution-stage 512 \
  --output-dir /content/latent_output
```

训练 L4：

```bash
python scripts/train.py \
  --config configs/colab_l4_rolling.yaml \
  --resume auto
```

训练 T4：

```bash
python scripts/train.py \
  --config configs/colab_t4_rolling.yaml \
  --resume auto
```

## 14. 建议的第一天接手顺序

1. 已完成：初始化 Git 并提交恢复基线。
2. 已完成：本地 39 个 CPU/逻辑测试通过，并完成 0.83B DiT CUDA optimizer smoke。
3. 用 AnimeTimm 50k 的 1 个 tar 做真实 Wan encode smoke test。
4. 用小 depth 配置训练 100–500 step，确认 loss、resume 和非方形 batch。
5. 已完成本地部署入口与 preflight；仍需在 Colab 实测 rolling raw 下载、Wan 换入换出与 checkpoint mirror。
6. L4 上测 27 层峰值；若不稳定先用 20 层。
7. 先跑 50k/150k 验证数据和重建质量，再考虑 full 5.3M。
