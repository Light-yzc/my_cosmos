# L4 24GB + FlashAttention-2 训练

推荐配置是 `configs/colab_l4_fa2_24gb.yaml`。它使用完整 27 层
Cosmos-style 0.83B 主干、BF16、外部 FlashAttention-2 图像
self-attention、AdamW8bit、activation checkpointing 和 768 分辨率档。

## Colab / Linux 安装

仓库已经有 `pyproject.toml`，不要再次执行 `uv init`。首次安装分两步，
先准备 PyTorch 和普通训练依赖，再编译或安装 FA2：

```bash
cd /content/my_cosmos
uv sync --extra train --extra test
MAX_JOBS=4 uv sync --extra train --extra test --extra fa2
```

项目在 Linux 上固定使用 PyTorch 2.9.0 + CUDA 12.8，并固定
FlashAttention 2.8.3。这个组合与当前 Colab 的 CUDA 12.8 toolkit
一致，并有 CPython 3.12 / Linux x86_64 的官方 FA2 wheel。不要使用
`--torch-backend=auto`，否则 uv 可能选择比本机 `nvcc` 更新的 CUDA
13 PyTorch，导致 FA2 回退源码编译后报 CUDA mismatch。

如果之前已经执行过 `uv init` / `uv sync` 并装出了
`torch 2.13+cu130`，更新项目后执行：

```bash
uv lock --upgrade-package torch --upgrade-package flash-attn
MAX_JOBS=4 uv sync --extra train --extra test --extra fa2 --reinstall-package flash-attn
```

如果 `flash-attn` 报找不到 `CUDA_HOME` 或 `nvcc`，说明环境只有 CUDA
runtime，没有开发工具链。优先使用匹配当前 PyTorch/CUDA/Python ABI 的
预编译 wheel；否则换到带 CUDA toolkit 的 NVIDIA PyTorch `devel`
容器。不要为了编译 FA2 随意替换已工作的 CUDA PyTorch。

验证：

```bash
uv run --extra train --extra fa2 python -c \
  "import torch, flash_attn; print(torch.__version__, torch.version.cuda, flash_attn.__version__, torch.cuda.get_device_name())"
```

## FA2 对照 benchmark

以下脚本会启动两个独立进程，以相同的完整 0.83B 模型、BF16、
AdamW8bit 和 768×768 输入分别测试 PyTorch SDPA 与外部 FA2：

```bash
uv run --extra train --extra fa2 python scripts/benchmark_l4_attention.py \
  --config configs/colab_l4_fa2_24gb.yaml \
  --pixel-size 768x768 \
  --warmup 3 \
  --iterations 10
```

输出最后三行类似：

```text
sdpa_seconds=...
fa2_seconds=...
fa2_speedup=1.xxx (...%)
```

外部 FA2 当前只替换图像 self-attention。cross-attention 和 text adapter
保留 PyTorch SDPA，因为它们带文本 padding mask；batch size 1 时做
varlen unpad/repad 的收益通常较小。若实测 `fa2_speedup <= 1`，将配置中的
`self_attention_backend` 改回 `sdpa`，不要仅凭名称假定外部包更快。

## Drive 和训练

先挂载 Drive。配置采用本地原子保存、后台镜像的方式：

```text
本地临时/最新 checkpoint：
  /content/checkpoints_l4_fa2

持久 checkpoint mirror：
  /content/drive/MyDrive/cosmos
```

不直接把 `output_dir` 指向 Drive，是为了避免训练线程同步写入数 GB
checkpoint。`--resume auto` 会同时检查本地和 Drive mirror。

预检：

```bash
uv run --extra train --extra fa2 python scripts/colab_preflight.py \
  --config configs/colab_l4_fa2_24gb.yaml
```

训练：

```bash
uv run --extra train --extra fa2 python scripts/train.py \
  --config configs/colab_l4_fa2_24gb.yaml \
  --resume auto
```

默认是 768 档。若源图主要来自 640px AnimeTimm WebDataset，建议先把
`resolution_stage` 改成 `"512"` 完成基础训练，再使用真实高分辨率来源
切回 768；长期把 640px 图片放大到 768 不会产生新的细节。
