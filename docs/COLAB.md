# Colab 滚动流式训练

推荐入口是 `notebooks/colab_rolling_train.ipynb`。它使用 raw WebDataset 滚动模式：训练当前 CPU latent block 时，后台只预取下一个 raw tar；到 optimizer step 边界后暂时把 T5 留在 CPU、把 Wan encoder 换入 GPU 编码下一块，再继续 DiT 训练。

## 前置条件

1. 把本仓库推送到你自己的 Git 仓库，并把 notebook 中的 `PROJECT_REPO` 改成该地址。
2. 在 Hugging Face 接受以下仓库的使用条款：
   - `google/t5gemma-2-270m-270m`
   - `animetimm/danbooru-wdtagger-v4-w640-ws-50k`，以及之后实际使用的数据集。
3. 在 Colab Secrets 中创建 `HF_TOKEN`，并允许 notebook 读取。
4. 选择 GPU runtime。notebook 会在支持 BF16 且显存至少 20 GiB 时选择 L4 配置，否则选择 T4/FP16 配置。

不要执行 Wan2.2 仓库的整份 `requirements.txt`。它固定旧版 `transformers`，与这里的 T5Gemma2 加载路径冲突。本项目只使用 `wan/modules/vae2_2.py`，所需的 `einops` 已列入本项目依赖。

## 首次 smoke run

notebook 默认值是：

```python
DATASET_REPO = "animetimm/danbooru-wdtagger-v4-w640-ws-50k"
SMOKE_RUN = True
SHARD_LIMIT = 1
```

smoke run 使用独立的 checkpoint 目录、256-sample rolling block，并在 8 个 optimizer step 后停止。它不会污染正式 checkpoint。按顺序运行全部 cell；预检必须全部通过后才会加载 DiT。

预检会验证：

- CUDA 与 BF16/FP16 能力；
- `bitsandbytes` 是否可用；
- Wan 源码、VAE checkpoint 和 T5 encoder-only checkpoint；
- shard list 与本地 shard 路径；
- rolling block 是否位于梯度累积边界；
- cache 磁盘余量；
- Google Drive 是否真的挂载。

## 正式训练

smoke run 通过后重新从设置 cell 开始，至少修改：

```python
SMOKE_RUN = False
SHARD_LIMIT = None
DATASET_REPO = "animetimm/danbooru-wdtagger-v4-w640-ws-full"
```

50k、150k 和 full 是不同的 shard 集合。不要在训练中途修改正式任务的 shard list；resume cursor 依赖固定的 shard 集合、顺序和 seed。需要换数据集时，新建正式 checkpoint 目录或明确开始新的训练阶段。

正式配置：

- L4：`configs/colab_l4_rolling.yaml`，27 层、BF16、梯度累积 16。
- T4：`configs/colab_t4_rolling.yaml`，20 层、FP16、梯度累积 32。

两者都禁止在 `bitsandbytes` 失败时静默退回 PyTorch AdamW，因为额外的 FP32 optimizer state 很可能导致单卡 OOM。

## 存储与恢复

- raw tar cache：`/content/raw_cache`，最大 8 GiB，只预取 1 个下一 shard，用完即删。
- 本地 checkpoint：`/content/checkpoints_l4` 或 `/content/checkpoints_t4`，先写隐藏临时目录再原子 rename。
- Drive mirror：`/content/drive/MyDrive/cosmos_anime/...`，后台复制完成后才更新 `latest.txt`。
- `--resume auto` 同时搜索本地与 Drive，并忽略不完整 checkpoint。

Colab 重启后重新运行 notebook。只要 Drive mirror 已完成，训练会从最新 optimizer step 和 source cursor 继续。下载中断留下的 `.part` 文件支持 HTTP Range 续传。

## 手动预检

在 Colab 项目目录中可以单独运行：

```bash
python scripts/colab_preflight.py \
  --config configs/colab_l4_rolling.yaml
```

预检失败时不要绕过错误直接启动训练。尤其是 `8-bit optimizer`、`Google Drive`、`Wan VAE checkpoint` 和 `precision` 错误。
