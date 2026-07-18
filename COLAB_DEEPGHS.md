# Colab L4：DeepGHS Danbooru 2024 流式训练

数据源为 `deepghs/danbooru2024-webp-4Mpixel`，约 805 万张图片，
单图限制为 4MP。图片 tar 与同仓库的 `metadata.parquet` 分开存放，因此
首次使用需要建立一次元数据索引；索引存到 Google Drive 后，后续 Colab
会话无需重建。旧版用其他仓库生成的 17125 分区索引并非严格同源，
bootstrap 会自动识别并一次性替换为 v2 索引。

## 1. 环境和权限

先在 Hugging Face 网页接受 DeepGHS 数据集的 gated/sensitive 访问条款，
然后在 Colab 设置 token：

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

```bash
%cd /content/my_cosmos
!uv sync --extra train --extra fa2
```

## 全自动方式

完成上面的 token 设置和 `uv sync` 后，以下一条命令会自动挂载 Drive、
下载或复用 Wan2.2 源码与 VAE、提取或复用 T5Gemma encoder、建立元数据
索引、生成全部 1000 个图片 tar 清单、执行 preflight，并直接开始训练：

```bash
!uv run python scripts/bootstrap_deepghs_colab.py
```

第一次建议只跑两个图片 tar。bootstrap 会自动把它限制为 8 optimizer
step，并使用 `/content/checkpoints_l4_fa2_smoke`、
`/content/drive/MyDrive/cosmos/smoke` 和独立 W&B run，不会污染正式断点：

```bash
!uv run python scripts/bootstrap_deepghs_colab.py --smoke-shards 2
```

可用 `--smoke-steps N` 改变 smoke 长度。

只准备所有资源但暂不启动训练：

```bash
!uv run python scripts/bootstrap_deepghs_colab.py --prepare-only
```

以下章节是需要拆开执行或排查某一步时使用的手动命令。

## 2. 首次建立标签索引

源 Parquet 和构建临时文件放在 Colab 本地盘，最终索引才复制到 Drive：

```bash
!uv run python scripts/prepare_deepghs_metadata.py \
  --download-dir /content/deepghs_metadata_source \
  --build-dir /content/deepghs_metadata_build \
  --output /content/drive/MyDrive/cosmos/deepghs_metadata
```

脚本只扫描 DeepGHS 自带元数据一次，按 `Danbooru ID % 1000` 生成与
`images/0000.tar` 至 `images/0999.tar` 对应的 1000 个分区。若目标已存在，
脚本会校验 `_index_manifest.json` 的版本、来源和分区数；只有完全匹配才
跳过。旧索引会先在 Colab 本地构建完成，再替换 Drive 上的目录。

## 3. 生成图片 tar 清单

先用少量 shard 做 smoke，确认完整训练链路：

```bash
!uv run python scripts/list_hf_shards.py \
  --repo deepghs/danbooru2024-webp-4Mpixel \
  --split images \
  --output /content/deepghs_raw_shards.txt \
  --limit 2
```

正式训练时删除 `--limit 2`。清单只是 1000 个远程 URL，不会预下载图片。

## 4. 流式训练

```bash
!uv run python scripts/preflight_colab.py \
  --config configs/colab_l4_fa2_deepghs.yaml

!uv run python scripts/train.py \
  --config configs/colab_l4_fa2_deepghs.yaml \
  --resume auto
```

训练过程会异步下载下一份图片 tar；当前 tar 在 GPU 上批量做 Wan VAE
编码，随后卸载 VAE 并训练 DiT。tar 用完即删除，checkpoint 镜像写入
`/content/drive/MyDrive/cosmos`。配置将 `max_upscale` 设为 1.10，避免把
低分辨率图片强行放大到 768 桶。

下载日志每约 2 秒输出一次当前文件名、百分比、已下载/总大小、平均速度
和耗时。断流会显示异常、等待时间和重试次数；已有文件会显示
`cache hit`。Hugging Face 模型和 DeepGHS 同源 metadata 使用独立进度条，
元数据分桶阶段也会显示 DuckDB 扫描进度。

DeepGHS 配置默认每编码 256 张就切换到 DiT 训练，并每约 5 秒显示一次
Wan VAE 编码速度和 ETA；标签读取、tar 索引扫描及 block 完成也分别输出
状态。DiT 按 ratio 组成 microbatch 4，gradient accumulation 为 4，有效
batch 仍为 16。七个 ratio bucket 最多留下 21 张等待同形状伙伴，因此
`text_cache_size=224`，保证首个文本窗口不会为了凑数先触发第二个 VAE
block。

L4 配置还会用一个有界 CPU 线程提前解码 16 张图片，并首先探测 batch 16
进行 Wan 编码；若 OOM 会自动递归回退并记住 8、4、2、1 中的安全值，
后续 batch 不会反复触发相同 OOM。数据日志按 tar 显示
`当前 shard/总 shard`、`已扫描/总图片`、通过数、各类跳过数、图片速度、
ETA 和 CUDA reserved。训练每个 optimizer step 都显示输入等待占比和
CUDA 峰值；如果 `input_wait_ratio` 很高，瓶颈仍在数据/编码而不是 DiT。

DeepGHS L4 配置明确关闭全模型 gradient checkpointing，让 768 训练保留
激活并减少重计算；目标是优先使用 L4 的空闲显存换速度。W&B 默认启用，
项目名为 `cosmos-anime`，记录 loss、LR、gradient norm、吞吐、输入等待
比例、step 时间、CUDA allocated/reserved 峰值以及 shard/sample 游标。
run ID 保存在 Drive 的 `wandb-run-id.txt`，重启会续到同一个 run。把
`WANDB_API_KEY` 加入 Colab Secrets 即可。

Rolling 模式在两个 GPU 重负载阶段之间做真正的互斥换入：Wan 编码前将
DiT 和 optimizer state 卸载到 CPU；256 个 latent 就绪后先把 Wan 卸载，
再恢复 DiT/optimizer。这样 Wan 不会与 0.83B 训练状态争抢 22 GiB 显存，
DiT 阶段仍可关闭 checkpoint。首次 batch 探测 OOM 后会先退出异常作用域、
释放失败激活并清理 cache，再以较小 batch 重试；同时启用 expandable
segments 减少显存碎片。

Bootstrap 会在加载训练模型前真实下载 `images/0000.json` 来验证 gated
图片权限。若这里返回 403，需要用 `HF_TOKEN` 所属的同一个账号打开
DeepGHS 数据集页面并接受访问条款；能下载公开的 Danbooru 元数据不能
证明该账号已经获得 gated 图片权限。

每 250 个 optimizer step 保存一次 checkpoint，即每 4000 个训练样本一次。
checkpoint 先原子写到 `/content/checkpoints_l4_fa2`，再由后台线程镜像到
`/content/drive/MyDrive/cosmos`；两侧只保留最近 2 个完整 checkpoint。
bootstrap 默认带 `--resume auto`，Colab 重连后直接重跑同一条命令即可。

## 5. 从 checkpoint 采样检查效果

采样阶段依次使用 GPU：T5 编码后卸载，DiT 完成 flow ODE 后卸载，最后
加载 Wan2.2 decoder，因此不要求三个模型同时占显存：

```bash
!uv run python scripts/sample.py \
  --config configs/colab_l4_fa2_deepghs.yaml \
  --checkpoint /content/drive/MyDrive/cosmos \
  --prompt "1girl, solo, detailed eyes, best quality" \
  --negative-prompt "low quality, blurry" \
  --width 768 --height 768 \
  --steps 28 --solver heun --guidance-scale 5 \
  --output /content/drive/MyDrive/cosmos/sample.png
```

宽高必须是 32 的倍数。刚开始从零训练时图片会接近噪声，先观察 loss 是否
下降，再固定相同 prompt 和 seed 对比不同 checkpoint。
