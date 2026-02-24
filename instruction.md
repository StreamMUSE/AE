# StreamMUSE 实验运行指南

## 概述

本指南介绍如何使用 StreamMUSE (`StreamMUSE/`) 代码库进行实时伴奏生成实验，以及如何使用评估工具包 (`eval/`) 对实验结果进行量化评估。

**核心工作流**：
1. **准备阶段**：获取代码、下载模型、准备数据
2. **生成阶段**：启动服务器 + 客户端生成伴奏
3. **评估阶段**：计算音乐质量指标 + 系统指标
4. **汇总阶段**：整理结果用于分析

## 准备工作

### 1. 获取代码库

StreamMUSE: https://github.com/StreamMUSE/StreamMUSE

eval: https://github.com/StreamMUSE/eval

```bash
# StreamMUSE 主代码库
git clone <streammuse-repo-url>

# 评估工具包
git clone <eval-repo-url>
```

期望文件夹结构：
```
AE/
├── StreamMUSE/           # StreamMUSE repo
│   
└── eval/                 # eval repo
│   
└── instruction.md

```

### 2. 下载模型参数

从 [Hugging Face](https://huggingface.co/Jianshu001/music) 下载预训练模型：

**使用 huggingface-cli（推荐）**
```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载到 StreamMUSE/ckpt/ 或者任意你喜欢的路径
mkdir -p StreamMUSE/ckpt

# 下载模型文件
huggingface-cli download Jianshu001/music cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt --local-dir StreamMUSE/ckpt --local-dir-use-symlinks False

```

### 3. 准备格式化数据集

数据集结构要求：
```
input/
├── mel/           # 旋律 MIDI 文件
│   ├── 001.mid
│   ├── 002.mid
│   └── ...
└── acc/           # 伴奏 MIDI 文件（用于评估）
    ├── 001.mid
    ├── 002.mid
    └── ...
```
**使用 StreamMUSE 自带测试数据**：为了方便测试，StreamMUSE 自带两个小的测试文件。可以用于快速跑通流程。


**下载测试数据**：从 [Hugging Face](https://huggingface.co/datasets/S-tanley/formatted_dataset/tree/main/test64_top1) 下载数据集（格式已经处理好），同一个 repo 里还有很多其他处理好的数据集，但是本片论文用的测试数据是 test64_top1。

### 4. 安装依赖

StreamMUSE 使用 `uv` 管理 Python 环境，根据 [uv 官网](https://docs.astral.sh/uv/) 的指引，下载 uv。

## 快速开始（测试流程）

### 启动服务器

```bash
cd StreamMUSE

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1  # 指定 GPU（多 GPU 环境需要）
export CHECKPOINT_PATH=<你的模型路径>
export MODEL_MAX_SEQ_LEN_FRAMES=384  # 模型窗口长度（帧）

# 启动 FastAPI 服务器
PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988
```

**参数说明**：
- `CUDA_VISIBLE_DEVICES`：仅在多 GPU 环境需要指定
- `CHECKPOINT_PATH`：模型检查点路径（注意转义空格）
- `MODEL_MAX_SEQ_LEN_FRAMES`：context window length

### 运行客户端生成伴奏

```bash
cd StreamMUSE

uv run real_time_experiment_runner.py \
    --dataset-dir input/mel \
    --injection-length 128 \
    --generation-length 576 \
    --out-root <你希望保存的文件名>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/batch_run \
    --server-url http://localhost:8988/generate_accompaniment \
    --generation-interval-ticks 2 \
    --generation-length-per-request 5
```
> **注意**：为了确保文件生成到对应的地方，在这里你需要更改 StreamMUSE/app/client.py 中的一个参数：
> 讲这个文件里第 1053 行代码，base_log_dir 最开始的文件名，和外面 <你希望保存的文件名> 对应上。
> ![对应代码](img/code_example.png)
> 如图，将 `experiments-AE5` 改成 <你希望保存的文件名>。


> **注意**：输出目录的命名需要和你选择的参数匹配，格式为：
> `interval_<生成间隔>_gen_frame_<每次请求长度>/prompt_<注入长度>_gen_<总生成长度>/`
>
> 例如，如果你使用 `--generation-interval-ticks 2 --generation-length-per-request 5 --injection-length 128 --generation-length 576`，
> 则路径需要为为 `interval_2_gen_frame_5/prompt_128_gen_576/`

**预期输出结构**：
```
<你希望保存的文件名>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/
├── batch_run/                    # 日志文件目录
│   ├── 001/                      # 每首歌曲一个目录
│   │   ├── inferences.json       # 推理请求/响应的完整 JSON 日志
│   │   └── tick_history.json     # 每个 tick 的 hit/miss/backup 记录
│   ├── 002/
│   └── ...
├── generated/                    # 生成的 MIDI 文件目录
│   ├── 001.mid                   # 生成的伴奏 MIDI 文件
│   ├── 002.mid
│   └── ...
└── gt_generation/                # Ground truth 目录（用于评估对比）
    ├── 001.mid                   # 真实的伴奏 MIDI 文件
    ├── 002.mid
    └── ...
```

## 复现完整实验

### 生成阶段

#### 方法一：手动运行（单组参数）

```bash
# 终端1：启动服务器
cd AE
export CHECKPOINT_PATH=~/ugrip/models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch\=00.val_loss\=0.90296.ckpt
export MODEL_MAX_SEQ_LEN_FRAMES=384
PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988

# 终端2：运行客户端
cd AE
uv run real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_1_gen_frame_3/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 1 --generation-length-per-request 3
```

#### 方法二：批量运行（多组参数）

使用预配置的 `test-run.sh`：
```bash
cd StreamMUSE
chmod +x test-run.sh
./test-run.sh
```

`test-run.sh` 内容示例（可根据需要修改）：
```bash
#!/usr/bin/env bash
# 运行多组参数组合
python3 real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_1_gen_frame_3/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 1 --generation-length-per-request 3
python3 real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 2 --generation-length-per-request 5
# ... 更多组合
```

### 评估阶段

#### 1. 计算音乐质量指标

**单次评估（详细输出）**：
```bash
cd eval

uv run evaluate_accompaniment_metrics.py \
    --generated-dir /home/ubuntu/ugrip/stanleyz/AE/experiments-AE2/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_384/batch_run \
    --groundtruth-dir /home/ubuntu/ugrip/stanleyz/AE/input/acc \
    --output-json results/interval2_gen5_metrics.json \
    --melody-track-names Guitar \
    --auto-phrase-analysis \
```

**批量评估（推荐）**：
```bash
cd eval
./batch_evaluate_stanley.sh
```

编辑 `batch_evaluate_stanley.sh` 配置（根据实验需求调整）：
```bash
# 要遍历的 interval 值列表
INTERVALS=(1 2 4 7)

# 要遍历的 generation frame 值列表
GEN_FRAMES=(3 5 9 15)

# 根目录（根据实际位置调整），主要是把 /home/ubuntu/ugrip/stanleyz/AE/experiments-AE4 换成对应的path
REALTIME_ROOT="/home/ubuntu/ugrip/stanleyz/AE/experiments-AE4/realtime/baseline"
```

#### 2. 计算系统指标（延迟、命中率等）

```bash
cd eval

uv run compute_final_system_metric.py \
    ../StreamMUSE/<实验结果文件夹>/realtime/ \
    -o results-<实验结果文件夹>/final-sys-results
```

**输出文件结构**：
```
results-experiment2/final-sys-results/
├── interval_1_gen_frame_3.json
├── interval_2_gen_frame_5.json
└── ...
```

每个 JSON 文件包含：
- `global_hit_rate`：全局命中率
- `global_avg_backup`：平均备份时间
- `ISR_w`：加权中断服务率
- 其他系统性能指标

#### 3. 结果汇总与分析

**汇总音乐质量指标**：
```bash
cd eval

# 创建汇总表格
uv run summarize_metrics.py results-experiments2/
```

**添加 NLL 并生成最终表格**：
```bash
cd eval

# 将 NLL（负对数似然）添加到汇总表中
uv run add_nll_to_summary.py results-experiments2/ -o final_experiment_results.csv
```

**最终输出**：
- `final_experiment_results.csv`：包含所有实验配置的音乐质量指标
- `results-experiments2/final-sys-results/`：系统指标 JSON 文件

## 参数详细说明

### 服务器参数（app.server）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | 指定 GPU 设备 | `0`, `1`, `0,1` |
| `CHECKPOINT_PATH` | 模型检查点路径 | `~/ugrip/models/ModelBaseline/model.ckpt` |
| `MODEL_MAX_SEQ_LEN_FRAMES` | 最大序列长度 | `384`, `576` |
| 端口 | 服务器监听端口 | `8988` |

### 客户端参数（real_time_experiment_runner.py）

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--dataset-dir` | 输入旋律目录 | `input/mel` |
| `--injection-length` | 提示长度（帧） | `128`, `256` |
| `--generation-length` | 总生成长度（帧） | `384`, `576` |
| `--out-root` | 输出根目录 | `experiments-AE2/realtime/baseline/...` |
| `--server-url` | 服务器端点 | `http://localhost:8988/generate_accompaniment` |
| `--generation-interval-ticks` | 生成间隔 | `1`, `2`, `4`, `7` |
| `--generation-length-per-request` | 每次请求长度 | `3`, `5`, `9`, `15` |

### 评估参数（evaluate_accompaniment_metrics.py）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--generated-dir` | 生成文件目录 | （必需） |
| `--groundtruth-dir` | 真实伴奏目录 | （必需） |
| `--melody-track-names` | 旋律轨道名 | `Guitar` |
| `--auto-phrase-analysis` | 启用乐句分析 | 关闭 |
| `--frechet-music-distance` | 启用 FMD | 关闭 |
| `--polydis-root` | PolyDis 路径 | （无） |
| `--output-json` | 输出 JSON 文件 | （可选） |

## 目录结构参考

### AE/（StreamMUSE 主代码）

```
AE/
├── app/                    # 服务器和客户端
│   ├── server.py          # FastAPI 服务器
│   └── client.py          # 实时客户端
├── input/                 # 输入数据
│   ├── mel/              # 旋律 MIDI
│   └── acc/              # 伴奏 MIDI（用于评估）
├── experiments-AE[0-9]*/  # 实验结果（按实验编号）
│   └── realtime/baseline/
│       └── interval_[I]_gen_frame_[G]/prompt_[P]_gen_[T]/
│           ├── batch_run/         # 日志文件目录
│           │   ├── 001/           # 每首歌曲
│           │   │   ├── inferences.json   # 推理请求/响应日志
│           │   │   └── tick_history.json # 系统指标记录
│           │   └── ...
│           ├── generated/         # 生成的 MIDI 文件
│           │   ├── 001.mid
│           │   └── ...
│           └── gt_generation/     # Ground truth 伴奏（用于评估）
│               ├── 001.mid
│               └── ...
├── extract/               # 数据提取脚本
├── preprocess/           # 数据预处理
├── ckpt/                 # 模型检查点
├── test-run.sh           # 批量生成脚本
└── README.md            # 项目文档
```

### eval/（评估工具包）

```
eval/
├── evaluate_accompaniment_metrics.py     # 核心评估器
├── batch_evaluate_stanley.sh            # 批量评估脚本
├── batch_evaluate_accompaniment_metrics.sh  # 通用批量脚本
├── compute_final_system_metric.py       # 系统指标计算
├── compute_dmr.py                       # DMR 计算
├── summarize_metrics.py                 # 结果汇总
├── add_nll_to_summary.py               # 添加 NLL
├── batch_run_evaluate.py               # Python 批量评估
├── batch_runs.conf                     # 批量运行配置示例
├── run_prompt_polydis_eval.sh          # PolyDis 快速评估
└── results-experiments[0-9]*/          # 评估结果
    ├── interval1_gen3_metrics.json     # 单组结果
    ├── interval2_gen5_metrics.json
    ├── final-sys-results/              # 系统指标
    │   ├── interval_1_gen_frame_3.json
    │   └── ...
    └── final_results.csv               # 最终汇总表
```

## 常见问题与解决方案

### 1. 服务器启动失败

**问题**：`CHECKPOINT_PATH not found` 或 CUDA 错误

```bash
# 解决方案：
# 1. 检查路径是否正确（注意转义空格）
ls -la ~/ugrip/models/ModelBaseline/

# 2. 简化路径（避免特殊字符）
cp "~/ugrip/models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt" ~/model.ckpt
export CHECKPOINT_PATH=~/model.ckpt

# 3. 检查 GPU 可用性
nvidia-smi
```

### 2. 客户端连接失败

**问题**：`Connection refused` 或超时

```bash
# 解决方案：
# 1. 确认服务器正在运行
netstat -tulpn | grep 8988

# 2. 检查服务器日志是否有错误
# 3. 验证 URL 格式
curl http://localhost:8988/docs  # 应返回 FastAPI 文档
```

### 3. 评估时缺少 PolyDis

**问题**：`PolyDis root not found` 或导入错误

```bash
# 解决方案：
# 1. 克隆 PolyDis 仓库
git clone https://github.com/ZZWaang/icm-deep-music-generation.git ~/poly_dis

# 2. 更新评估命令
--polydis-root ~/poly_dis

# 3. 或跳过 PolyDis 指标
# 移除 --polydis-root 参数
```

### 4. 路径不匹配错误

**问题**：`No matching files between generated and groundtruth`

```bash
# 解决方案：
# 1. 检查文件名是否对应
ls generated_dir/*.mid | head -5
ls groundtruth_dir/*.mid | head -5

# 2. 确保使用相同的基础名
# 生成文件: 001_generated.mid
# 真实文件: 001.mid 或 001_acc.mid

# 3. 使用 --keep-melody 如果生成文件只含伴奏
```

### 5. UV 环境问题

**问题**：`uv: command not found` 或包缺失

```bash
# 解决方案：
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 或使用系统 Python
python -m pip install -r requirements.txt
# 然后替换 "uv run" 为 "python"
```

## 实验设计建议

### 参数组合矩阵

建议的实验配置：
```bash
# injection-length: 128, 256
# generation-length: 384, 576, 768
# interval-ticks: 1, 2, 4, 7
# length-per-request: 3, 5, 9, 15
```

### 评估指标解读

- **JSD (Jensen-Shannon Divergence)**: 值越小表示分布越接近
  - `pitch_jsd`: 音高分布相似度
  - `onset_jsd`: 起始时间分布相似度
  - `duration_jsd`: 时值分布相似度
- **FMD (Frèchet Music Distance)**: 整体音乐特征距离
- **PolyDis 指标**: 纹理和和弦相似度
- **系统指标**: 实时性能（命中率、延迟等）

## 扩展与定制

### 添加新的评估指标

1. 在 `evaluate_accompaniment_metrics.py` 中添加函数
2. 在 `@dataclass` 中定义数据结构
3. 更新 `summarize_metrics.py` 的提取逻辑

### 修改生成策略

1. 编辑 `app/client.py` 中的生成逻辑
2. 调整 `real_time_experiment_runner.py` 的参数处理
3. 创建新的实验配置脚本

### 支持新数据集

1. 在 `extract/` 中添加提取脚本
2. 确保输出符合 `mel/` 和 `acc/` 结构
3. 更新数据预处理步骤（如果需要）

---

## 联系与支持

- **代码问题**: 查看各目录下的 README.md
- **实验复现**: 确保使用相同版本依赖
- **结果差异**: 检查随机种子、GPU 型号等

**关键路径示例**：
- 模型检查点：`~/ugrip/models/ModelBaseline/`
- 输入数据：`AE/input/`
- 实验结果：`AE/experiments-AE[编号]/`
- 评估结果：`eval/results-experiments[编号]/`

遵循本指南可完整复现 StreamMUSE 实时伴奏生成实验，并获得可比较的定量评估结果。