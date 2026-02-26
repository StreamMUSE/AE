# StreamMUSE Artifact Evaluation Guide

- [Overview](#overview)
- [Artifact #1: StreamMUSE](#artifact-1-streammuse)
- [Artifact #2: Eval Toolkit](#artifact-2-eval-toolkit)
- [Reproducing our Experiments](#reproducing-our-experiments)
  - [Reproducing Table 3 (Main Results)](#reproducing-table-3-main-results)
    - [Quick Start (Test Workflow)](#quick-start-test-workflow)
      - [Start the Server](#start-the-server)
      - [Run Client to Generate Accompaniment](#run-client-to-generate-accompaniment)
      - [Expected Output Structure](#expected-output-structure)
    - [Reproducing Full Experiments](#reproducing-full-experiments)
      - [Generation Phase](#generation-phase)
        - [Method 1: Manual Run (Single Parameter Set)](#method-1-manual-run-single-parameter-set)
        - [Method 2: Batch Run (Multiple Parameter Sets)](#method-2-batch-run-multiple-parameter-sets)
      - [Evaluation Phase](#evaluation-phase)
        - [1. Compute Musical Quality Metrics](#1-compute-musical-quality-metrics)
        - [2. Compute System Metrics (Latency, Hit Rate, etc.)](#2-compute-system-metrics-latency-hit-rate-etc)
        - [3. Compute NLL (Model Likelihood)](#3-compute-nll-model-likelihood)
        - [4. Results Aggregation and Analysis](#4-results-aggregation-and-analysis)
      - [Detailed Parameter Reference](#detailed-parameter-reference)
  - [Reproducing Figure 3](#reproducing-fig-3)
  - [Reproducing Figure 4](#reproducing-fig-4)

---

## Overview

This guide provides instructions for reproducing the experiments in the StreamMUSE paper. StreamMUSE is a real-time music accompaniment generation system that generates piano accompaniment given a live melody input. **For more details, please refer to our paper:**

_B. Zheng, A. H. Yang, et al. Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation. ([PDF](paper/StreamMUSE.pdf))_

Find the latest version of guidance [here](https://github.com/StreamMUSE/AE).

**Workflow Summary:**

1. **Setup**: Clone repositories, download model checkpoint, prepare dataset
2. **Generation**: Start server + run client to generate accompaniments
3. **Evaluation**: Compute musical quality metrics and system performance metrics
4. **Aggregation**: Collect results for analysis

**Hardware Requirements:**

- GPU with CUDA support (tested on NVIDIA A100, RTX A4000)
- 16GB+ GPU memory recommended
- Linux/macOS environment

**Working Directory Convention:**
Unless otherwise specified, all commands should be executed from the root directory (`AE/`). When a command block starts with `cd StreamMUSE` or `cd eval`, execute that command from the root directory, and subsequent commands in that block should be run from the respective subdirectory.

---

## Artifact #1: StreamMUSE

**Source:** https://github.com/StreamMUSE/StreamMUSE

**Description:** Main codebase for real-time accompaniment generation, including:

- FastAPI server for model inference (`app/server.py`)
- Client for real-time interaction (`app/client.py`)
- Batch experiment runner (`real_time_experiment_runner.py`)
- Data preprocessing and extraction tools

**Expected Folder Structure:**

```
AE/
├── StreamMUSE/           # StreamMUSE repository
│   ├── app/              # Server and client code
│   ├── ckpt/             # Model checkpoint directory
│   ├── input/            # Input data (mel/acc)
│   └── ...
└── eval/                 # Eval toolkit (Artifact #2)
```

**Quick Start:**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/StreamMUSE/StreamMUSE.git
   ```

2. **Download model checkpoint:**

   ```bash
   pip install huggingface-hub
   mkdir -p StreamMUSE/ckpt
   hf download Jianshu001/music cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch=00.val_loss=0.90296.ckpt --local-dir StreamMUSE/ckpt
   ```

   In this example code, the target path is `StreamMUSE/ckpt`, you can replace this with any path you like.

3. **Prepare dataset:**

   ```bash
   # Dataset structure should be:
   <dataset-name>/
   ├── mel/           # Melody MIDI files
   │   ├── 001.mid
   │   └── ...
   └── acc/           # Accompaniment MIDI files (for evaluation)
       ├── 001.mid
       └── ...
   ```

   **Using test dataset inside StreamMUSE**: There is one small dataset inside StreamMUSE (`StreamMUSE/input`), which can be directly used for running. I recommend you use this small dataset first to test if everything works well.

   **Download dataset**: Download from [Hugging Face Datasets](https://huggingface.co/datasets/S-tanley/formatted_dataset/tree/main/test64_top1). This dataset (test64_top1) is the test dataset we are using for the experiments in the paper. **Important**: If you are using this dataset, it will run about 1 h for one combination.

4. **Install dependencies:**
   This project uses uv to manage the environment, according to the [uv official website](https://docs.astral.sh/uv/), download uv.

---

## Artifact #2: Eval Toolkit

**Source:** https://github.com/StreamMUSE/eval

**Description:** Evaluation toolkit for computing musical quality and system performance metrics:

- Musical metrics: JSD (pitch/duration/onset), FMD, PolyDis metrics
- System metrics: hit rate, backup level, latency
- **NLL aggregation**: collects raw NLL JSONs computed by StreamMUSE and merges into the final summary table
- Batch evaluation scripts

**Installation:**

```bash
git clone https://github.com/StreamMUSE/eval.git
```

---

## Reproducing our Experiments

All experiments follow the same pattern:

1. Start the StreamMUSE server
2. Run the specific client with specific parameters to evaluate the tradeoff between real-time responsiveness and music quality:
   - **I (Inference Interval)**: `--generation-interval-ticks` controls how often the model generates new content (e.g., every 1, 2, 4, or 7 ticks). Smaller I means more frequent generation, better responsiveness but higher computational cost.
   - **GL (Generation Length)**: `--generation-length-per-request` controls how many frames the model generates per request (e.g., 3, 5, 9, or 15 frames). Smaller GL reduces latency but may degrade music coherence.
3. Evaluate results using the eval toolkit or other evaluation codes.

---

### Reproducing Table 3 (Main Results)

This section reproduces the main experimental results table with different generation intervals and frame sizes. **Note that there are three different setting of this experiment. If you are in the local-server/remote setting, remember to do the port forwarding step.**

For the **real-time latency figures** in the paper (Fig. 3 and Fig. 4), we use a dedicated benchmark + analysis pipeline described in the sections [Reproducing Fig. 3](#reproducing-fig-3) and [Reproducing Fig. 4](#reproducing-fig-4) below.

#### Quick Start (Test Workflow)

##### Start the Server

```bash
cd StreamMUSE

# Set environment variables
export CUDA_VISIBLE_DEVICES=<GPU_ID>  # Specify GPU (only needed in multi-GPU environments)
export CHECKPOINT_PATH=<your-model-path>
export MODEL_MAX_SEQ_LEN_FRAMES=384  # Model window length (frames)

# Start FastAPI server
PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988
```

**Parameter Description:**

- `CUDA_VISIBLE_DEVICES`: Only needed when you have multiple GPUs
- `CHECKPOINT_PATH`: Model checkpoint path (be careful with escaping spaces)
- `MODEL_MAX_SEQ_LEN_FRAMES`: Context window length

##### Port Forwarding (for local-server/remote server deployment)

If you deploy the server on a remote machine (e.g., a GPU server) and run the client locally, use SSH port forwarding to tunnel the server's port to your local machine.

**Setup port forwarding:**

```bash
# On your local machine (client side)
ssh -L 8988:localhost:8988 <remote-server-ip>
```

This forwards `localhost:8988` on your local machine to `localhost:8988` on the remote server.

**Then in the client command, use localhost:**

```bash
--server-url http://localhost:8988/generate_accompaniment
```

> **Note:** Skip this section if both server and client run on the same machine (local setting).

##### Run Client to Generate Accompaniment

```bash
cd StreamMUSE

uv run real_time_experiment_runner.py \
    --dataset-dir input/mel \
    --injection-length 128 \
    --generation-length 576 \
    --out-root <your-desired-output-name>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/batch_run \
    --server-url http://localhost:8988/generate_accompaniment \
    --generation-interval-ticks 2 \
    --generation-length-per-request 5
```

> **Note**: To ensure files are generated in the correct location, you need to modify a parameter in `StreamMUSE/app/client.py`:
> At line 1053, change the beginning filename in `base_log_dir` to match `<your-desired-output-name>` outside.
> ![Corresponding Code](img/code_example.png)
> As shown in the figure, change `experiments-AE5` to `<your-desired-output-name>`.

> **Note**: The output directory naming needs to match your chosen parameters, with the format:
> `interval_<generation-interval>_gen_frame_<frames-per-request>/prompt_<injection-length>_gen_<total-generation-length>/`
>
> For example, if you use `--generation-interval-ticks 2 --generation-length-per-request 5 --injection-length 128 --generation-length 576`,
> then the path should be `interval_2_gen_frame_5/prompt_128_gen_576/`

##### Expected Output Structure

```
<your-desired-output-name>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/
├── batch_run/                    # Log files directory
│   ├── 001/                      # One directory per song
│   │   ├── inferences.json       # Complete JSON logs of inference requests/responses
│   │   └── tick_history.json     # Hit/miss/backup records per tick
│   ├── 002/
│   └── ...
├── generated/                    # Generated MIDI files directory
│   ├── 001.mid                   # Generated accompaniment MIDI file
│   ├── 002.mid
│   └── ...
└── gt_generation/                # Ground truth directory (for evaluation comparison)
    ├── 001.mid                   # Real accompaniment MIDI file
    ├── 002.mid
    └── ...
```

---

#### Reproducing Full Experiments

##### Generation Phase

###### Method 1: Manual Run (Single Parameter Set)

```bash
# Terminal 1: Start server
cd StreamMUSE
export CHECKPOINT_PATH=~/ugrip/models/ModelBaseline/cp_transformer_909+ac+1k7_trackemb_interleavepos_v0.2_large_batch_40_schedule.epoch\=00.val_loss\=0.90296.ckpt
export MODEL_MAX_SEQ_LEN_FRAMES=384
PYTHONPATH="$(pwd)" uv run -- uvicorn app.server:app --host 0.0.0.0 --port 8988

# Terminal 2: Run client
cd StreamMUSE
uv run real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_1_gen_frame_3/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 1 --generation-length-per-request 3
```

###### Method 2: Batch Run (Multiple Parameter Sets)

Use the pre-configured `test-run.sh`:

```bash
cd StreamMUSE
# start server like previous do first, same code as before, I just omit here

chmod +x test-run.sh
./test-run.sh
```

Example content of `test-run.sh` (can be modified as needed):

```bash
#!/usr/bin/env bash
# Run multiple parameter combinations
python3 real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_1_gen_frame_3/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 1 --generation-length-per-request 3
python3 real_time_experiment_runner.py --dataset-dir input/mel --injection-length 128 --generation-length 576 --out-root experiments-AE5/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/batch_run --server-url http://localhost:8988/generate_accompaniment --generation-interval-ticks 2 --generation-length-per-request 5
# ... more combinations
```

##### Evaluation Phase

###### 1. Compute Musical Quality Metrics

**Single Evaluation (Detailed Output):**

```bash
cd eval

uv run evaluate_accompaniment_metrics.py \
    --generated-dir <your-experiment-folder-path>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_384/generated \
    --groundtruth-dir <your-experiment-folder-path>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_384/gt-generation \
    --output-json results/interval2_gen5_metrics.json \
    --melody-track-names Guitar \
    --auto-phrase-analysis
```

`generated-dir`, `groundtruth-dir` and `output-json` need to be alter to your corresponding path.

**Batch Evaluation (Recommended):**

```bash
cd eval
./batch_evaluate_stanley.sh
```

Edit `batch_evaluate_stanley.sh` configuration (adjust according to experimental needs):

```bash
# List of interval values to iterate
INTERVALS=(1 2 4 7)

# List of generation frame values to iterate
GEN_FRAMES=(3 5 9 15)

# Root directory (adjust according to actual location)
REALTIME_ROOT="<your-experiment-folder-path>/realtime/baseline" # for example, "/home/ubuntu/ugrip/stanleyz/AE/experiments-AE4/realtime/baseline"

# The expexcted out put folder
OUT_ROOT="<expected-output-folder>"
```

###### 2. Compute System Metrics (Latency, Hit Rate, etc.)

```bash
cd eval

uv run compute_final_system_metric.py \
    ../StreamMUSE/<your-experiment-folder-path>/realtime/ \
    -o results-<experiment-results-folder>/final-sys-results
```

**Output File Structure:**

```
results-<experiment-results-folder>/final-sys-results/
├── interval_1_gen_frame_3.json
├── interval_2_gen_frame_5.json
└── ...
```

Each JSON file contains:

- `global_hit_rate`: Global hit rate
- `global_avg_backup`: Average backup time
- `ISR_w`: Weighted interrupt service rate
- Other system performance metrics

###### 3. Compute NLL (Model Likelihood)

> **Run from `StreamMUSE/`, not `eval/`.** NLL computation requires the model checkpoint.

**Batch run (recommended) — mirrors the same combinations as `test-run.sh`:**

```bash
cd StreamMUSE

# Edit CONFIG section inside run_nll.sh first:
#   EXP_ROOT       = your experiment folder name (e.g. experiments-AE5)
#   CKPT_PATH      = path to model checkpoint
#   EVAL_NLL_DIR   = path to eval/results-<experiment>/nll_runs
chmod +x nll_compute/run_nll.sh
bash nll_compute/run_nll.sh
```

**Single combination (manual):**

```bash
cd StreamMUSE

uv run python -m nll_compute.runners.run_cal_nll \
    --midi_dir <your-experiment>/realtime/baseline/interval_2_gen_frame_5/prompt_128_gen_576/generated \
    --ckpt_path <your-ckpt-path> \
    --save_json_path ../eval/results-<experiment>/nll_runs/experiments_interval2_gen5.json \
    --window 256 --offset 64
```

> **Path convention:** NLL output files must be saved to `results-<experiment>/nll_runs/experiments_*.json` (filename must start with `experiments` and contain `interval` + `gen_frame` numbers). This is how `add_nll_to_summary.py` auto-discovers and matches them to the CSV rows.

**Expected output location:**

```
eval/results-<experiment>/
└── nll_runs/
    ├── experiments_interval1_gen3.json
    ├── experiments_interval2_gen5.json
    └── ...
```

###### 4. Results Aggregation and Analysis

**Aggregate Musical Quality Metrics:**

```bash
cd eval

# Create summary table (musical quality + system metrics)
uv run summarize_metrics.py results-<experiment-results-folder>/
```

**Add NLL and Generate Final Table:**

```bash
cd eval

# Merge NLL columns into the summary CSV → produces final table
uv run add_nll_to_summary.py results-<experiment-results-folder>/ -o final_experiment_results.csv
```

**Final Output:**

- `final_experiment_results.csv`: All metrics for all configurations (musical quality + system + NLL)
- `results-<experiment>/final-sys-results/`: System metrics JSON files
- `results-<experiment>/nll_runs/`: Raw NLL JSON files

##### Detailed Parameter Reference

**Server Parameters (app.server)**

| Parameter                  | Description             | Example                                   |
| -------------------------- | ----------------------- | ----------------------------------------- |
| `CUDA_VISIBLE_DEVICES`     | Specify GPU device      | `0`, `1`, `0,1`                           |
| `CHECKPOINT_PATH`          | Model checkpoint path   | `~/ugrip/models/ModelBaseline/model.ckpt` |
| `MODEL_MAX_SEQ_LEN_FRAMES` | Maximum sequence length | `384`, `576`                              |
| Port                       | Server listening port   | `8988`                                    |

**Client Parameters (real_time_experiment_runner.py)**

| Parameter                         | Description                      | Example                                        |
| --------------------------------- | -------------------------------- | ---------------------------------------------- |
| `--dataset-dir`                   | Input melody directory           | `input/mel`                                    |
| `--injection-length`              | Prompt length (frames)           | `128`, `256`                                   |
| `--generation-length`             | Total generation length (frames) | `384`, `576`                                   |
| `--out-root`                      | Output root directory            | `experiments-AE2/realtime/baseline/...`        |
| `--server-url`                    | Server endpoint                  | `http://localhost:8988/generate_accompaniment` |
| `--generation-interval-ticks`     | Generation interval              | `1`, `2`, `4`, `7`                             |
| `--generation-length-per-request` | Length per request               | `3`, `5`, `9`, `15`                            |

**Evaluation Parameters (evaluate_accompaniment_metrics.py)**

| Parameter                  | Description               | Default    |
| -------------------------- | ------------------------- | ---------- |
| `--generated-dir`          | Generated files directory | (required) |
| `--groundtruth-dir`        | Ground truth directory    | (required) |
| `--melody-track-names`     | Melody track name         | `Guitar`   |
| `--auto-phrase-analysis`   | Enable phrase analysis    | disabled   |
| `--frechet-music-distance` | Enable FMD                | disabled   |
| `--polydis-root`           | PolyDis path              | (none)     |
| `--output-json`            | Output JSON file          | (optional) |

---

### Reproducing Fig. 3

This figure corresponds to the **experiment fitting analysis** (generation length vs latency with fitted curves).

**Additional parameter context (for this figure):**

- **Generation length (frames)**: The `generation_length_frames` values swept in the benchmark configs (1, 3, 5, …, 15). These appear on the x-axis of Fig. 3.
- **Round-trip time (ms)**: End-to-end latency for each request, from client send to response received. The quadratic formulas in the config approximate the mean round-trip time at each generation length (the y-axis of Fig. 3).

**Real-time deployment settings (used in Fig. 3 & Fig. 4):**

| Setting        | Description                                                                                           | Example experiment folder                       |
| ------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `local`       | Client and server on the **same machine** (no network hop; measures pure model + runtime).           | `StreamMUSE/experiments/local_gl_test/`         |
| `local_server`| Client and server on the **same local network** (e.g., PC client ↔ Mac server over LAN).             | `StreamMUSE/experiments/local_server_gl_test/`  |
| `remote`      | Client on local machine, server on a **remote cloud GPU machine** (e.g., Hyperstack ↔ home PC).      | `StreamMUSE/experiments/remote_gl_test_new/`    |

In practice, you may run all three settings on a single physical machine (for convenience). What matters for the analysis is that you produce **three separate experiment folders**, one per setting as above; the exact host/port you use is flexible as long as the client can reach the server.

**Prerequisites:**

- You will run three real-time benchmark experiments (one per deployment setting) using the provided YAML configs:
  - `StreamMUSE/experiments/local_gl_test_config.yaml`        → writes to `StreamMUSE/experiments/local_gl_test/`
  - `StreamMUSE/experiments/local_server_gl_test_config.yaml` → writes to `StreamMUSE/experiments/local_server_gl_test/`
  - `StreamMUSE/experiments/remote_gl_test_new_config.yaml`   → writes to `StreamMUSE/experiments/remote_gl_test_new/`

**Run the three benchmark sweeps (from `AE/` root):**

```bash
cd StreamMUSE

# 1) Pure local (PC-PC)
#    (client and server on the same machine; you can use the same URL as other settings if desired)
uv run app/benchmarking/bulk_benchmark.py experiments/local_gl_test_config.yaml

# 2) Local network (PC-Mac)
#    (client and server on the same LAN; update server.url in the YAML if your server runs on another host)
uv run app/benchmarking/bulk_benchmark.py experiments/local_server_gl_test_config.yaml

# 3) Remote cloud (Hyperstack-PC)
#    (client on your local machine, server on a remote/cloud machine; set server.url accordingly)
uv run app/benchmarking/bulk_benchmark.py experiments/remote_gl_test_new_config.yaml
```

> **Note:** All three configs assume that the `server.url` field points to a running StreamMUSE server. For a simplified setup, you can run the server once (e.g., `http://0.0.0.0:8988/generate_accompaniment`) and set the same URL in all three YAML files; you will still obtain three distinct experiment folders corresponding to the three settings.

Each command will create an `experiments/<setting>/` directory containing the generation-length sweep CSVs:

- `StreamMUSE/experiments/local_gl_test/*.csv`
- `StreamMUSE/experiments/local_server_gl_test/*.csv`
- `StreamMUSE/experiments/remote_gl_test_new/*.csv`

**Analysis step (from `AE/` root):**

```bash
cd eval

# (Optional) Refit quadratic formulas for your own runs
uv run -m src.benchmarks.fit_latency_formulas configs/analysis_config_fig3_fig4.yaml
# Copy the printed formula blocks (a, b, c) back into configs/analysis_config_fig3_fig4.yaml

# Then run YAML-configured bulk analysis for Fig. 3 & Fig. 4
uv run -m src.benchmarks.yaml_bulk_analysis \
    configs/analysis_config_fig3_fig4.yaml
```

**Key outputs (Fig. 3):**

- `eval/results/benchmarks/fig3_fig4/plots/experiment_fitting_analysis.png`  
  This is the **Fig. 3** plot (box plots overlaid with formula-based fitted lines and ±5% confidence bands).
- `eval/results/benchmarks/fig3_fig4/yaml_analysis_summary.md`  
  Summary of the experiments, formulas, and analysis settings.

---

### Reproducing Fig. 4

This figure corresponds to the **BPM-based parameter constraint analysis**.

**Additional parameter context (for this figure):**

- **BPM and τ (ms per tick)**: For each BPM in `constraint_analysis.bpm_list`, the analysis uses τ = 15000 / BPM (ms per tick) to convert musical time to milliseconds when checking real-time constraints.
- **Percentile**: `constraint_analysis.percentile` (e.g., 99.5) specifies which latency percentile must satisfy the constraints for a configuration to be considered valid.

Running the same command as in Fig. 3 (above) also generates Fig. 4 using the BPM and constraint settings in
`eval/configs/analysis_config_fig3_fig4.yaml`.

**Key outputs (Fig. 4):**

- `eval/results/benchmarks/fig3_fig4/plots/parameter_constraint_analysis_bpm.png`  
  This is the **Fig. 4** plot (constraint heatmaps across generation lengths and inference intervals, for multiple BPMs).
- `eval/results/benchmarks/fig3_fig4/plots/constraint_analysis_*.png`  
  Optional per-experiment constraint visualizations (supporting material, not directly shown in the paper).

You can regenerate both Fig. 3 and Fig. 4 at any time by re-running:

```bash
cd eval
uv run -m src.benchmarks.yaml_bulk_analysis configs/analysis_config_fig3_fig4.yaml
```