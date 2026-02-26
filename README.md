# StreamMUSE Artifact Evaluation

This repository contains the artifact evaluation package for the paper "Real-Time Language Model Jamming: A Case Study for Live Music Accompaniment Generation" by B. Zheng, A. H. Yang, et al.

## What This Artifact Contains

This artifact includes two main components: StreamMUSE and the Eval Toolkit.

**StreamMUSE** is the main codebase for real-time music accompaniment generation. It provides a FastAPI server for model inference, a client for real-time interaction, a batch experiment runner for automated evaluation, and data preprocessing tools. The system generates piano accompaniment given a live melody input.

**Eval Toolkit** provides evaluation scripts for computing both musical quality metrics (such as JSD, FMD, and PolyDis metrics) and system performance metrics (including hit rate, backup level, and latency). It also includes utilities for aggregating results and computing model likelihood (NLL).

## Hardware and Environment Requirements

The experiments require a GPU with CUDA support (tested on NVIDIA A100 and RTX A4000), at least 16GB of GPU memory, and a Linux or macOS environment. The project uses uv for Python environment management.

## How to Use This Artifact

**For detailed reproduction instructions, please see [instruction.md](./instruction.md).**

The instruction document is organized as follows:

The Overview section provides a workflow summary covering setup, generation, evaluation, and aggregation phases, along with hardware requirements and working directory conventions.

The Artifact sections describe the two main components (StreamMUSE and Eval Toolkit), including their purposes, folder structures, and quick start steps for cloning repositories, downloading the model checkpoint from Hugging Face, preparing datasets, and installing dependencies.

The Reproducing Our Experiments section provides comprehensive instructions for replicating the main results (Table 3), including:
- A quick start test workflow for verifying the setup
- Instructions for starting the server and running the client
- Expected output structure
- Full experimental procedures covering both manual and batch run methods
- Evaluation phases for computing musical quality metrics, system metrics, and NLL
- Results aggregation and analysis
- Detailed parameter reference tables

The document also includes placeholders for reproducing Figure 3 and Figure 4 from the paper.

## Quick Links

- Full Instructions: See [instruction.md](./instruction.md) for detailed reproduction steps
- Latest Version: https://github.com/StreamMUSE/AE
- Paper PDF: See [StreamMUSE.pdf](./paper/StreamMUSE.pdf)

## Citation

If you use this code, please cite our paper using the BibTeX entry provided in the instruction document.
