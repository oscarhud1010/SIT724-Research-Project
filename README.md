# LLM-Based UX Evaluation Pipeline
Honours thesis project (SIT724) — a reproducible methodology for comparing multimodal LLM responses against human UX questionnaire data.

## Overview
This project tests whether a multimodal LLM (Qwen2.5-VL 72B) can approximate human responses to UX questionnaires when shown the same visual interfaces.

This repo contains the raw data obtained for each step, the question file used to elicit responses from the model, and the Python script (`llama-cpp-python`) used to initialise, query, and record LLM responses on the Deakin University High Performance Cluster (HPC).

## Tech Stack
- Python, `llama-cpp-python` for local GGUF model inference
- Image-to-base64 conversion to put images into an LLM-readable format
- JSON-based prompt generation and response logging
- Slurm job scheduling for HPC cluster execution

## Repo Structure
Each step folder contains:
- The raw LLM results/responses
- The question set used as input (JSON)
- The Python script used to elicit responses on the Deakin HPC

## Adapting the Pipeline
To apply this methodology to other LLMs, the Python script can be edited to change:
- Image path (the image shown to the LLM to rate)
- Model path (the GGUF model being queried)
- Question file path (the question set asked of the LLM)
- The specific prompt used for that stage

## Notes
This pipeline is part of a broader Honours thesis exploring LLM-based UX evaluation methods.
