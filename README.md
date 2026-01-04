MLLM-as-a-Judge
================

This repository contains the data, prompts, results, and code for our **MLLM-as-a-Judge** project.

## Folder Overview

- **`prompts/`**: Includes all prompts used in our experiments.
  - Files (and any subfolders) labeled **`v1`** are the simplest **all-factor** prompt versions.
  - Files (and any subfolders) labeled **`v2`** are **aligned-by-factor-within-category** versions, where different factors are separated and explicitly aligned within each category.
  - Files (and any subfolders) labeled **`v3`** are the **main prompts** used in our final experiments.

- **`results/`**: Contains all experimental results.
  - **`bad_examples/`**: Counterexamples for each factor, along with scores produced by the main prompt.
  - **`good_examples/`**: Scores from the main prompt on ground-truth images.
  - **`Human_Evaluation/`**: Human evaluators’ ratings and related data.
  - **`Traditional_metrics/`**: Data comparing our **Judge factors** with **traditional metrics** (including the normalized and correlation results).

- **`src/`**: Source code used in our experiments and analysis.

For more details on how to run the code or reproduce the experiments, please refer to the scripts and comments in the `src/` folder.

## Benchmark Usage

For the usage of our benchmark, please visit google drive link:https://drive.google.com/drive/folders/1iRMfqdk6AJdLM88Dxjxg8POusNYzkbxf?usp=sharing

In the google drive, the input_image and api_image folder corresponds to the original image and the edited image. The human_evaluation_results_updated.csv file corresponds to the human evaluators' scores.


