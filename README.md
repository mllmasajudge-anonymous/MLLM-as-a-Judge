MLLM-as-a-Judge
================

This repository contains the data, prompts, results, and code for our **MLLM-as-a-Judge** project.

## Folder Overview

- **`benchmark/`**: Contains the benchmark we constructed for evaluating MLLMs as judges.

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


