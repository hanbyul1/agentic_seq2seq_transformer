# Agentic Seq2Seq Transformer

A modular sequence-to-sequence transformer framework for developing agent-based pipelines with post-deployment specialization, continual adaptation, and efficient fine-tuning, specifically for software engineering workflows.

## Overview

Agentic Seq2Seq Transformer implements a multi-stage, agentic architecture that enables independent specialization of large language model (LLM) agents for complex, sequential tasks (e.g., issue analysis and code generation). Each agent is modularized with its own parameter set, allowing for targeted fine-tuning after deployment while keeping the shared backbone and other agents frozen. This design supports efficient, incremental improvement and scalable adaptation in realistic pipelines.

- Model: Modular encoder-decoder backbone with multiple agent modules (adapter, router, role heads).
- Pipeline: Supports joint training and static agent fine-tuning for continual improvement.

## Key Features

- Agent Modularization: Each agent in the pipeline (for example, for issue analysis or code generation) is implemented as an independent module, with its own parameter set isolated from others.
- Static Fine-Tuning: The framework supports post-deployment adaptation by enabling fine-tuning of a single agent, while leaving all other agents and the backbone unchanged.
- Token-Level Evaluation: Training and evaluation routines measure cross-entropy loss and accuracy at the token level for both training and test splits.
- Diagnostic Metrics: The system tracks performance improvements, output quality, and quantifies the effect of upstream agent outputs on downstream tasks.
- Flexible Utilities: Data processing, tokenization (using SentencePiece), and evaluation utilities are implemented in a modular, script-driven manner for easy adaptation.

## Project Structure

- agentic-transformer-v17.ipynb: Main Jupyter notebook containing the model implementation, training workflow, and evaluation logic.
- output.tex: Plain text file with evaluation results and experiment outputs (not a full LaTeX paper).
- README.md: (This file) Project summary, installation instructions, and usage guidance.

## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/hanbyul1/agentic_seq2seq_transformer.git
    cd agentic_seq2seq_transformer
    ```
2. Set up your environment:
    - Requires Python 3.8 or higher and PyTorch.
    - Install required dependencies using pip:
        ```
        pip install torch sentencepiece
        ```
3. Run the main notebook:
    - Open agentic-transformer-v17.ipynb in Jupyter Notebook or JupyterLab.
    - Follow the cells to load data, configure the model, perform joint training and agent-specific fine-tuning, and analyze results.

## Main Results

- Evaluation: The framework has been evaluated on SWE-bench, a dataset of real-world software engineering issues and patches.
- Performance: Fine-tuning a single agent with this model reduces cross-entropy loss by up to 7.6% and increases token-level accuracy by up to 11%, while requiring updates to only 19% of the model’s parameters for each agent. This yields approximately 11.5× computational savings compared to retraining the full model.

## Research Context

This repository supports research on modular LLM-based agents in software engineering pipelines. The code and outputs are provided for transparency and reproducibility.

## License

This project is open-source for research and educational use. Please provide appropriate citation if you use this work.

## Acknowledgments

- The model is implemented using PyTorch, with data processing supported by Hugging Face Datasets.
- The approach is inspired by recent work in agentic model architectures, modular learning, and software engineering automation.
