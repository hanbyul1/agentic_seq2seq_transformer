Agentic Seq2Seq Transformer

A modular sequence-to-sequence transformer framework for developing agent-based pipelines with post-deployment specialization, continual adaptation, and efficient fine-tuning, specifically for software engineering workflows.

Overview
Agentic Seq2Seq Transformer implements a multi-stage, agentic architecture that enables independent specialization of large language model (LLM) agents for complex, sequential tasks (e.g., issue analysis and code generation). Each agent is modularized with its own parameter set, allowing for targeted fine-tuning after deployment while keeping the shared backbone and other agents frozen. This design supports efficient, incremental improvement and scalable adaptation in realistic pipelines.
•	Model: Modular encoder-decoder backbone with multiple agent modules (adapter, router, role heads).
•	Pipeline: Supports joint training and static agent fine-tuning for continual improvement.

Key Features
•	Agent Modularization: Each agent (e.g., issue analysis, code generation) is an independent, parameter-isolated module.
•	Static Fine-Tuning: Enables post-deployment adaptation of a single agent, leaving the backbone and other agents untouched.
•	Token-Level Evaluation: Measures cross-entropy and accuracy for both training and test data.
•	Diagnostic Metrics: Tracks improvements, output quality, and the effect of upstream agent outputs.
•	Flexible Utilities: Data processing, tokenization (SentencePiece), and evaluation are modular and script-driven.

Project Structure
•	agentic-transformer-v17.ipynb: Main Jupyter notebook for model, training, and evaluation.
•	output.tex: Plain text file containing evaluation results and experiment outputs (not a full LaTeX paper).
•	README.md: (You are here) Project summary, installation, and usage instructions.

Getting Started
1.	Clone the repository
git clone https://github.com/hanbyul1/agentic_seq2seq_transformer.git
cd agentic_seq2seq_transformer

2.	Set up environment
o	Requires Python 3.8+ and PyTorch.
o	Install dependencies as needed (pip install torch sentencepiece etc).
3.	Run the notebook
o	Open agentic-transformer-v17.ipynb in Jupyter Notebook or JupyterLab.
o	Follow the cells to load data, configure the model, run training (joint and agent-specific fine-tuning), and analyze results.

Main Results
•	Evaluation: Tested on SWE-bench, a dataset of real-world software engineering issues and patches.
•	Performance: Fine-tuning a single agent reduces cross-entropy loss by up to 7.6% and increases accuracy by up to 11%, while using only 19% of the model parameters per agent—yielding 11.5x computational savings versus retraining the entire model.

Research Context
This repository supports research on modular LLM-based agents for software engineering. The code and outputs are provided for transparency and reproducibility.

License
This project is open-source for research and educational purposes. Please cite if you use this work.

Acknowledgments
•	Model and dataset built on PyTorch and Hugging Face Datasets.
•	Inspired by research in agentic architectures, modular learning, and software engineering automation.
