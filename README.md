# Agentic Seq2Seq Transformer

A modular sequence-to-sequence transformer framework for developing agent-based pipelines with post-deployment specialization, continual adaptation, and efficient fine-tuning, evaluated on both **HumanEval** and **SWE-bench**.

---

## Overview

Agentic Seq2Seq Transformer implements a multi-stage, agentic architecture that enables independent specialization of large language model (LLM) agents for complex, sequential tasks. The framework supports two representative domains:

- **HumanEval**: Specification → Implementation (code generation)
- **SWE-bench**: Issue Analysis → Code Generation (software engineering tasks)

Each agent is modularized with its own parameter set, allowing targeted fine-tuning after deployment while keeping the shared backbone and other agents frozen. This design enables efficient incremental improvement and scalable adaptation in realistic multi-stage pipelines.

- **Model**: Modular encoder–decoder backbone with agent-specific modules (adapters, routers, role heads)
- **Training Modes**:
  - Joint training (all agents)
  - Static agent fine-tuning (single-agent specialization)

---

## Key Features

- **Agent Modularization**  
  Each agent (e.g., Specification, Implementation, Issue Analysis, Code Generation) is implemented as an independent module with isolated parameters.

- **Static Fine-Tuning**  
  Post-deployment adaptation is achieved by fine-tuning a single agent while freezing the backbone and other agents.

- **Cross-Domain Evaluation**  
  The framework is validated on:
  - HumanEval (structured code synthesis)
  - SWE-bench (real-world software issue resolution)

- **Token-Level Metrics**  
  Training and evaluation track:
  - Cross-entropy (CE)
  - Token-level accuracy

- **Pipeline Diagnostics**  
  Measures how upstream agent outputs affect downstream performance.

- **Computational Efficiency Analysis**  
  Quantifies parameter-update savings using effective fine-tuning cost:
  ```
  epochs_ejte = epochs_ft × 0.191
  ```

---

## Project Structure

```
.
├── agentic-transformer-v17-humaneval.ipynb   # HumanEval experiments
├── agentic-transformer-v17-swe-bench.ipynb   # SWE-bench experiments
├── Book1-humaneval.xlsx                      # HumanEval processed results
├── Book1-swe-bench.xlsx                      # SWE-bench processed results
├── output-humaneval.tex                      # LaTeX-ready HumanEval results
├── output-swe-bench.tex                      # LaTeX-ready SWE-bench results
├── README.md
```

---

## Getting Started

1. Clone the repository:
    ```
    git clone https://github.com/hanbyul1/agentic_seq2seq_transformer.git
    cd agentic_seq2seq_transformer
    ```

2. Set up environment:
    - Python 3.8+
    - PyTorch

    ```
    pip install torch sentencepiece
    ```

3. Run experiments:
    - Open notebooks:
      - `agentic-transformer-v17-humaneval.ipynb`
      - `agentic-transformer-v17-swe-bench.ipynb`
    - Execute cells sequentially

---

## Main Results

### HumanEval
- Specification agent:
  - Small but consistent accuracy gains (~+3%)
  - Limited or unstable cross-entropy improvements
- Implementation agent:
  - Strong early improvements (up to +99% relative accuracy gain)
  - Diminishing returns in later rounds
- Indicates early-stage benefit of specialization, followed by saturation

### SWE-bench
- Fine-tuning reduces cross-entropy by up to ~7.6%
- Accuracy improves by up to ~11%
- Demonstrates effectiveness in realistic software engineering tasks

### Computational Efficiency
- Only **19.1% of parameters** updated during fine-tuning
- Achieves up to **~24× savings ratio** in later rounds
- Average savings:
  - SWE-bench: ~11.5×
  - HumanEval: ~14.8×

---

## Research Contributions

- Unified evaluation across **synthetic (HumanEval)** and **real-world (SWE-bench)** tasks
- Demonstrates **agent-level specialization** without retraining the full model
- Analyzes **efficiency vs. performance trade-offs**
- Reveals **early-stage gains and late-stage saturation behavior**

---

## Notes

- Efficiency metrics measure **parameter-update cost**, not wall-clock time
- Results emphasize **relative trends**, not absolute SOTA performance
- Minor fluctuations may occur due to stochastic training

---

## Future Work

- Dynamic routing instead of static specialization
- Improved calibration for specification generation
- Cross-agent feedback mechanisms
- Scaling to larger LLM backbones

---

## Author

Dae-Kyoo Kim  
Oakland University  

---

## License

This project is open-source for research and educational use. Please cite appropriately.

---

## Acknowledgments

- PyTorch for model implementation  
- SentencePiece for tokenization  
- Hugging Face datasets for preprocessing utilities  
- Inspiration from modular and agentic LLM architectures
