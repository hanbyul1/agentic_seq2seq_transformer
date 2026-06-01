# Agentic Seq2Seq Transformer

A research repository containing implementations and experimental artifacts for evaluating a **role-specialized Agentic Seq2Seq Transformer** against a **Mixture-of-Experts (MoE) baseline** on software engineering and code generation benchmarks.

The repository includes:

- Agentic Transformer implementations
- MoE baseline implementations
- HumanEval experiments
- SWE-bench experiments
- Prediction outputs
- Visualization scripts
- Generated LaTeX result tables

---

## Repository Structure

```text
agentic_seq2seq_transformer/
│
├── agentic-transformer-v17-humaneval.py
├── agentic-transformer-v17-swe-bench.py
│
├── moe-humaneval.py
├── moe-swe-bench.py
│
├── comparison-visualization.py
│
├── agentic-transformer-humaneval-output.tex
├── agentic-transformer-swe-bench-output.tex
├── moe-humaneval-output.tex
├── moe-swe-bench-output.tex
│
└── outputs/
    │
    ├── humaneval/
    │   ├── humaneval-sample1.txt
    │   ├── humaneval-sample2.txt
    │   ├── ...
    │   └── predictions.jsonl
    │
    ├── moe_humaneval/
    │   ├── humaneval-sample1.txt
    │   ├── ...
    │   └── predictions.jsonl
    │
    ├── swebench/
    │   ├── swebench-sample1.txt
    │   ├── ...
    │   └── predictions.jsonl
    │
    └── swebench_moe/
        ├── swebench-sample1.txt
        ├── ...
        └── predictions.jsonl
```

---

## Overview

This repository investigates whether a lightweight role-specialized transformer architecture can achieve competitive performance relative to a traditional Mixture-of-Experts (MoE) model while enabling:

- explicit role specialization,
- structured multi-stage generation,
- easier post-deployment adaptation,
- improved maintainability,
- reduced retraining costs.

The proposed architecture employs fixed agent roles instead of dynamically activated experts.

---

## Agentic Transformer

The Agentic Transformer consists of specialized roles operating sequentially within a static pipeline.

### HumanEval Pipeline

```text
Problem Description
        │
        ▼
Specification Agent
        │
        ▼
Implementation Agent
        │
        ▼
Generated Solution
```

### SWE-bench Pipeline

```text
Issue Description
        │
        ▼
Issue Analysis Agent
        │
        ▼
Code Generation Agent
        │
        ▼
Patch Candidate
```

Characteristics:

- Static role assignment
- Structured intermediate representations
- Shared backbone transformer
- Role-specific adapters and output heads
- Pipeline-based generation

---

## MoE Baseline

The repository also contains a Mixture-of-Experts baseline implementation.

Characteristics:

- Dynamic routing
- Multiple experts
- Learned gating network
- Shared token representation
- Expert activation per input

The MoE implementation serves as a comparison point for evaluating:

- output quality,
- adaptation cost,
- computational efficiency,
- architectural maintainability.

---

## Benchmarks

### HumanEval

HumanEval evaluates code generation capability using programming problems and reference unit tests.

Evaluated outputs include:

- generated solutions,
- intermediate specifications,
- prediction logs.

### SWE-bench

SWE-bench evaluates software issue resolution using real-world GitHub issues.

Evaluated outputs include:

- issue analyses,
- generated patches,
- prediction logs.

---

## Output Artifacts

The `outputs/` directory contains generated predictions and experiment artifacts.

### HumanEval

```text
outputs/humaneval/
```

Contains:

- generated specifications,
- generated implementations,
- prediction records,
- experiment logs.

### MoE HumanEval

```text
outputs/moe_humaneval/
```

Contains corresponding MoE outputs.

### SWE-bench

```text
outputs/swebench/
```

Contains:

- issue analyses,
- generated patches,
- prediction records.

### MoE SWE-bench

```text
outputs/swebench_moe/
```

Contains corresponding MoE outputs.

---

## Result Tables

Generated LaTeX tables:

| File | Description |
|--------|--------|
| `agentic-transformer-humaneval-output.tex` | HumanEval results for Agentic Transformer |
| `agentic-transformer-swe-bench-output.tex` | SWE-bench results for Agentic Transformer |
| `moe-humaneval-output.tex` | HumanEval results for MoE |
| `moe-swe-bench-output.tex` | SWE-bench results for MoE |

These files can be directly included in academic manuscripts.

---

## Visualization

The script:

```text
comparison-visualization.py
```

generates comparative plots and visual summaries of:

- Agentic Transformer performance,
- MoE performance,
- benchmark-specific metrics,
- adaptation efficiency.

---

## Research Goals

### RQ1

**To what extent can agentic behavior be realized through native architectural integration within transformer models for software engineering tasks?**

This study investigates whether agent roles can be embedded directly within a transformer architecture, enabling role specialization and structured workflow execution within a unified model.

### RQ2

**To what extent can agentic modeling reduce post-deployment adaptation cost while maintaining competitive output quality?**

This study investigates whether agent-level specialization can enable localized post-deployment adaptation by updating only selected agents while preserving competitive output quality.

---

## Experimental Workflow

```text
Training
    │
    ▼
Role Specialization
    │
    ▼
Benchmark Evaluation
    │
    ▼
Prediction Generation
    │
    ▼
Visualization
    │
    ▼
LaTeX Result Generation
```

---

## Requirements

Typical dependencies include:

```bash
pip install torch
pip install numpy
pip install pandas
pip install matplotlib
```

Additional requirements may vary depending on the experiment configuration.

---

## Citation

If you use this repository in academic work, please cite the associated publication when available.

```bibtex
@misc{agentic_seq2seq_transformer,
  title={Agentic Seq2Seq Transformer},
  author={Dae-Kyoo Kim},
  year={2026}
}
```

---

## License

This repository is intended for academic research and experimentation.
Please refer to the repository license for usage terms.
