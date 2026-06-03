#!/usr/bin/env python3
"""
Post-deployment adaptation validation for agentic-transformer-v17-humaneval.py.

Purpose
-------
This script treats a trained HumanEval agentic checkpoint as a deployed model,
freezes the shared backbone and all non-target agents, adapts only one selected
agent, and reports whether the target role improves while the rest of the model
remains stable.

It is intentionally separate from the main HumanEval baseline script so the
baseline training/evaluation pipeline remains unchanged.

Typical usage
-------------
# Train a deployed checkpoint if missing, then adapt the Specification Agent.
python post_deployment_humaneval_validation.py \
  --baseline ./agentic-transformer-v17-humaneval.py \
  --checkpoint ./outputs/humaneval/deployed_joint.pt \
  --target-agent spec \
  --limit 164 \
  --joint-epochs 2 \
  --adapt-epochs 2

# Use an existing checkpoint and adapt the Implementation Agent.
python post_deployment_humaneval_validation.py \
  --baseline ./agentic-transformer-v17-humaneval.py \
  --checkpoint ./outputs/humaneval/deployed_joint.pt \
  --target-agent impl \
  --adapt-epochs 4

Notes
-----
- "Deployment" is simulated by loading a fixed trained checkpoint.
- "Post-deployment adaptation" is simulated by updating only the selected agent.
- The adaptation data is role-specific:
    * Specification Agent: prompt -> structured specification.
    * Implementation Agent: generated specification context -> implementation.
- The shared encoder/decoder backbone and the non-target agent remain frozen.
- The checkpoint stores the SentencePiece tokenizer model so token IDs remain
  consistent across post-deployment runs.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    PROJECT_DIR
    / "outputs"
    / "humaneval"
    / "deployed_joint.pt"
)

DEFAULT_POSTDEPLOY_DIR = (
    PROJECT_DIR
    / "outputs"
    / "post_deployment_humaneval"
)

DEFAULT_METRICS = (
    DEFAULT_POSTDEPLOY_DIR
    / "metrics.json"
)

# -----------------------------------------------------------------------------
# Dynamic import of the baseline file. The source filename contains hyphens, so
# normal Python import syntax cannot be used.
# -----------------------------------------------------------------------------


def load_baseline_module(path: str):
    baseline_path = Path(path).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    module_name = "agentic_transformer_v17_humaneval"
    spec = importlib.util.spec_from_file_location(module_name, str(baseline_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import baseline module from {baseline_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Config / data / model helpers.
# -----------------------------------------------------------------------------


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def make_cfg_from_args(b, args):
    cfg = copy.deepcopy(b.CFG)

    cfg.seed = args.seed
    cfg.limit = args.limit
    cfg.max_in_len = args.max_in_len
    cfg.max_out_len = args.max_out_len
    cfg.spm_vocab = args.spm_vocab
    cfg.decode_max_len = args.decode_max_len
    cfg.spec_decode_len = args.spec_decode_len
    cfg.impl_decode_len = args.impl_decode_len

    cfg.n_agents = args.n_agents
    cfg.model_dim = args.model_dim
    cfg.n_heads = args.n_heads
    cfg.n_layers_enc = args.n_layers_enc
    cfg.n_layers_dec = args.n_layers_dec
    cfg.max_len_cap = args.max_len_cap

    cfg.pipe_epochs = args.joint_epochs
    cfg.pipe_batch = args.batch_size
    cfg.pipe_lr = args.joint_lr

    cfg.ft_epochs = args.adapt_epochs
    cfg.ft_batch = args.batch_size
    cfg.ft_lr = args.adapt_lr
    cfg.ft_unfreeze_adapters = not args.freeze_adapters
    cfg.ft_unfreeze_dec_norms = args.unfreeze_dec_norms

    cfg.max_repair_attempts = args.max_repair_attempts
    cfg.n_validation_samples = args.generate_samples
    cfg.out_dir = args.out_dir

    cfg.spec_validity_threshold = args.spec_validity_threshold
    cfg.spec_gate_samples = args.spec_gate_samples
    cfg.lambda_constraint = args.lambda_constraint

    # Important: some baseline helper functions read global CFG directly.
    b.CFG = cfg

    return cfg


def build_data(b, cfg):
    b.set_seed(cfg.seed)
    data = b.HumanEvalData(
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
    )
    return data


def get_tokenizer_proto(tok) -> Optional[bytes]:
    sp = getattr(tok, "sp", None)
    if sp is None:
        return None

    if hasattr(sp, "serialized_model_proto"):
        try:
            return sp.serialized_model_proto()
        except Exception:
            return None

    if hasattr(sp, "SerializeToString"):
        try:
            return sp.SerializeToString()
        except Exception:
            return None

    return None


def restore_tokenizer_from_proto(tok, proto: Optional[bytes]) -> bool:
    if not proto:
        return False

    sp = getattr(tok, "sp", None)
    if sp is None:
        return False

    try:
        if hasattr(sp, "LoadFromSerializedProto"):
            sp.LoadFromSerializedProto(proto)
        elif hasattr(sp, "load_from_serialized_proto"):
            sp.load_from_serialized_proto(proto)
        else:
            return False

        tok.vocab_size = sp.get_piece_size()
        tok.pad_idx = 1
        tok.unk_idx = 0
        tok.bos_idx = 2
        tok.eos_idx = 3
        return True
    except Exception as exc:
        print(f"[Tokenizer][WARNING] Could not restore serialized tokenizer: {exc}")
        return False


def build_splits_and_model_from_data(b, data, cfg) -> Tuple[Any, Dict[str, Any]]:
    ids, X, Y, P = data.as_tensors_with_spec_targets(spec_max_len=cfg.max_in_len)

    n = len(ids)
    if n < 4:
        raise RuntimeError(f"Need at least 4 HumanEval examples; got {n}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)

    ids = [ids[i] for i in perm.tolist()]
    X = X[perm]
    Y = Y[perm]
    P = P[perm]

    ADAPT_HOLDOUT = 33

    deploy_n = n - ADAPT_HOLDOUT
    if deploy_n < 4:
        raise RuntimeError(
            f"Need at least {ADAPT_HOLDOUT + 4} HumanEval examples; got {n}."
        )

    split = max(1, int(deploy_n * 0.8))
    if split >= deploy_n:
        split = deploy_n - 1

    max_len_for_model = max(
        cfg.max_len_cap,
        cfg.max_in_len + max(
            cfg.max_out_len,
            cfg.spec_decode_len,
            cfg.impl_decode_len,
        ) + 8,
    )

    model = b.AgenticTransformerSeq2Seq(
        vocab_size=data.tok.vocab_size,
        n_agents=cfg.n_agents,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_heads,
        n_layers_enc=cfg.n_layers_enc,
        n_layers_dec=cfg.n_layers_dec,
        max_len=max_len_for_model,
        pad_idx=data.tok.pad,
    )

    tensors = {
        "ids": ids,
        "X": X,
        "Y": Y,
        "P": P,

        "ids_train": ids[:split],
        "ids_test": ids[split:deploy_n],
        "ids_adapt": ids[deploy_n:],

        "X_train": X[:split],
        "Y_train": Y[:split],
        "P_train": P[:split],

        "X_test": X[split:deploy_n],
        "Y_test": Y[split:deploy_n],
        "P_test": P[split:deploy_n],

        "X_adapt": X[deploy_n:],
        "Y_adapt": Y[deploy_n:],
        "P_adapt": P[deploy_n:],

        "train_size": split,
        "test_size": deploy_n - split,
        "adapt_size": n - deploy_n,
        "max_len_for_model": max_len_for_model,
    }

    return model, tensors


# -----------------------------------------------------------------------------
# Checkpoint helpers.
# -----------------------------------------------------------------------------


def save_checkpoint(path: str, model, cfg, data, meta: Dict[str, Any]) -> None:
    ckpt_path = Path(path).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dataclass_to_dict(cfg),
            "tokenizer_model_proto": get_tokenizer_proto(data.tok),
            "meta": meta,
        },
        ckpt_path,
    )

    print(f"[Checkpoint] saved HumanEval deployed checkpoint: {ckpt_path}")


def read_checkpoint_if_exists(path: str, device: str) -> Optional[Dict[str, Any]]:
    ckpt_path = Path(path).expanduser().resolve()
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt

    return {"model_state_dict": ckpt}


def load_checkpoint_into_model(path: str, model, ckpt: Dict[str, Any]) -> None:
    ckpt_path = Path(path).expanduser().resolve()
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"[Checkpoint] loaded: {ckpt_path}")


# -----------------------------------------------------------------------------
# Metrics and evaluation helpers.
# -----------------------------------------------------------------------------


def clone_frozen_reference(model) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def changed_parameter_report(model, before: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    changed = []
    unchanged = []
    max_abs_delta = 0.0

    for name, p in model.named_parameters():
        old = before.get(name)
        new = p.detach().cpu()

        if old is None or old.shape != new.shape:
            changed.append(name)
            max_abs_delta = float("inf")
            continue

        delta = (new - old).abs().max().item() if new.numel() else 0.0
        max_abs_delta = max(max_abs_delta, delta)

        if delta > 1e-9:
            changed.append(name)
        else:
            unchanged.append(name)

    return {
        "num_changed_tensors": len(changed),
        "num_unchanged_tensors": len(unchanged),
        "max_abs_delta": max_abs_delta,
        "changed_tensors": changed,
    }


def classify_changed_tensors(changed_tensors, target_agent_id: int) -> Dict[str, Any]:
    target_prefix = f"routing.agents.{target_agent_id}."
    target_changed = [n for n in changed_tensors if n.startswith(target_prefix)]
    non_target_changed = [n for n in changed_tensors if not n.startswith(target_prefix)]

    return {
        "target_changed_count": len(target_changed),
        "non_target_changed_count": len(non_target_changed),
        "target_changed_tensors": target_changed,
        "non_target_changed_tensors": non_target_changed,
    }


def compute_trainable_stats(model) -> Dict[str, float]:
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "AC_trainable_ratio": float(trainable / max(total, 1)),
    }


def build_gold_spec_context(b, tok, P, cfg):
    return b.build_spec_plus_anchor_context(
        tok,
        P,
        raw_x=None,
        max_in_len=cfg.max_in_len,
    )[:, : cfg.max_in_len]


def build_generated_spec_context(b, model, tok, X, cfg, device):
    return b.build_agent_spec_context_inputs(
        model,
        tok,
        X.to(device),
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=device,
    )[:, : cfg.max_in_len]


@torch.no_grad()
def evaluate_spec_consistency(b, model, tok, X, cfg, device: str, n_samples: int, repeats: int) -> float:
    """Generate each spec several times and compute lexical consistency.

    Greedy decoding is deterministic in the current baseline, so this often
    returns high consistency. It is still useful as an explicit stability check.
    """

    model.to(device)
    model.eval()

    all_specs = []
    X_eval = X[: min(n_samples, X.size(0))].to(device)

    for _ in range(max(1, repeats)):
        spec_rows = b.generate_valid_spec_rows(
            model,
            tok,
            X_eval,
            spec_max_len=cfg.spec_decode_len,
            device=device,
            fallback_to_prompt_docstring=False,
        )

        for row in spec_rows:
            txt = tok.decode(
                [int(t) for t in row.tolist() if int(t) not in (tok.pad, tok.bos, tok.eos)]
            )
            all_specs.append(txt)

    return float(b.pairwise_consistency(all_specs)) if all_specs else 0.0


@torch.no_grad()
def evaluate_snapshot(b, model, data, tensors, cfg, device: str) -> Dict[str, float]:
    X_test = tensors["X_test"]
    Y_test = tensors["Y_test"]
    P_test = tensors["P_test"]

    spec_ce, spec_acc = b._eval_spec_ce_acc(model, X_test, P_test, device=device)

    spec_validity = b.evaluate_spec_generation_validity(
        model,
        data.tok,
        X_test,
        spec_max_len=cfg.spec_decode_len,
        device=device,
        n_samples=min(cfg.spec_gate_samples, X_test.size(0)),
    )

    spec_consistency = evaluate_spec_consistency(
        b,
        model,
        data.tok,
        X_test,
        cfg,
        device,
        n_samples=min(8, X_test.size(0)),
        repeats=3,
    )

    # Implementation without specification context.
    impl_raw_ce, impl_raw_acc = b._eval_impl_ce_acc(model, X_test, Y_test, device=device)

    # Implementation with gold specification context.
    X_gold_spec = build_gold_spec_context(b, data.tok, P_test, cfg)
    impl_gold_spec_ce, impl_gold_spec_acc = b._eval_impl_ce_acc(
        model,
        X_gold_spec,
        Y_test,
        device=device,
    )

    # Implementation with generated specification context.
    X_generated_spec = build_generated_spec_context(b, model, data.tok, X_test, cfg, device)
    impl_generated_spec_ce, impl_generated_spec_acc = b._eval_impl_ce_acc(
        model,
        X_generated_spec,
        Y_test,
        device=device,
    )

    # Full deployed pipeline evaluation.
    pipe_ce, pipe_acc = b.eval_pipeline(
        model,
        data.tok,
        X_test,
        Y_test,
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=device,
    )

    return {
        "spec_ce": spec_ce,
        "spec_tok_acc": spec_acc,
        "spec_validity": float(spec_validity),
        "spec_consistency": float(spec_consistency),
        "impl_raw_ce": impl_raw_ce,
        "impl_raw_tok_acc": impl_raw_acc,
        "impl_gold_spec_ce": impl_gold_spec_ce,
        "impl_gold_spec_tok_acc": impl_gold_spec_acc,
        "impl_generated_spec_ce": impl_generated_spec_ce,
        "impl_generated_spec_tok_acc": impl_generated_spec_acc,
        "pipeline_ce": pipe_ce,
        "pipeline_tok_acc": pipe_acc,
        "pipeline_lift_ce_vs_raw": impl_generated_spec_ce - impl_raw_ce,
        "pipeline_lift_acc_vs_raw": impl_generated_spec_acc - impl_raw_acc,
    }


def delta_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for key, before_value in before.items():
        if key in after:
            out[f"delta_{key}"] = after[key] - before_value
    return out


def print_metrics_block(title: str, metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        elif isinstance(value, list):
            print(f"{key}: {len(value)} item(s)")
        else:
            print(f"{key}: {value}")


def assert_parameter_locality(locality: Dict[str, Any], strict: bool) -> None:
    if locality["non_target_changed_count"] == 0:
        print("\n[OK] Parameter locality passed: only target-agent tensors changed.")
        return

    print("\n[WARNING] Non-target/backbone tensors changed:")
    for name in locality["non_target_changed_tensors"][:40]:
        print(f"  - {name}")
    if len(locality["non_target_changed_tensors"]) > 40:
        print("  ...")

    if strict:
        raise RuntimeError("Parameter locality failed under --strict-locality.")


# -----------------------------------------------------------------------------
# Training/deployment simulation.
# -----------------------------------------------------------------------------


def train_deployed_checkpoint(b, model, data, tensors, cfg, device):
    """Train the HumanEval pipeline up to the deployed checkpoint.

    This follows the HumanEval baseline's deployed pipeline:
      Stage 0: train Specification Agent on Prompt -> Spec with backbone trainable.
      Stage 1: freeze Spec + backbone and train Implementation Agent from generated Spec.
    """

    print("\n[Deployment Simulation] Stage 0: SPEC supervision")

    b.train_spec_supervised(
        model,
        tensors["X_train"],
        tensors["P_train"],
        epochs=cfg.ft_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=device,
        unfreeze_backbone=True,
        unfreeze_A_adapter=True,
        unfreeze_dec_norms=True,
    )

    print("\n[Deployment Simulation] Stage 1: Generated SPEC -> IMPL")

    b.train_stage1_interleaved(
        model,
        tensors["X_train"],
        tensors["Y_train"],
        tensors["P_train"],
        tok=data.tok,
        spec_max_len=cfg.spec_decode_len,
        epochs=cfg.pipe_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=device,
        unfreeze_backbone=False,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=False,
        max_in_len=cfg.max_in_len,
    )


def set_target_trainable(b, model, target_agent_id: int, cfg) -> None:
    b._set_ft_requires_grad(
        model,
        user_id=target_agent_id,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
    )


def adapt_target_agent(b, model, data, tensors, cfg, target_agent_id: int, args, device: str) -> None:
    if target_agent_id == b.AGENT_SPECIFICATION:
        b.fine_tune_static(
            model,
            tensors["X_adapt"],
            tensors["Y_adapt"],
            user_id=target_agent_id,
            epochs=cfg.ft_epochs,
            batch_size=cfg.ft_batch,
            lr=cfg.ft_lr,
            weight_decay=args.weight_decay,
            unfreeze_adapters=cfg.ft_unfreeze_adapters,
            unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
            unfreeze_decoder_tail_blocks=0,
            device=device,
            tok=data.tok,
            P=tensors["P_adapt"],
            gist_ctx_fn=None,
            max_in_len=cfg.max_in_len,
            patience=args.patience,
        )
        return

    # Implementation adaptation uses generated specification input.
    b.fine_tune_static(
        model,
        tensors["X_adapt"],
        tensors["Y_adapt"],
        user_id=target_agent_id,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=args.weight_decay,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=0,
        device=device,
        tok=data.tok,
        gist_ctx_fn=lambda xb: b.build_agent_spec_context_inputs(
            model,
            data.tok,
            xb,
            spec_max_len=cfg.spec_decode_len,
            max_in_len=cfg.max_in_len,
            device=device,
        ),
        X_gist=None,
        max_in_len=cfg.max_in_len,
        patience=args.patience,
    )


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------


def run(args) -> None:

    Path(args.checkpoint).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.out_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(args.metrics_out).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    b = load_baseline_module(args.baseline)
    cfg = make_cfg_from_args(b, args)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = read_checkpoint_if_exists(str(checkpoint_path), device)

    print("[Setup] Building HumanEval data")
    data = build_data(b, cfg)

    if ckpt is not None:
        restored = restore_tokenizer_from_proto(data.tok, ckpt.get("tokenizer_model_proto"))
        if restored:
            print("[Tokenizer] Restored tokenizer from deployed checkpoint.")
        else:
            print(
                "[Tokenizer][WARNING] Checkpoint has no tokenizer proto or restore failed. "
                "Using a newly trained tokenizer; this is safe only if it matches the checkpoint tokenizer."
            )

    print("[Setup] Building deterministic split and model shell")
    model, tensors = build_splits_and_model_from_data(b, data, cfg)
    model.to(device)

    print(
        f"[Data] train={tensors['train_size']} | test={tensors['test_size']} | adapt={tensors['adapt_size']} | "
        f"vocab={data.tok.vocab_size} | max_len={tensors['max_len_for_model']}"
    )

    if ckpt is not None:
        load_checkpoint_into_model(str(checkpoint_path), model, ckpt)
    else:

        print(
            "[Checkpoint] HumanEval deployed checkpoint not found.\n"
            "[Checkpoint] Training fallback checkpoint automatically. "
            "For baseline-aligned evaluation, first run the HumanEval baseline "
            "script so it saves outputs/humaneval/deployed_joint.pt."
        )

        print("\n[Deployment Simulation] No checkpoint found.")
        train_deployed_checkpoint(
            b,
            model,
            data,
            tensors,
            cfg,
            device,
        )

        save_checkpoint(
            str(checkpoint_path),
            model,
            cfg,
            data,
            meta={
                "meaning": (
                    "HumanEval trained checkpoint "
                    "treated as deployed model"
                ),
                "train_size": tensors["train_size"],
                "test_size": tensors["test_size"],
                "baseline": str(
                    Path(args.baseline)
                    .expanduser()
                    .resolve()
                ),
            },
        )

    print("\n[Deployment] Fixed checkpoint evaluation before adaptation.")
    before = evaluate_snapshot(b, model, data, tensors, cfg, device)
    print_metrics_block("Before post-deployment adaptation", before)

    # ------------------------------------------------------------------
    # Generate BEFORE-adaptation samples.
    # ------------------------------------------------------------------

    if args.generate_samples > 0:   

        before_dir = (
            Path(args.out_dir)
            / "before_adaptation"
        )

        before_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        b.generate_validated_samples(
            model,
            data.tok,
            tensors["ids_test"],
            tensors["X_test"],
            output_dir=str(before_dir),
            sample_prefix=f"before-humaneval-{args.target_agent}",
            spec_max_len=cfg.spec_decode_len,
            out_max_len=cfg.impl_decode_len,
            max_in_len=cfg.max_in_len,
            n_samples=args.generate_samples,
            max_repair_attempts=cfg.max_repair_attempts,
            device=device,
        )
        
    if args.target_agent == "spec":
        target_agent_id = b.AGENT_SPECIFICATION
        target_label = "SPECIFICATION"
    elif args.target_agent == "impl":
        target_agent_id = b.AGENT_IMPLEMENTATION
        target_label = "IMPLEMENTATION"
    else:
        raise ValueError("--target-agent must be one of: spec, impl")

    set_target_trainable(b, model, target_agent_id, cfg)
    ac_stats = compute_trainable_stats(model)

    # Baseline implementation may also expose a richer efficiency function.
    rich_efficiency = {}
    if hasattr(b, "compute_agentic_efficiency_stats"):
        try:
            rich_efficiency = b.compute_agentic_efficiency_stats(
                model,
                active_agent_id=target_agent_id,
            )
        except Exception:
            rich_efficiency = {}

    print_metrics_block(
        f"Adaptation cost setup for target agent: {target_label}",
        {
            "target_agent_id": target_agent_id,
            **ac_stats,
            "backbone_frozen": True,
            "non_target_agent_frozen": True,
            "decoder_norms_trainable": bool(cfg.ft_unfreeze_dec_norms),
        },
    )

    if rich_efficiency:
        print_metrics_block("Detailed parameter-efficiency statistics", rich_efficiency)

    parameter_snapshot = clone_frozen_reference(model)

    print("\n[Post-Deployment Adaptation]")
    print(
        f"Updating only target agent: {target_label}. "
        "The shared backbone and non-target agent remain frozen unless decoder norms are explicitly enabled."
    )

    adapt_target_agent(b, model, data, tensors, cfg, target_agent_id, args, device)

    after = evaluate_snapshot(b, model, data, tensors, cfg, device)
    deltas = delta_metrics(before, after)

    print_metrics_block("After post-deployment adaptation", after)
    print_metrics_block("Before/after deltas", deltas)

    changed = changed_parameter_report(model, parameter_snapshot)
    locality = classify_changed_tensors(changed["changed_tensors"], target_agent_id)

    print_metrics_block(
        "Parameter-locality validation",
        {
            "changed_tensor_count": changed["num_changed_tensors"],
            "unchanged_tensor_count": changed["num_unchanged_tensors"],
            "max_abs_parameter_delta": changed["max_abs_delta"],
            "target_agent_changed_tensors": locality["target_changed_count"],
            "non_target_or_backbone_changed_tensors": locality["non_target_changed_count"],
        },
    )

    assert_parameter_locality(locality, args.strict_locality)

    # ------------------------------------------------------------------
    # Generate AFTER-adaptation samples.
    # ------------------------------------------------------------------

    if args.generate_samples > 0:

        after_dir = (
            Path(args.out_dir)
            / "after_adaptation"
        )

        after_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        b.generate_validated_samples(
            model,
            data.tok,
            tensors["ids_test"],
            tensors["X_test"],
            output_dir=str(after_dir),
            sample_prefix=f"after-humaneval-{args.target_agent}",
            spec_max_len=cfg.spec_decode_len,
            out_max_len=cfg.impl_decode_len,
            max_in_len=cfg.max_in_len,
            n_samples=args.generate_samples,
            max_repair_attempts=cfg.max_repair_attempts,
            device=device,
        )

    if args.save_adapted_checkpoint:
        save_checkpoint(
            args.save_adapted_checkpoint,
            model,
            cfg,
            data,
            meta={
                "meaning": "HumanEval post-deployment adapted checkpoint",
                "target_agent": args.target_agent,
                "target_agent_id": target_agent_id,
                "source_checkpoint": str(checkpoint_path),
            },
        )

    metrics_path = Path(args.metrics_out).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "target_agent": args.target_agent,
        "target_agent_id": target_agent_id,
        "config": dataclass_to_dict(cfg),
        "before": before,
        "after": after,
        "delta": deltas,
        "adaptation_cost": ac_stats,
        "detailed_efficiency": rich_efficiency,
        "parameter_locality": {
            "changed_tensor_count": changed["num_changed_tensors"],
            "unchanged_tensor_count": changed["num_unchanged_tensors"],
            "max_abs_parameter_delta": changed["max_abs_delta"],
            **locality,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[Metrics] saved: {metrics_path}")


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def parse_args():
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Validate agent-specific post-deployment adaptation on HumanEval."
    )

    p.add_argument(
        "--baseline",
        default=str(script_dir / "agentic-transformer-v17-humaneval.py"),
        help="Path to agentic-transformer-v17-humaneval.py",
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )
 
    p.add_argument(
        "--target-agent",
        choices=["spec", "impl"],
        default="spec",
        help="Agent to adapt after deployment.",
    )

    # Data/model knobs.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=164)
    p.add_argument("--max-in-len", type=int, default=512)
    p.add_argument("--max-out-len", type=int, default=384)
    p.add_argument("--spm-vocab", type=int, default=4096)
    p.add_argument("--decode-max-len", type=int, default=512)
    p.add_argument("--spec-decode-len", type=int, default=320)
    p.add_argument("--impl-decode-len", type=int, default=160)

    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--model-dim", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers-enc", type=int, default=4)
    p.add_argument("--n-layers-dec", type=int, default=4)
    p.add_argument("--max-len-cap", type=int, default=640)

    # Training knobs.
    p.add_argument("--joint-epochs", type=int, default=20)
    p.add_argument("--adapt-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--joint-lr", type=float, default=2e-4)
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--freeze-adapters", action="store_true")
    p.add_argument(
        "--unfreeze-dec-norms",
        action="store_true",
        help="Allow decoder normalization parameters to update during adaptation.",
    )
    p.add_argument("--lambda-constraint", type=float, default=0.01)

    # Validation / output knobs.
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_POSTDEPLOY_DIR),
    )
    p.add_argument(
        "--metrics-out",
        default=str(DEFAULT_METRICS),
    )
    p.add_argument("--generate-samples", type=int, default=10)
    p.add_argument("--max-repair-attempts", type=int, default=3)
    p.add_argument("--spec-validity-threshold", type=float, default=0.90)
    p.add_argument("--spec-gate-samples", type=int, default=33)

    p.add_argument(
        "--strict-locality",
        action="store_true",
        help="Fail if any non-target/backbone tensor changes during adaptation.",
    )
    p.add_argument(
        "--save-adapted-checkpoint",
        default="",
        help="Optional path to save the adapted checkpoint.",
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
