#!/usr/bin/env python3
"""
Post-deployment adaptation validation for agentic-transformer-v17-swe-bench.py.

Purpose
-------
This script treats a jointly trained checkpoint as a deployed model, freezes the
shared backbone and all non-target agents, adapts only one selected agent, and
reports whether the target role improves while the rest of the model remains
stable.

It is intentionally separate from the main SWE-bench baseline script so the
baseline training/evaluation pipeline remains unchanged.

Typical usage
-------------
# 1) Train a small joint checkpoint if none exists, then adapt the issue-analysis agent.
python post_deployment_adaptation_eval.py \
  --target-agent issue
  --demo-data \
  --limit 128 \
  --joint-epochs 2 \
  --adapt-epochs 2

# 2) Use an existing checkpoint and adapt the code-generation agent.
python post_deployment_adaptation_eval.py \
  --baseline ./agentic-transformer-v17-swe-bench.py \
  --checkpoint ./outputs/swebench/deployed_joint.pt \
  --target-agent code \
  --limit 1000 \
  --adapt-epochs 4

Notes
-----
- "Deployment" is simulated by loading a fixed jointly trained checkpoint.
- "Post-deployment adaptation" is simulated by updating only the selected agent.
- The adaptation data is role-specific:
    * Issue-analysis agent: issue -> structured issue gist.
    * Code-generation agent: issue gist + raw issue context -> patch/code.
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

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "outputs" / "swebench" / "checkpoints" / "deployed_joint.pt"
)
DEFAULT_POSTDEPLOY_DIR = (
    PROJECT_DIR / "outputs" / "post_deployment_swebench"
)

DEFAULT_METRICS = DEFAULT_POSTDEPLOY_DIR / "metrics.json"

# -----------------------------------------------------------------------------
# Dynamic import of the baseline file. The source filename contains hyphens, so
# normal Python import syntax cannot be used.
# -----------------------------------------------------------------------------


def load_baseline_module(path: str):
    baseline_path = Path(path).expanduser().resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    module_name = "agentic_transformer_v17_swe_bench"
    spec = importlib.util.spec_from_file_location(module_name, str(baseline_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import baseline module from {baseline_path}")

    module = importlib.util.module_from_spec(spec)

    # Important for dataclasses and other module-level reflection.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Config, tokenizer, data, and model setup helpers.
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
    cfg.demo_data = args.demo_data
    cfg.max_in_len = args.max_in_len
    cfg.max_out_len = args.max_out_len
    cfg.spm_vocab = args.spm_vocab

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

    cfg.decode_max_len = args.decode_max_len
    cfg.max_repair_attempts = args.max_repair_attempts
    cfg.out_dir = args.out_dir

    return cfg


def build_data(b, cfg):
    b.set_seed(cfg.seed)

    data = b.SWEText2PatchData(
        split="train",
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
        demo_data=cfg.demo_data,
    )
    return data


def get_tokenizer_proto(tok) -> Optional[bytes]:
    """Return serialized SentencePiece model bytes if the API is available."""
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
    """Restore SentencePiece model bytes into an already-created tokenizer."""
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


def build_constraint_token_ids(b, data):
    diff_token_ids = []
    bad_token_ids = []

    for tok_piece in [
        "def",
        "class",
        "return",
        "import",
        "if",
        "else",
        "for",
        "while",
        "try",
        "except",
        ":",
        "(",
        ")",
    ]:
        tid = data.tok.sp.piece_to_id(tok_piece)
        if tid not in (data.tok.unk_idx, data.tok.pad, data.tok.bos, data.tok.eos):
            if tid not in diff_token_ids:
                diff_token_ids.append(tid)

    tid = data.tok.sp.piece_to_id("<unk>")
    if tid >= 0:
        bad_token_ids.append(tid)

    return diff_token_ids, bad_token_ids


def build_splits_and_model_from_data(b, data, cfg) -> Tuple[Any, Dict[str, Any]]:
    """Build deterministic train/test tensors and a model shell.

    This should be called after tokenizer restoration, if a checkpoint tokenizer is
    available, so X/Y/P token IDs match the checkpoint vocabulary.
    """

    ids, X, Y, P = data.as_tensors_with_issue_targets(issue_max_len=cfg.max_out_len)

    n = len(ids)
    if n < 4:
        raise RuntimeError(
            f"Need at least 4 examples for train/test split; got {n}. "
            "Increase --limit or use --demo-data."
        )

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)

    ids = [ids[i] for i in perm.tolist()]
    X = X[perm]
    Y = Y[perm]
    P = P[perm]

    split_info = {}

    if "split_info" in cfg.__dict__:
        split_info = cfg.__dict__["split_info"]

    adapt_idx = split_info.get("adapt_idx")
    final_test_idx = split_info.get("final_test_idx")

    if adapt_idx is None or final_test_idx is None:
        raise RuntimeError(
            "Checkpoint split_info must contain adapt_idx and final_test_idx. "
            "Re-run the baseline after saving the 800/100/100 split."
        )

    adapt_idx = torch.tensor(adapt_idx, dtype=torch.long)
    final_test_idx = torch.tensor(final_test_idx, dtype=torch.long)

    diff_token_ids, bad_token_ids = build_constraint_token_ids(b, data)
    pass_token_id = data.tok.sp.piece_to_id("pass")

    max_len_for_model = max(cfg.max_len_cap, X.size(1) + cfg.max_out_len)

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

    model._pass_token_id = pass_token_id

    split_data = {
        "ids": ids,
        "X": X,
        "Y": Y,
        "P": P,
        "X_train": X[adapt_idx],
        "Y_train": Y[adapt_idx],
        "P_train": P[adapt_idx],
        "X_test": X[final_test_idx],
        "Y_test": Y[final_test_idx],
        "P_test": P[final_test_idx],
        "train_size": len(adapt_idx),
        "test_size": len(final_test_idx),
        "adapt_idx": adapt_idx.tolist(),
        "final_test_idx": final_test_idx.tolist(),
        "diff_token_ids": diff_token_ids,
        "bad_token_ids": bad_token_ids,
        "max_len_for_model": max_len_for_model,
    }

    return model, split_data


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

    print(f"[Checkpoint] saved deployed joint checkpoint: {ckpt_path}")


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
    print(f"[Checkpoint] loaded deployed joint checkpoint: {ckpt_path}")


# -----------------------------------------------------------------------------
# Metrics and validation helpers.
# -----------------------------------------------------------------------------


def clone_frozen_reference(model) -> Dict[str, torch.Tensor]:
    """Clone all parameters so we can verify frozen parts did not change."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def changed_parameter_report(model, before: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    changed = []
    unchanged = []
    max_abs_delta = 0.0

    for name, p in model.named_parameters():
        if name not in before:
            changed.append(name)
            continue

        old = before[name]
        new = p.detach().cpu()

        if old.shape != new.shape:
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


def build_generated_gist_plus_raw_context(b, model, tok, X, cfg, device: str):
    """Build implementation input from generated issue gist plus raw context.

    This mirrors the role-decomposition view:
        issue -> generated/validated issue gist -> patch
    """

    issue_ctx, _ = b._generate_issue_gist_context(
        model,
        tok,
        X.to(device),
        issue_max_len=min(cfg.max_out_len, 256),
    )

    return b._build_gist_plus_raw_context(
        tok,
        issue_ctx,
        X,
        max_in_len=cfg.max_in_len,
    )[:, : cfg.max_in_len]


def build_gold_gist_plus_raw_context(b, tok, X, P, cfg):
    """Build implementation input from gold issue gist plus raw context.

    This is useful for evaluating the implementation agent without making the
    result depend on the current quality of generated issue gists.
    """

    rows = []
    for row in P:
        txt = tok.decode(
            [
                t.item()
                for t in row
                if t.item() not in (tok.pad, tok.bos, tok.eos)
            ]
        )

        wrapped = f"<ISSUE_GIST>\n{txt}\n</ISSUE_GIST>"

        rows.append(
            torch.tensor(
                tok.sp.encode(wrapped, out_type=int),
                dtype=torch.long,
            )
        )

    issue_ctx = b.pad_sequence(
        rows,
        batch_first=True,
        padding_value=tok.pad,
    )

    return b._build_gist_plus_raw_context(
        tok,
        issue_ctx,
        X,
        max_in_len=cfg.max_in_len,
    )[:, : cfg.max_in_len]


def evaluate_snapshot(b, model, data, tensors, cfg, device: str) -> Dict[str, float]:
    """Evaluate issue role, raw code role, gold-gist code role, and generated-gist code role."""

    X_test = tensors["X_test"]
    Y_test = tensors["Y_test"]
    P_test = tensors["P_test"]

    issue_ce, issue_acc = b._eval_issue_ce_acc(model, X_test, P_test, device=device)

    # Direct code evaluation without gist context.
    code_raw_ce, code_raw_acc = b._eval_code_ce_acc(model, X_test, Y_test, device=device)

    # Implementation-agent evaluation using gold issue gists.
    X_gold_gist = build_gold_gist_plus_raw_context(
        b,
        data.tok,
        X_test,
        P_test,
        cfg,
    )
    code_gold_gist_ce, code_gold_gist_acc = b._eval_code_ce_acc(
        model,
        X_gold_gist,
        Y_test,
        device=device,
    )

    # Pipeline-style evaluation using generated issue gists.
    X_generated_gist = build_generated_gist_plus_raw_context(
        b,
        model,
        data.tok,
        X_test,
        cfg,
        device,
    )
    code_generated_gist_ce, code_generated_gist_acc = b._eval_code_ce_acc(
        model,
        X_generated_gist,
        Y_test,
        device=device,
    )

    return {
        "issue_ce": issue_ce,
        "issue_tok_acc": issue_acc,
        "code_raw_ce": code_raw_ce,
        "code_raw_tok_acc": code_raw_acc,
        "code_gold_gist_ce": code_gold_gist_ce,
        "code_gold_gist_tok_acc": code_gold_gist_acc,
        "code_generated_gist_ce": code_generated_gist_ce,
        "code_generated_gist_tok_acc": code_generated_gist_acc,
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
    for name in locality["non_target_changed_tensors"][:30]:
        print(f"  - {name}")
    if len(locality["non_target_changed_tensors"]) > 30:
        print("  ...")

    if strict:
        raise RuntimeError("Parameter locality failed under --strict-locality.")


# -----------------------------------------------------------------------------
# Main adaptation procedure.
# -----------------------------------------------------------------------------


def run(args) -> None:

    Path(args.checkpoint).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(args.out_dir).mkdir(
        parents=True,
        exist_ok=True
    )
    b = load_baseline_module(args.baseline)
    cfg = make_cfg_from_args(b, args)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    ckpt = read_checkpoint_if_exists(str(checkpoint_path), device)

    if ckpt is None:
        raise FileNotFoundError(
            f"Deployment checkpoint not found: {checkpoint_path}\n"
            "Run the baseline SWE-bench experiment first so it saves the deployed checkpoint."
        )

    if "split_info" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain split_info. "
            "Re-run the baseline after adding adapt_idx and final_test_idx to split_info."
        )

    cfg.split_info = ckpt["split_info"]

    print("[Setup] Building data")
    data = build_data(b, cfg)

    if ckpt is not None:
        restored = restore_tokenizer_from_proto(
            data.tok,
            ckpt.get("tokenizer_model_proto"),
        )
        if restored:
            print("[Tokenizer] Restored tokenizer from deployed checkpoint.")
        else:
            print(
                "[Tokenizer][WARNING] Checkpoint has no tokenizer proto or restore failed. "
                "The script will use a newly trained tokenizer; this is safe only if it "
                "matches the checkpoint tokenizer."
            )

    print("[Setup] Building deterministic split and model shell")
    model, tensors = build_splits_and_model_from_data(b, data, cfg)
    model.to(device)

    print(
        f"[Data] train={tensors['train_size']} | test={tensors['test_size']} | "
        f"vocab={data.tok.vocab_size} | max_len={tensors['max_len_for_model']}"
    )

    load_checkpoint_into_model(str(checkpoint_path), model, ckpt)

    # ------------------------------------------------------------------
    # Treat loaded/saved checkpoint as deployed model.
    # ------------------------------------------------------------------

    print("\n[Deployment] Joint checkpoint is fixed and evaluated before adaptation.")
    before = evaluate_snapshot(b, model, data, tensors, cfg, device)
    print_metrics_block("Before post-deployment adaptation", before)

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
            tensors["X_test"],
            output_dir=str(before_dir),
            sample_prefix=f"before-{args.target_agent}",
            issue_max_len=min(cfg.max_out_len, 256),
            out_max_len=args.decode_max_len,
            max_in_len=cfg.max_in_len,
            n_samples=args.generate_samples,
            max_repair_attempts=args.max_repair_attempts,
            device=device,
        )

    if args.target_agent == "issue":
        target_agent_id = b.AGENT_ISSUE_ANALYSIS
        target_label = "ISSUE_ANALYSIS"
    elif args.target_agent == "code":
        target_agent_id = b.AGENT_CODE_GENERATION
        target_label = "CODE_GENERATION"
    else:
        raise ValueError("--target-agent must be one of: issue, code")

    # Explicitly set the same freeze pattern used by baseline fine_tune_static so
    # adaptation cost can be reported before training begins.
    b._set_ft_requires_grad(
        model,
        user_id=target_agent_id,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
    )

    ac_stats = b.compute_trainable_stats(model)

    print_metrics_block(
        f"Adaptation cost setup for target agent: {target_label}",
        {
            "target_agent_id": target_agent_id,
            "trainable_params": ac_stats["trainable"],
            "total_params": ac_stats["total"],
            "AC_trainable_ratio": ac_stats["ratio"],
            "backbone_frozen": True,
            "non_target_agent_frozen": True,
        },
    )

    parameter_snapshot = clone_frozen_reference(model)

    # ------------------------------------------------------------------
    # Post-deployment adaptation.
    # ------------------------------------------------------------------

    print("\n[Post-Deployment Adaptation]")
    print(
        f"Updating only target agent: {target_label}. "
        "Shared backbone and non-target agent remain frozen."
    )

    b.fine_tune_static(
        model,
        tensors["X_train"],
        tensors["Y_train"],
        user_id=target_agent_id,
        diff_token_ids=tensors["diff_token_ids"],
        bad_token_ids=tensors["bad_token_ids"],
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=args.weight_decay,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        device=device,
        tok=data.tok,
        P=tensors["P_train"],
        gist_ctx_fn=None,
        max_in_len=cfg.max_in_len,
        use_concat_first_epoch=False,
        patience=args.patience,
    )

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
    # Optional small generation sample after adaptation.
    # ------------------------------------------------------------------

    if args.generate_samples > 0:
        output_dir = (
            Path(args.out_dir)
            / "after_adaptation"
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        b.generate_validated_samples(
            model,
            data.tok,
            tensors["X_test"],
            output_dir=str(output_dir),
            sample_prefix=f"postdeploy-{args.target_agent}",
            issue_max_len=min(cfg.max_out_len, 256),
            out_max_len=args.decode_max_len,
            max_in_len=cfg.max_in_len,
            n_samples=args.generate_samples,
            max_repair_attempts=args.max_repair_attempts,
            device=device,
        )

    # ------------------------------------------------------------------
    # Optionally save adapted model.
    # ------------------------------------------------------------------

    if args.save_adapted_checkpoint:
        adapted_path = Path(args.save_adapted_checkpoint).expanduser().resolve()
        save_checkpoint(
            str(adapted_path),
            model,
            cfg,
            data,
            meta={
                "meaning": "post-deployment adapted checkpoint",
                "target_agent": args.target_agent,
                "target_agent_id": target_agent_id,
                "source_checkpoint": str(checkpoint_path),
            },
        )

    # ------------------------------------------------------------------
    # Save metrics for paper tables.
    # ------------------------------------------------------------------

    metrics_path = Path(args.metrics_out).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "target_agent": args.target_agent,
        "target_agent_id": target_agent_id,
        "config": dataclass_to_dict(cfg),
        "before": before,
        "after": after,
        "delta": deltas,
        "adaptation_cost": {
            "trainable_params": ac_stats["trainable"],
            "total_params": ac_stats["total"],
            "AC_trainable_ratio": ac_stats["ratio"],
        },
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
    p = argparse.ArgumentParser(
        description="Validate agent-specific post-deployment adaptation."
    )

    p.add_argument(
        "--baseline",
        default="agentic-transformer-v17-swe-bench.py",
        help="Path to agentic-transformer-v17-swe-bench.py",
    )
    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
    )

    p.add_argument(
        "--target-agent",
        choices=["issue", "code"],
        default="issue",
        help="Agent to adapt after deployment.",
    )

    # Data/model knobs. Defaults mirror the baseline but are easy to reduce for
    # CPU smoke tests.
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--demo-data", action="store_true")
    p.add_argument("--max-in-len", type=int, default=256)
    p.add_argument("--max-out-len", type=int, default=320)
    p.add_argument("--spm-vocab", type=int, default=2048)

    p.add_argument("--n-agents", type=int, default=2)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers-enc", type=int, default=3)
    p.add_argument("--n-layers-dec", type=int, default=3)
    p.add_argument("--max-len-cap", type=int, default=640)

    # Training knobs.
    p.add_argument("--joint-epochs", type=int, default=25)
    p.add_argument("--adapt-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--joint-lr", type=float, default=2e-4)
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument(
        "--freeze-adapters",
        action="store_true",
        help="If set, only role heads are adapted; adapters remain frozen.",
    )

    # Evaluation/output.
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_POSTDEPLOY_DIR)
    )

    p.add_argument(
        "--metrics-out",
        default=str(DEFAULT_METRICS)
    )
    p.add_argument("--generate-samples", type=int, default=10)
    p.add_argument("--decode-max-len", type=int, default=96)
    p.add_argument("--max-repair-attempts", type=int, default=3)

    # Validation/output controls.
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
