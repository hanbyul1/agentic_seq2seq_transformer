# ============================================================
# MoE seq2seq baseline — HumanEval
# CPU-only baseline aligned with the MoE joint-training algorithm
#
# Architecture:
#   Input -> shared encoder -> router -> top-k experts
#   -> weighted expert combination -> shared decoder output head
#
# Training objective:
#   SeqCELoss + load_balance_weight * LoadBalanceLoss
#
# No agentic mechanisms are included:
#   - no specification stage
#   - no repair loop
#   - no validation gate
#   - no post-deployment expert-only adaptation
#   - no structural constraint loss
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import os
import re
import ast
import json
import random
import tempfile

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence


# ============================================================
# Repro
# ============================================================

DEVICE = "cpu"

UNK = 0
PAD = 1
BOS = 2
EOS = 3


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Config — MoE joint-training baseline only
# ============================================================

@dataclass
class Config:
    seed: int = 42

    # data
    limit: int = 164
    max_in_len: int = 512
    max_out_len: int = 384
    spm_vocab: int = 4096
    decode_max_len: int = 512

    # model
    model_dim: int = 384
    n_heads: int = 4
    n_layers_enc: int = 4
    n_layers_dec: int = 4
    max_len_cap: int = 640

    # MoE routing
    n_experts: int = 2
    top_k: int = 1
    expert_hidden_dim: int = 124
    load_balance_weight: float = 0.01

    # joint training
    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-4

    # output
    out_dir: str = "outputs/moe_humaneval"


CFG = Config()


# ============================================================
# SentencePiece tokenizer
# ============================================================

try:
    import sentencepiece as spm
    HAVE_SPM = True
except ImportError:
    HAVE_SPM = False


class SubwordTokenizer:
    def __init__(
        self,
        texts: Sequence[str],
        vocab_size: int = 4096,
    ):
        if not HAVE_SPM:
            raise RuntimeError("Install sentencepiece: pip install sentencepiece")

        import contextlib

        @contextlib.contextmanager
        def _silence_cpp_stdio():
            try:
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                saved_out, saved_err = os.dup(1), os.dup(2)
                try:
                    os.dup2(devnull_fd, 1)
                    os.dup2(devnull_fd, 2)
                    yield
                finally:
                    os.dup2(saved_out, 1)
                    os.dup2(saved_err, 2)
                    os.close(saved_out)
                    os.close(saved_err)
                    os.close(devnull_fd)
            except Exception:
                yield

        with tempfile.TemporaryDirectory() as tmpd:
            corpus = os.path.join(tmpd, "spm_corpus.txt")

            with open(corpus, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(str(t).replace("\r", " ") + "\n")

            model_prefix = os.path.join(tmpd, "spm_model")

            cmd = (
                f"--input={corpus} "
                f"--model_prefix={model_prefix} "
                f"--vocab_size={int(vocab_size)} "
                f"--character_coverage=0.9995 "
                f"--model_type=unigram "
                f"--pad_id=1 --unk_id=0 --bos_id=2 --eos_id=3 "
                f"--hard_vocab_limit=false "
                f"--byte_fallback=true "
                f"--split_by_whitespace=false "
                f"--input_sentence_size=0 "
                f"--max_sentence_length=20000"
            )

            with _silence_cpp_stdio():
                spm.SentencePieceTrainer.Train(cmd)

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_prefix}.model")

        self.vocab_size = self.sp.get_piece_size()
        self.pad_idx = PAD
        self.unk_idx = UNK
        self.bos_idx = BOS
        self.eos_idx = EOS

    def encode(
        self,
        text: str,
        *,
        add_bos_eos: bool,
        max_len: int,
    ) -> torch.Tensor:
        ids = self.sp.encode(str(text), out_type=int)

        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]

        ids = ids[:max_len] or [self.unk_idx]

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return self.sp.decode(ids)

    @property
    def pad(self):
        return self.pad_idx

    @property
    def bos(self):
        return self.bos_idx

    @property
    def eos(self):
        return self.eos_idx


# ============================================================
# HumanEval data
# ============================================================

try:
    from datasets import load_dataset
    HAVE_HF = True
except Exception:
    HAVE_HF = False


def extract_prompt_anchor(prompt: str) -> str:
    txt = str(prompt)

    m = re.search(
        r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:",
        txt,
    )

    if m:
        return m.group(0).strip()

    return "def generated_function(*args):"


def build_full_humaneval_solution(prompt: str, solution: str) -> str:
    signature = extract_prompt_anchor(prompt)

    body = solution.strip("\n")

    if body.lstrip().startswith("def "):
        return body.strip()

    body_lines = [
        ln.rstrip()
        for ln in body.splitlines()
        if ln.strip()
    ]

    if not body_lines:
        body_lines = ["pass"]

    return (
        signature
        + "\n"
        + "\n".join(
            ln if ln.startswith("    ") else "    " + ln
            for ln in body_lines
        )
    )


class HumanEvalData:
    def __init__(
        self,
        limit: Optional[int] = 164,
        max_in_len: int = 512,
        max_out_len: int = 384,
        spm_vocab_size: int = 4096,
    ):
        if not HAVE_HF:
            raise RuntimeError("Install datasets: pip install datasets")

        print("[Data] Load HumanEval…", flush=True)

        ds = load_dataset("openai_humaneval", split="test")

        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        self.samples: List[Tuple[str, str, str]] = []

        for i, ex in enumerate(ds):
            iid = f"he-{i}"
            prompt = str(ex["prompt"])
            solution = build_full_humaneval_solution(
                prompt,
                str(ex["canonical_solution"]),
            )

            self.samples.append(
                (
                    iid,
                    prompt,
                    solution.strip(),
                )
            )

        texts = (
            [x for _, x, _ in self.samples]
            + [y for _, _, y in self.samples]
        )

        self.tok = SubwordTokenizer(
            texts,
            vocab_size=spm_vocab_size,
        )

        self.max_in_len = max_in_len
        self.max_out_len = max_out_len

    def as_tensors(self) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        ids = []
        xs = []
        ys = []

        for iid, x, y in self.samples:
            ids.append(iid)

            xs.append(
                self.tok.encode(
                    x,
                    add_bos_eos=False,
                    max_len=self.max_in_len,
                )
            )

            ys.append(
                self.tok.encode(
                    y,
                    add_bos_eos=True,
                    max_len=self.max_out_len,
                )
            )

        X = pad_sequence(
            xs,
            batch_first=True,
            padding_value=self.tok.pad,
        )

        Y = pad_sequence(
            ys,
            batch_first=True,
            padding_value=self.tok.pad,
        )

        return ids, X, Y


# ============================================================
# Model blocks
# ============================================================

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        pad_idx: int,
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.tok_embedding = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=pad_idx,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, model_dim) * 0.01
        )

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, x: torch.Tensor):
        B, T = x.shape

        mask = x == self.pad_idx

        h = (
            self.tok_embedding(x)
            + self.pos_embedding[:, :T, :]
        )

        mem = self.encoder(
            h,
            src_key_padding_mask=mask,
        )

        valid = (~mask).float()

        pooled = (
            mem * valid.unsqueeze(-1)
        ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        return mem, pooled, mask


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        pad_idx: int,
        tok_embedding: nn.Embedding,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.tok_embedding = tok_embedding

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, model_dim) * 0.01
        )

        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

    def _subsequent_mask(self, L: int, device):
        return torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(
        self,
        y_in: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ):
        B, T = y_in.shape

        y_emb = (
            self.tok_embedding(y_in)
            + self.pos_embedding[:, :T, :]
        )

        tgt_mask = self._subsequent_mask(
            T,
            y_in.device,
        )

        tgt_key_padding_mask = y_in == self.pad_idx

        return self.decoder(
            y_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )


class Expert(nn.Module):
    def __init__(
        self,
        model_dim: int,
        expert_hidden_dim: int = 124,
    ):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, expert_hidden_dim),
            nn.GELU(),
            nn.Linear(expert_hidden_dim, model_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + self.ffn(states)


class MoERouter(nn.Module):
    def __init__(
        self,
        model_dim: int,
        n_experts: int,
        top_k: int = 1,
    ):
        super().__init__()

        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts")

        self.n_experts = n_experts
        self.top_k = top_k

        self.router = nn.Linear(
            model_dim,
            n_experts,
        )

    def forward(self, pooled: torch.Tensor):
        logits = self.router(pooled)

        dense_probs = torch.softmax(
            logits,
            dim=-1,
        )

        if self.top_k < self.n_experts:
            top_vals, top_idx = torch.topk(
                dense_probs,
                k=self.top_k,
                dim=-1,
            )

            sparse_probs = torch.zeros_like(dense_probs)
            sparse_probs.scatter_(
                1,
                top_idx,
                top_vals,
            )

            sparse_probs = sparse_probs / sparse_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        else:
            sparse_probs = dense_probs

        return sparse_probs, dense_probs, logits


class MoETransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_experts: int,
        top_k: int,
        model_dim: int,
        n_heads: int,
        n_layers_enc: int,
        n_layers_dec: int,
        max_len: int,
        pad_idx: int,
        expert_hidden_dim: int,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers_enc,
            max_len=max_len,
            pad_idx=pad_idx,
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers_dec,
            max_len=max_len,
            pad_idx=pad_idx,
            tok_embedding=self.encoder.tok_embedding,
        )

        self.experts = nn.ModuleList([
            Expert(
                model_dim=model_dim,
                expert_hidden_dim=expert_hidden_dim,
            )
            for _ in range(n_experts)
        ])

        self.router = MoERouter(
            model_dim=model_dim,
            n_experts=n_experts,
            top_k=top_k,
        )

        self.lm_head = nn.Linear(
            model_dim,
            vocab_size,
        )

        self.pad_idx = pad_idx
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(
        self,
        x: torch.Tensor,
        y_in: torch.Tensor,
    ):
        mem, pooled, src_mask = self.encoder(x)

        dec_states = self.decoder(
            y_in,
            mem,
            src_mask,
        )

        sparse_weights, dense_weights, router_logits = self.router(pooled)

        expert_outputs = []

        for expert in self.experts:
            expert_outputs.append(
                expert(dec_states)
            )

        expert_outputs = torch.stack(
            expert_outputs,
            dim=1,
        )

        # [B, E, 1, 1]
        mix_weights = sparse_weights.unsqueeze(-1).unsqueeze(-1)

        mixed = (
            expert_outputs * mix_weights
        ).sum(dim=1)

        logits = self.lm_head(mixed)

        return logits, sparse_weights, dense_weights, router_logits


# ============================================================
# Losses / metrics
# ============================================================

class SeqCELoss(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        B, T, V = logits.shape

        return self.ce(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
        )


class LoadBalanceLoss(nn.Module):
    def forward(self, router_probs: torch.Tensor):
        # router_probs is dense routing probability G_b before top-k masking.
        # This follows the algorithmic role of encouraging expert utilization.
        expert_usage = router_probs.mean(dim=0)

        target = torch.full_like(
            expert_usage,
            1.0 / expert_usage.numel(),
        )

        return F.mse_loss(
            expert_usage,
            target,
        )


def shift_targets(y: torch.Tensor):
    return y[:, :-1], y[:, 1:]


@torch.no_grad()
def token_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int,
):
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx

    correct = (
        (preds == targets) & mask
    ).sum().item()

    total = mask.sum().item()

    return correct, total


@torch.no_grad()
def validate_python_syntax(text: str) -> bool:
    code = text.strip()

    if not code:
        return False

    try:
        ast.parse(code)
        return True
    except Exception:
        return False


def validate_code_structure(text: str) -> bool:
    code = text.strip()

    if len(code) < 8:
        return False

    return (
        ("def " in code or "class " in code or "return " in code or "import " in code)
        and "\n" in code
    )


def tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


def pairwise_similarity(samples: List[str]) -> float:
    if len(samples) <= 1:
        return 1.0

    scores = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a = set(tokenize_text(samples[i]))
            b = set(tokenize_text(samples[j]))

            if not a or not b:
                continue

            scores.append(
                len(a & b) / max(len(a | b), 1)
            )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# ============================================================
# Efficiency reporting
# ============================================================

def set_trainable_moe_joint(model: MoETransformerSeq2Seq):
    for p in model.parameters():
        p.requires_grad = True


def compute_efficiency_stats(model: MoETransformerSeq2Seq):
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    per_expert_params = [
        sum(p.numel() for p in expert.parameters())
        for expert in model.experts
    ]

    active_expert_params = sum(
        per_expert_params[:model.top_k]
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / max(total_params, 1),
        "per_expert_params": per_expert_params,
        "active_expert_params": active_expert_params,
        "active_parameter_ratio": active_expert_params / max(total_params, 1),
    }


def print_efficiency_stats(
    model: MoETransformerSeq2Seq,
    *,
    stage_name: str,
):
    stats = compute_efficiency_stats(model)

    print("\n------------------------------------------------------------")
    print(f"[MoE][Efficiency][{stage_name}]")
    print("------------------------------------------------------------\n")

    print(f"total_params={stats['total_params']}")
    print(f"trainable_params={stats['trainable_params']}")
    print(f"trainable_ratio={stats['trainable_ratio']:.6f}")
    print(f"active_expert_params={stats['active_expert_params']}")
    print(f"active_parameter_ratio={stats['active_parameter_ratio']:.6f}")

    for i, n in enumerate(stats["per_expert_params"]):
        print(f"expert_{i}_params={n}")


# ============================================================
# Training / eval
# ============================================================

def train_moe_supervised(
    model: MoETransformerSeq2Seq,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    load_balance_weight: float,
    device: str = DEVICE,
):
    model.to(device)

    set_trainable_moe_joint(model)

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = optim.Adam(
        params,
        lr=lr,
    )

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    balance_fn = LoadBalanceLoss()

    N = X_train.size(0)

    for ep in range(1, epochs + 1):
        model.train()

        loss_sum = 0.0
        seq_sum = 0.0
        balance_sum = 0.0

        correct = 0
        total = 0

        route_mass = torch.zeros(
            model.n_experts,
            dtype=torch.float32,
        )

        active_experts_sum = 0.0
        batches = 0

        for i in range(0, N, batch_size):
            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)

            y_in, y_tgt = shift_targets(yb)

            logits, sparse_weights, dense_weights, _router_logits = model(
                xb,
                y_in,
            )

            seq_loss = loss_fn(
                logits,
                y_tgt,
            )

            balance_loss = balance_fn(
                dense_weights,
            )

            loss = (
                seq_loss
                + load_balance_weight * balance_loss
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                params,
                1.0,
            )
            opt.step()

            with torch.no_grad():
                c, t = token_accuracy(
                    logits,
                    y_tgt,
                    model.pad_idx,
                )

                correct += c
                total += t

                route_mass += sparse_weights.detach().cpu().sum(dim=0)

                active_experts_sum += float(
                    (sparse_weights.detach().cpu() > 0).float().sum(dim=1).mean().item()
                )
                batches += 1

            loss_sum += float(loss.detach()) * xb.size(0)
            seq_sum += float(seq_loss.detach()) * xb.size(0)
            balance_sum += float(balance_loss.detach()) * xb.size(0)

        route_dist = route_mass / route_mass.sum().clamp_min(1.0)
        active_experts = active_experts_sum / max(batches, 1)
        active_ratio = active_experts / max(model.n_experts, 1)

        print(
            f"[MoE][Training][Epoch {ep}] IMPLEMENTATION: "
            f"CE={loss_sum/float(N):.3f} | "
            f"tok_acc={correct/max(total,1):.3f}"
        )
        print(
            f"balance={balance_sum/float(N):.4f} | "
            f"active_experts={active_experts:.2f}/{model.n_experts} | "
            f"active_ratio={active_ratio:.3f} | "
            f"route={route_dist.tolist()}"
        )


@torch.no_grad()
def eval_moe_ce_acc(
    model: MoETransformerSeq2Seq,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    device: str = DEVICE,
):
    model.to(device)
    model.eval()

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    y_in, y_tgt = shift_targets(
        Y.to(device)
    )

    logits, sparse_weights, _dense_weights, _ = model(
        X.to(device),
        y_in,
    )

    ce = float(
        loss_fn(
            logits,
            y_tgt,
        ).item()
    )

    correct, total = token_accuracy(
        logits,
        y_tgt,
        model.pad_idx,
    )

    acc = correct / max(total, 1)

    route_dist = (
        sparse_weights.detach()
        .cpu()
        .sum(dim=0)
    )

    route_dist = (
        route_dist
        / route_dist.sum().clamp_min(1.0)
    )

    print("\n------------------------------------------------------------")
    print("[MoE][Eval][PIPELINE-LIFT]")
    print("------------------------------------------------------------\n")
    print("Single-stage architecture (no intermediate specification stage)\n")
    print(f"IMPLEMENTATION CE={ce:.3f}")
    print(f"IMPLEMENTATION tok_acc={acc:.3f}")

    for i, r in enumerate(route_dist.tolist()):
        print(f"expert_{i}_route_share={r:.6f}")

    return ce, acc


# ============================================================
# Generation / inference
# ============================================================

@torch.no_grad()
def generate_moe(
    model: MoETransformerSeq2Seq,
    X: torch.Tensor,
    *,
    max_len: int,
    temperature: float = 1.0,
):
    model.eval()

    B = X.size(0)

    ys = torch.full(
        (B, 1),
        BOS,
        dtype=torch.long,
        device=X.device,
    )

    finished = torch.zeros(
        B,
        dtype=torch.bool,
        device=X.device,
    )

    for _ in range(1, max_len):
        logits, _sparse_weights, _dense_weights, _ = model(
            X,
            ys,
        )

        step_logits = logits[:, -1, :]

        step_logits[:, UNK] = float("-inf")
        step_logits[:, PAD] = float("-inf")
        step_logits[:, BOS] = float("-inf")

        if temperature != 1.0:
            step_logits = step_logits / max(temperature, 1e-8)

        next_tok = torch.argmax(
            step_logits,
            dim=-1,
            keepdim=True,
        )

        ys = torch.cat(
            [ys, next_tok],
            dim=1,
        )

        finished |= next_tok.squeeze(1) == EOS

        if finished.all():
            break

    return ys


@torch.no_grad()
def generate_samples(
    model: MoETransformerSeq2Seq,
    tok: SubwordTokenizer,
    ids: List[str],
    X: torch.Tensor,
    *,
    output_dir: str,
    sample_prefix: str,
    out_max_len: int,
    n_samples: int = 10,
    device: str = DEVICE,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    model.to(device)
    model.eval()

    n_samples = min(
        n_samples,
        X.size(0),
    )

    results = []
    predictions = []
    generated_outputs = []

    for i in range(n_samples):
        instance_id = ids[i]
        x = X[i:i + 1].to(device)

        out = generate_moe(
            model,
            x,
            max_len=out_max_len,
        )

        raw_txt = tok.decode([
            int(t)
            for t in out[0].tolist()
            if int(t) not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        raw_txt = raw_txt.strip()
        generated_outputs.append(raw_txt)

        syntax_valid = validate_python_syntax(raw_txt)
        structure_valid = validate_code_structure(raw_txt)
        final_valid = syntax_valid and structure_valid

        results.append({
            "patch_structure_validity": structure_valid,
            "python_syntax_validity": syntax_valid,
            "generation_validity": final_valid,
            "accepted_generation": final_valid,
        })

        predictions.append({
            "instance_id": instance_id,
            "model_name_or_path": "MoETransformerSeq2Seq",
            "model_patch": raw_txt,
        })

        print(f"\n=== Example {i + 1} ===\n")
        print("[IMPLEMENTATION]")
        print(raw_txt)

        out_path = os.path.join(
            output_dir,
            f"{sample_prefix}-sample{i + 1}.txt",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"PATCH_STRUCTURE_VALID: {structure_valid}\n")
            f.write(f"PYTHON_SYNTAX_VALID: {syntax_valid}\n")
            f.write(f"FINAL_ACCEPTED: {final_valid}\n\n")
            f.write("===== RAW GENERATED CODE =====\n")
            f.write(raw_txt)
            f.write("\n")

        print(
            f"[Inference] saved {out_path} "
            f"(valid={final_valid})"
        )

    patch_structure_validity = (
        sum(r["patch_structure_validity"] for r in results)
        / max(len(results), 1)
    )

    python_syntax_validity = (
        sum(r["python_syntax_validity"] for r in results)
        / max(len(results), 1)
    )

    generation_validity_rate = (
        sum(r["generation_validity"] for r in results)
        / max(len(results), 1)
    )

    accepted_generation_rate = (
        sum(r["accepted_generation"] for r in results)
        / max(len(results), 1)
    )

    similarity = pairwise_similarity(generated_outputs)
    unique_outputs = len(set(generated_outputs))

    print("\n------------------------------------------------------------")
    print("[MoE][Testing][OUTPUT-VALIDITY]")
    print("------------------------------------------------------------\n")

    print(f"patch_structure_validity={patch_structure_validity:.4f}")
    print(f"python_syntax_validity={python_syntax_validity:.4f}")
    print(f"generation_validity_rate={generation_validity_rate:.4f}")
    print(f"accepted_generation_rate={accepted_generation_rate:.4f}")

    print(
        f"\n[Inference Summary] "
        f"valid={sum(r['accepted_generation'] for r in results)}/{n_samples} "
        f"({100.0 * accepted_generation_rate:.2f}%) "
        f"| generation_validity_rate={generation_validity_rate:.4f} "
        f"| unique_outputs={unique_outputs}/{n_samples} "
        f"| pairwise_similarity={similarity:.4f}"
    )

    pred_path = os.path.join(
        output_dir,
        "predictions.jsonl",
    )

    with open(pred_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row) + "\n")

    print(f"[MoE][JSONL Export] saved -> {pred_path}")


# ============================================================
# Orchestration
# ============================================================

def run_all(cfg: Config = CFG):
    set_seed(cfg.seed)

    print("\n" + "=" * 60)
    print("Round 1")
    print("=" * 60 + "\n")

    print("MoE Baseline: HumanEval")

    data = HumanEvalData(
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
    )

    ids, X, Y = data.as_tensors()

    for i in range(min(3, len(data.samples))):
        print("\n====================")
        print("PROMPT")
        print(data.samples[i][1][:1000])

        print("\nTARGET")
        print(
            data.tok.decode([
                int(t)
                for t in Y[i].tolist()
                if int(t) not in (
                    data.tok.pad,
                    data.tok.bos,
                    data.tok.eos,
                )
            ])[:1000]
        )

    N = len(ids)

    g = torch.Generator().manual_seed(cfg.seed)

    perm = torch.randperm(
        N,
        generator=g,
    )

    ids = [
        ids[i]
        for i in perm.tolist()
    ]

    X = X[perm]
    Y = Y[perm]

    split = int(N * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    Y_train = Y[:split]
    Y_test = Y[split:]

    print(
        f"[Info] Train: {split} pairs, "
        f"Test: {N - split} pairs"
    )

    max_len_for_model = max(
        cfg.max_len_cap,
        cfg.max_in_len + cfg.max_out_len + 8,
    )

    model = MoETransformerSeq2Seq(
        vocab_size=data.tok.vocab_size,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_heads,
        n_layers_enc=cfg.n_layers_enc,
        n_layers_dec=cfg.n_layers_dec,
        max_len=max_len_for_model,
        pad_idx=data.tok.pad,
        expert_hidden_dim=cfg.expert_hidden_dim,
    )

    print(
        f"[MoE][Model] experts={cfg.n_experts} | "
        f"top_k={cfg.top_k} | "
        f"model_dim={cfg.model_dim}"
    )

    set_trainable_moe_joint(model)
    print_efficiency_stats(
        model,
        stage_name="INITIAL",
    )

    print(
        "\n[MoE][Training] "
        "Joint Expert-Routed IMPLEMENTATION"
    )

    train_moe_supervised(
        model,
        X_train,
        Y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        load_balance_weight=cfg.load_balance_weight,
        device=DEVICE,
    )

    print_efficiency_stats(
        model,
        stage_name="AFTER-JOINT-TRAINING",
    )

    print("\n[MoE][Eval After Joint Training]")

    ce, acc = eval_moe_ce_acc(
        model,
        X_test,
        Y_test,
        device=DEVICE,
    )

    print(
        f"\n[MoE][Testing][IMPLEMENTATION] "
        f"CE={ce:.3f} | tok_acc={acc:.3f} | N={X_test.size(0)}"
    )

    print("\n[MoE][Testing][SPEC-STATS]")
    print("sampled=0")
    print("avg_tokens=0.00")
    print("avg_lines=0.00")
    print("field_coverage=0.00")

    print("\n[MoE][Training] Stage 2A: N/A")
    print("No dedicated specification expert exists in standard MoE.")
    print("[MoE][Eval][SPEC@Before FT] N/A")
    print("[MoE][Eval][SPEC@After FT] N/A")

    print("\n[MoE][Training] Stage 2B: N/A")
    print("No post-deployment expert specialization performed.")
    print(
        f"[MoE][Eval][IMPLEMENTATION@Before FT] "
        f"CE={ce:.3f} | tok_acc={acc:.3f}"
    )
    print(
        f"[MoE][Eval][IMPLEMENTATION@After FT] "
        f"CE={ce:.3f} | tok_acc={acc:.3f} | "
        f"ΔCE=+0.000 (0.00%) | Δacc=+0.000 (0.00%)"
    )

    stats = compute_efficiency_stats(model)
    print(
        f"[MoE][Adaptation-Cost][IMPLEMENTATION] "
        f"trainable_params={stats['trainable_params']} | "
        f"total_params={stats['total_params']} | "
        f"trainable_ratio={stats['trainable_ratio']:.6f}"
    )

    print("\n[Inference] Generating samples")

    generate_samples(
        model,
        data.tok,
        ids[split:],
        X_test,
        output_dir=cfg.out_dir,
        sample_prefix="moe-humaneval",
        out_max_len=cfg.decode_max_len,
        n_samples=10,
        device=DEVICE,
    )

    return model, data, (ids, X, Y)


if __name__ == "__main__":
    print("[MoE] Starting HumanEval baseline...", flush=True)
    model, data, tensors = run_all(CFG)
