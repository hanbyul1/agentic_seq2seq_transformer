# ============================================================
# MoE seq2seq baseline — SWE-bench
# CPU-only version
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import os
import re
import ast
import random
import tempfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


# ============================================================
# Repro
# ============================================================

DEVICE = "cpu"

PAD = 1
UNK = 0
BOS = 2
EOS = 3


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 42

    # data
    limit: int = 1000
    max_in_len: int = 256
    max_out_len: int = 320
    spm_vocab: int = 2048
    demo_data: bool = False

    # model
    model_dim: int = 256
    n_heads: int = 4
    n_layers_enc: int = 3
    n_layers_dec: int = 3
    max_len_cap: int = 640

    # MoE
    n_experts: int = 4
    top_k: int = 2
    expert_dim: int = 512
    load_balance_lambda: float = 0.01

    # training
    epochs: int = 25
    batch_size: int = 8
    lr: float = 2e-4

    # decoding
    decode_max_len: int = 96

    n_validation_samples: int =10

    out_dir: str = "outputs/swebench_moe"

CFG = Config()


# ============================================================
# Tokenizer
# ============================================================

try:
    import sentencepiece as spm
    HAVE_SPM = True
except Exception:
    HAVE_SPM = False


class SubwordTokenizer:

    def __init__(
        self,
        texts: Sequence[str],
        vocab_size: int = 2048,
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
                saved_out = os.dup(1)
                saved_err = os.dup(2)

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
                f"--pad_id=1 "
                f"--unk_id=0 "
                f"--bos_id=2 "
                f"--eos_id=3 "
                f"--hard_vocab_limit=false "
                f"--byte_fallback=false "
                f"--split_by_whitespace=true "
                f"--input_sentence_size=0 "
                f"--max_sentence_length=20000"
            )

            with _silence_cpp_stdio():
                spm.SentencePieceTrainer.Train(cmd)

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_prefix}.model")

        self.vocab_size = self.sp.get_piece_size()
        self.pad = PAD
        self.unk = UNK
        self.bos = BOS
        self.eos = EOS

    def encode(self, text: str, add_bos_eos: bool, max_len: int) -> torch.Tensor:
        ids = self.sp.encode(str(text), out_type=int)

        if add_bos_eos:
            ids = [self.bos] + ids + [self.eos]

        ids = ids[:max_len]

        if not ids:
            ids = [self.unk]

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)


# ============================================================
# Dataset
# ============================================================

try:
    from datasets import load_dataset
    HAVE_HF = True
except Exception:
    HAVE_HF = False


class SWEText2PatchData:

    def __init__(
        self,
        *,
        split: str = "train",
        limit: Optional[int] = 5000,
        max_in_len: int = 128,
        max_out_len: int = 256,
        spm_vocab_size: int = 2048,
        demo_data: bool = False,
    ):

        self.max_in_len = max_in_len
        self.max_out_len = max_out_len
        self.samples = []

        if demo_data:
            print("[Data] DEMO synthetic dataset")

            n = int(limit or 1024)

            for i in range(n):
                x = f"Issue {i}: crash when clicking widget. Trace id {i}."
                y = (
                    "diff --git a/app.py b/app.py\n"
                    "@@\n"
                    "-raise Exception('broken')\n"
                    "+return 'fixed'\n"
                )
                self.samples.append((f"demo-{i}", x, y))

        else:
            if not HAVE_HF:
                raise RuntimeError("Install datasets: pip install datasets")

            print("[Data] Loading SWE-bench...")

            ds = load_dataset(
                "princeton-nlp/SWE-bench",
                split=split,
            )

            if limit is not None:
                ds = ds.select(range(min(limit, len(ds))))

            for ex in ds:
                iid = str(ex.get("instance_id", ""))

                title = str(ex.get("title", "")).strip()
                desc = str(ex.get("problem_statement", "")).strip()
                hints = str(ex.get("hints_text", "")).strip()

                parts = []

                if title:
                    parts.append(f"<ISSUE_TITLE>\n{title}\n</ISSUE_TITLE>")

                if desc:
                    parts.append(f"<ISSUE_DESC>\n{desc}\n</ISSUE_DESC>")

                if hints:
                    parts.append(f"<HINTS>\n{hints}\n</HINTS>")

                x = "\n".join(parts)

                y = ""

                for key in ("patch", "base_patch", "model_patch", "test_patch"):
                    if key in ex and ex[key]:
                        y = str(ex[key])
                        break

                if y.strip():
                    self.samples.append((iid, x, y))

        texts = []

        for _, x, y in self.samples:
            texts.append(x)
            texts.append(y)

        self.tok = SubwordTokenizer(
            texts,
            vocab_size=spm_vocab_size,
        )

    def as_tensors(self):
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

        X = pad_sequence(xs, batch_first=True, padding_value=self.tok.pad)
        Y = pad_sequence(ys, batch_first=True, padding_value=self.tok.pad)

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

        h = self.tok_embedding(x) + self.pos_embedding[:, :T, :]

        src_mask = x == self.pad_idx

        memory = self.encoder(
            h,
            src_key_padding_mask=src_mask,
        )

        valid = (~src_mask).float()

        pooled = (
            memory * valid.unsqueeze(-1)
        ).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        return memory, pooled, src_mask


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

    def _causal_mask(self, L: int, device):
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

        B, L = y_in.shape

        y_emb = self.tok_embedding(y_in) + self.pos_embedding[:, :L, :]

        tgt_mask = self._causal_mask(L, y_in.device)

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
        expert_dim: int,
        vocab_size: int,
    ):

        super().__init__()

        self.ffn = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, model_dim),
            nn.GELU(),
        )

        self.out = nn.Linear(model_dim, vocab_size)

    def forward(self, dec_states: torch.Tensor):
        h = self.ffn(dec_states)
        return self.out(h)


class MoERouter(nn.Module):

    def __init__(
        self,
        model_dim: int,
        n_experts: int,
    ):

        super().__init__()

        self.router = nn.Linear(model_dim, n_experts)

    def forward(self, pooled: torch.Tensor):
        return self.router(pooled)


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
        expert_dim: int,
    ):

        super().__init__()

        self.pad_idx = pad_idx
        self.n_experts = n_experts
        self.top_k = top_k

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

        self.router = MoERouter(
            model_dim=model_dim,
            n_experts=n_experts,
        )

        self.experts = nn.ModuleList([
            Expert(
                model_dim=model_dim,
                expert_dim=expert_dim,
                vocab_size=vocab_size,
            )
            for _ in range(n_experts)
        ])

    def encode(self, x):
        return self.encoder(x)

    def _topk_gates(self, router_logits):
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(
            router_probs,
            k=min(self.top_k, self.n_experts),
            dim=-1,
        )

        gates = torch.zeros_like(router_probs)
        gates.scatter_(1, topk_idx, topk_vals)

        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return gates, router_probs, topk_idx

    def forward(self, x, y_in):
        memory, pooled, src_mask = self.encode(x)

        dec_states = self.decoder(
            y_in,
            memory,
            src_mask,
        )

        router_logits = self.router(pooled)

        gates, router_probs, selected = self._topk_gates(router_logits)

        expert_logits = []

        for expert in self.experts:
            expert_logits.append(expert(dec_states))

        expert_logits = torch.stack(expert_logits, dim=1)
        # shape: [B, E, L, V]

        combined_logits = (
            expert_logits
            * gates[:, :, None, None]
        ).sum(dim=1)

        return {
            "logits": combined_logits,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "gates": gates,
            "selected": selected,
        }


# ============================================================
# Losses
# ============================================================

class SeqCELoss(nn.Module):

    def __init__(self, pad_idx: int):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=0.00,
        )

    def forward(self, logits, targets):
        B, L, V = logits.shape

        return self.ce(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
        )


class LoadBalanceLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):

        super().__init__()

        self.eps = eps

    def forward(self, router_probs, gates):
        # importance: soft probability mass assigned to each expert
        importance = router_probs.mean(dim=0)

        # load: actual top-k selected traffic per expert
        load = (gates > 0).float().mean(dim=0)

        importance_loss = (
            importance.var(unbiased=False)
            / importance.mean().clamp_min(self.eps).pow(2)
        )

        load_loss = (
            load.var(unbiased=False)
            / load.mean().clamp_min(self.eps).pow(2)
        )

        return importance_loss + load_loss


def shift_targets(y):
    return y[:, :-1], y[:, 1:]

# ============================================================
# Parameter-efficiency metrics
# ============================================================

def compute_moe_efficiency_stats(model):

    total_params = 0
    trainable_params = 0

    for p in model.parameters():

        n = p.numel()

        total_params += n

        if p.requires_grad:
            trainable_params += n

    # --------------------------------------------------------
    # Active-expert parameter estimation
    # --------------------------------------------------------

    expert_params = 0

    for expert in model.experts:

        for p in expert.parameters():
            expert_params += p.numel()

    avg_expert_params = (
        expert_params
        / max(len(model.experts), 1)
    )

    active_expert_params = (
        avg_expert_params
        * model.top_k
    )

    active_ratio = (
        active_expert_params
        / max(total_params, 1)
    )

    trainable_ratio = (
        trainable_params
        / max(total_params, 1)
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "active_expert_params": int(active_expert_params),
        "active_ratio": active_ratio,
    }

# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_ce_acc(model, X, Y, *, device=DEVICE):

    model.to(device)
    model.eval()

    loss_fn = SeqCELoss(model.pad_idx)

    y_in, y_tgt = shift_targets(Y.to(device))

    out = model(
        X.to(device),
        y_in,
    )

    logits = out["logits"]

    ce = float(loss_fn(logits, y_tgt).item())

    preds = logits.argmax(dim=-1)
    mask = y_tgt != model.pad_idx

    acc = float(
        (((preds == y_tgt) & mask).float().sum()
         / mask.float().sum().clamp_min(1.0)).item()
    )

    return ce, acc

def print_moe_efficiency_stats(model):

    stats = compute_moe_efficiency_stats(model)

    print("\n------------------------------------------------------------")
    print("[MoE][Efficiency]")
    print("------------------------------------------------------------\n")

    print(
        f"total_params="
        f"{stats['total_params']}"
    )

    print(
        f"trainable_params="
        f"{stats['trainable_params']}"
    )

    print(
        f"trainable_ratio="
        f"{stats['trainable_ratio']:.6f}"
    )

    print(
        f"active_expert_params="
        f"{stats['active_expert_params']}"
    )

    print(
        f"active_parameter_ratio="
        f"{stats['active_ratio']:.6f}"
    )

# ============================================================
# Training
# ============================================================

def train_moe(
    model,
    X_train,
    Y_train,
    *,
    epochs,
    batch_size,
    lr,
    lambda_balance,
    device=DEVICE,
):

    model.to(device)

    opt = optim.Adam(
        model.parameters(),
        lr=lr,
    )

    seq_loss_fn = SeqCELoss(model.pad_idx)
    balance_loss_fn = LoadBalanceLoss()

    N = X_train.size(0)

    for ep in range(1, epochs + 1):

        model.train()

        total_seq = 0.0
        total_bal = 0.0
        correct = 0.0
        total = 0.0

        for i in range(0, N, batch_size):

            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)

            y_in, y_tgt = shift_targets(yb)

            out = model(xb, y_in)

            logits = out["logits"]

            seq_loss = seq_loss_fn(
                logits,
                y_tgt,
            )

            balance_loss = balance_loss_fn(
                out["router_probs"],
                out["gates"],
            )

            loss = seq_loss + lambda_balance * balance_loss

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = y_tgt != model.pad_idx

                correct += (
                    ((preds == y_tgt) & mask)
                    .float()
                    .sum()
                    .item()
                )

                total += mask.float().sum().item()

            total_seq += float(seq_loss.detach()) * xb.size(0)
            total_bal += float(balance_loss.detach()) * xb.size(0)

        avg_active_experts = float(
            model.top_k
        )

        active_ratio = (
            avg_active_experts
            / max(model.n_experts, 1)
        )

        print(
            f"[MoE][Training][Epoch {ep}] "
            f"IMPLEMENTATION: "
            f"CE={total_seq / N:.3f} | "
            f"tok_acc={correct / max(total,1.0):.3f}"
        )

        print(
            f"    balance={total_bal / N:.4f} | "
            f"active_experts={avg_active_experts:.2f}/{model.n_experts} | "
            f"active_ratio={active_ratio:.3f}"
        )


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_moe(
    model,
    X,
    *,
    max_len,
    device=DEVICE,
):

    model.to(device)
    model.eval()

    X = X.to(device)

    B = X.size(0)

    ys = torch.full(
        (B, 1),
        BOS,
        dtype=torch.long,
        device=device,
    )

    for _ in range(1, max_len):

        out = model(X, ys)
        selected = out["selected"]

        if not hasattr(model, "_routing_counter"):
            model._routing_counter = 0

        model._routing_counter += selected.numel()

        next_logits = out["logits"][:, -1, :]

        next_tok = torch.argmax(
            next_logits,
            dim=-1,
            keepdim=True,
        )

        ys = torch.cat([ys, next_tok], dim=1)

        if (next_tok == EOS).all():
            break

    return ys


# ============================================================
# Simple output validity checks
# ============================================================

def extract_python_from_patch(text: str) -> str:
    lines = []

    for line in text.splitlines():

        if line.startswith(("+++", "---", "@@", "diff --git")):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])

        elif not line.startswith("-"):
            lines.append(line)

    return "\n".join(lines)


def validate_patch_structure(text: str) -> bool:
    if not text.strip():
        return False

    return (
        "diff --git" in text
        or "@@" in text
        or re.search(r"^[+-]", text, re.MULTILINE) is not None
    )


def validate_python_syntax(text: str) -> bool:
    extracted = extract_python_from_patch(text)

    if not extracted.strip():
        return False

    try:
        ast.parse(extracted)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate_validated_samples(
    model,
    tok,
    X,
    *,
    output_dir,
    sample_prefix,
    max_len,
    n_samples,
    device=DEVICE,
):

    os.makedirs(output_dir, exist_ok=True)

    n_samples = min(n_samples, X.size(0))

    patch_valid_count = 0
    syntax_valid_count = 0

    generation_valid_count = 0
    accepted_generation_count = 0

    for i in range(n_samples):

        gen_ids = generate_moe(
            model,
            X[i:i + 1],
            max_len=max_len,
            device=device,
        )

        ids = [
            t
            for t in gen_ids[0].tolist()
            if t not in (tok.pad, tok.bos, tok.eos)
        ]

        txt = tok.decode(ids)

        patch_valid = validate_patch_structure(txt)
        syntax_valid = validate_python_syntax(txt)

        patch_valid_count += int(patch_valid)
        syntax_valid_count += int(syntax_valid)

        generation_valid = patch_valid and syntax_valid

        generation_valid_count += int(generation_valid)
        accepted_generation_count += int(generation_valid)

        out_path = os.path.join(
            output_dir,
            f"{sample_prefix}-sample{i + 1}.txt",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"PATCH_STRUCTURE_VALID: {patch_valid}\n")
            f.write(f"PYTHON_SYNTAX_VALID: {syntax_valid}\n\n")
            f.write("===== GENERATED PATCH =====\n")
            f.write(txt)
            f.write("\n")

        print(f"\n=== Example {i + 1} ===\n")

        print("[IMPLEMENTATION]")
        print(txt.strip())

        print(
            f"[Inference] saved {out_path}"
        )

    print("\n------------------------------------------------------------")
    print("[MoE][Testing][OUTPUT-VALIDITY]")
    print("------------------------------------------------------------\n")

    print(
        f"patch_structure_validity="
        f"{patch_valid_count / max(n_samples, 1):.4f}"
    )

    print(
        f"python_syntax_validity="
        f"{syntax_valid_count / max(n_samples, 1):.4f}"
    )

    print(
        f"generation_validity_rate="
        f"{generation_valid_count / max(n_samples, 1):.4f}"
    )

    accepted_rate = (
        accepted_generation_count
        / max(n_samples, 1)
    )

    print(
        f"accepted_generation_rate="
        f"{accepted_rate:.4f}"
    )

    print(
        f"\n[Inference Summary] "
        f"valid={generation_valid_count}/{n_samples} "
        f"({100.0 * generation_valid_count / max(n_samples,1):.2f}%) "
        f"| generation_validity_rate="
        f"{generation_valid_count / max(n_samples,1):.4f}"
    )


# ============================================================
# Main
# ============================================================

def run_all(cfg: Config = CFG):

    set_seed(cfg.seed)

    print("\n" + "=" * 60)
    print("Round 1")
    print("=" * 60 + "\n")

    print("MoE Baseline: SWE-bench\n")

    data = SWEText2PatchData(
        split="train",
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
        demo_data=cfg.demo_data,
    )

    ids, X, Y = data.as_tensors()

    N = len(ids)

    g = torch.Generator().manual_seed(cfg.seed)

    perm = torch.randperm(
        N,
        generator=g,
    )

    ids = [ids[i] for i in perm.tolist()]
    X = X[perm]
    Y = Y[perm]

    split = int(N * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    Y_train = Y[:split]
    Y_test = Y[split:]

    print(f"[Info] Train: {split} | Test: {N - split}")

    max_len_for_model = max(
        cfg.max_len_cap,
        X.size(1) + cfg.max_out_len,
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
        expert_dim=cfg.expert_dim,
    )

    print(
        f"[MoE][Model] experts={cfg.n_experts} | "
        f"top_k={cfg.top_k} | "
        f"model_dim={cfg.model_dim}"
    )

    print(
        "[MoE][Training] "
        "Stage 1: Expert-Routed IMPLEMENTATION"
    )
    
    train_moe(
        model,
        X_train,
        Y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        lambda_balance=cfg.load_balance_lambda,
        device=DEVICE,
    )

    ce, acc = evaluate_ce_acc(
        model,
        X_test,
        Y_test,
        device=DEVICE,
    )

    stats = compute_moe_efficiency_stats(model)

    print_moe_efficiency_stats(model)


    print("\n------------------------------------------------------------")
    print("[MoE][Eval][PIPELINE-LIFT]")
    print("------------------------------------------------------------\n")

    print(
        "Single-stage architecture "
        "(no intermediate specification stage)\n"
    )

    print(
        f"IMPLEMENTATION CE={ce:.3f}"
    )

    print(
        f"IMPLEMENTATION tok_acc={acc:.3f}"
    )

    print()

    print(
        f"[MoE][Testing][IMPLEMENTATION] "
        f"CE={ce:.3f} | "
        f"tok_acc={acc:.3f} | "
        f"N={X_test.size(0)}"
    )

    print("\n------------------------------------------------------------")
    print("[MoE][Testing][SPEC-STATS]")
    print("------------------------------------------------------------\n")

    print("sampled=0")
    print("avg_tokens=0.00")
    print("avg_lines=0.00")
    print("field_coverage=0.00")

    print("\n------------------------------------------------------------")
    print("[MoE][Training] Stage 2A: N/A")
    print("------------------------------------------------------------\n")

    print(
        "No dedicated specification expert "
        "exists in standard MoE."
    )

    print()

    print(
        "[MoE][Eval][SPEC@Before FT] "
        "N/A"
    )

    print(
        "[MoE][Eval][SPEC@After FT] "
        "N/A"
    )

    print("\n------------------------------------------------------------")
    print("[MoE][Training] Stage 2B: N/A")
    print("------------------------------------------------------------\n")

    print(
        "No post-deployment expert specialization "
        "performed."
    )

    print()

    print(
        "[MoE][Eval][IMPLEMENTATION@Before FT] "
        f"CE={ce:.3f} | "
        f"tok_acc={acc:.3f}"
    )

    print(
        "[MoE][Eval][IMPLEMENTATION@After FT] "
        f"CE={ce:.3f} | "
        f"tok_acc={acc:.3f} | "
        f"ΔCE=+0.000 (0.00%) | "
        f"Δacc=+0.000 (0.00%)"
    )

    print(
        f"[MoE][Adaptation-Cost][IMPLEMENTATION] "
        f"trainable_params={stats['trainable_params']} | "
        f"total_params={stats['total_params']} | "
        f"trainable_ratio={stats['trainable_ratio']:.6f}"
    )

    print("\n[Inference] Generating samples")

    generate_validated_samples(
        model,
        data.tok,
        X_test,
        output_dir=cfg.out_dir,
        sample_prefix="swebench-moe",
        max_len=cfg.decode_max_len,
        n_samples=cfg.n_validation_samples,
        device=DEVICE,
    )

    if hasattr(model, "_routing_counter"):

        print(
            f"[MoE][Routing] "
            f"routing_decisions={model._routing_counter}"
        )

    return model, data, (ids, X, Y)


if __name__ == "__main__":
    model, data, tensors = run_all(CFG)