# COMPLETE CONSISTENCY-CHECKED VERSION
# Agentic seq2seq — Static Role Pipeline (CPU-only, no autotune)
#
# This version fixes and normalizes the following consistency issues:
#
# 1. Unified decoding path for specification + implementation agents.
# 2. Shared encoder/decoder/token embedding path.
# 3. Consistent BOS/EOS/PAD handling.
# 4. Consistent static generation behavior.
# 5. Stable implementation-input construction.
# 6. Proper parameter freezing/unfreezing during specialization.
# 7. Explicit router semantics.
# 8. Fixed tensor-device consistency.
# 9. Fixed generation-time no-repeat-ngram behavior.
# 10. Fixed tokenizer decode cleanup.
# 11. Added full pipeline execution path.
# 12. Checkpoint load/save utilities not included in this compact version.
# 13. Added deterministic evaluation path.
# 14. Added exact end-to-end runnable implementation.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Callable
import os
import re
import ast
import random
import tempfile
import json
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence

# ============================================================
# Repro (CPU-only)
# ============================================================
DEVICE = "cpu"

# ===== Fixed role indices for strict pipeline =====
AGENT_SPECIFICATION   = 0
AGENT_IMPLEMENTATION  = 1

def agent_pretty_name(agent_id: int) -> str:
    return "Specification Agent" if agent_id == AGENT_SPECIFICATION else (
            "Implementation Agent" if agent_id == AGENT_IMPLEMENTATION else f"Agent {agent_id}"
    )

def set_seed(s: int = 42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

# ============================================================
# Config
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
    spec_decode_len: int = 320
    impl_decode_len: int = 160
    stage1_generated_spec_prob: float = 1.0

    # model
    n_agents: int = 2
    model_dim: int = 384
    n_heads: int = 4
    n_layers_enc: int = 4
    n_layers_dec: int = 4
    max_len_cap: int = 640

    # stage 1
    pipe_epochs: int = 20
    pipe_batch: int = 8
    pipe_lr: float = 2e-4

    # stage 2
    ft_epochs: int = 4
    ft_batch: int = 8
    ft_lr: float = 1e-4
    ft_unfreeze_adapters: bool = True
    ft_unfreeze_dec_norms: bool = False

    # validation
    max_repair_attempts: int = 3
    n_validation_samples: int = 10

    # outputs
    out_dir: str = "outputs/humaneval"
    deployed_checkpoint: str = "outputs/humaneval/deployed_joint.pt"

    spec_validity_threshold: float = 0.90
    spec_gate_samples: int = 33

    lambda_constraint: float = 0.01

CFG = Config()

def dataclass_to_dict(obj):
    return dict(obj.__dict__)


def get_tokenizer_proto(tok):
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


def save_deployed_checkpoint(path, model, cfg, data, meta=None):
    ckpt_path = Path(path).expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dataclass_to_dict(cfg),
            "tokenizer_model_proto": get_tokenizer_proto(data.tok),
            "meta": meta or {},
        },
        ckpt_path,
    )

    print(f"[Checkpoint] saved deployed HumanEval checkpoint: {ckpt_path}")

DEBUG_CTX = False
DEBUG_CTX_ONCE = True

# ============================================================
# Tokenizer: SentencePiece UNIGRAM (required)
# ============================================================
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK, PAD, BOS, EOS = range(4)

try:
    import sentencepiece as spm
    HAVE_SPM = True
except ImportError:
    HAVE_SPM = False

class SubwordTokenizer:
    """SPM UNIGRAM tokenizer trained on provided texts. No whitespace fallback."""
    def __init__(self, texts: Sequence[str], vocab_size: int = 8000, quiet: bool = True):
        if not HAVE_SPM:
            raise RuntimeError("SentencePiece missing. Install with: pip install sentencepiece")
        if vocab_size < 128:
            raise ValueError("spm_vocab must be >= 128")

        import contextlib

        @contextlib.contextmanager
        def _silence_cpp_stdio():
            try:
                import sys
                sys.stdout.flush(); sys.stderr.flush()
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                saved_out, saved_err = os.dup(1), os.dup(2)
                try:
                    os.dup2(devnull_fd, 1); os.dup2(devnull_fd, 2)
                    yield
                finally:
                    os.dup2(saved_out, 1); os.dup2(saved_err, 2)
                    os.close(saved_out); os.close(saved_err); os.close(devnull_fd)
            except Exception:
                yield

        self.quiet = quiet
        with tempfile.TemporaryDirectory() as tmpd:
            corpus = os.path.join(tmpd, "spm_corpus.txt")
            with open(corpus, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(str(t).replace("\r", " ") + "\n")

            model_prefix = os.path.join(tmpd, "spm_model")
            target_vocab = int(min(vocab_size, 80000))

            cmd = (
                f"--input={corpus} "
                f"--model_prefix={model_prefix} "
                f"--vocab_size={target_vocab} "
                f"--character_coverage=0.9995 "
                f"--model_type=unigram "
                f"--user_defined_symbols="
                f"<SPEC>,</SPEC>,"
                f"<SIGNATURE>,</SIGNATURE>,"
                f"<DESCRIPTION>,</DESCRIPTION> "
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
        self.pad_idx, self.unk_idx, self.bos_idx, self.eos_idx = 1, 0, 2, 3

    def encode(self, text: str, add_bos_eos: bool, max_len: int) -> torch.Tensor:
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
    def pad(self): return self.pad_idx
    @property
    def bos(self): return self.bos_idx
    @property
    def eos(self): return self.eos_idx

# ============================================================
# Data loading / batching
# ============================================================
try:
    from datasets import load_dataset
    HAVE_HF = True
except Exception:
    HAVE_HF = False

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
        signature + "\n" +
        "\n".join(
            ln if ln.startswith("    ") else "    " + ln
            for ln in body_lines
        )
    )

def extract_function_body(solution: str) -> str:

    lines = solution.splitlines()

    body = []

    found_def = False

    for line in lines:

        if line.strip().startswith("def "):
            found_def = True
            continue

        if found_def:

            # normal indented line
            if line.startswith("    "):
                body.append(line[4:])

            # preserve blank lines
            elif line.strip() == "":
                body.append("")

            # malformed but non-empty
            else:
                body.append(line)

    cleaned = "\n".join(body).strip()

    if not cleaned:
        return "pass"

    return cleaned

class HumanEvalData:
    def __init__(self, limit: Optional[int] = 164,
                    max_in_len: int = 512, max_out_len: int = 384,
                    spm_vocab_size: int = 8000):
        if not HAVE_HF:
            raise RuntimeError("Install `datasets` to use HumanEval: pip install datasets")

        print("[Data] Load HumanEval…")
        ds = load_dataset("openai_humaneval", split="test")
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        self.samples: List[Tuple[str, str, str]] = []
        for i, ex in enumerate(ds):
            iid = f"he-{i}"
            prompt = str(ex["prompt"])
            full_solution = build_full_humaneval_solution(
                prompt,
                str(ex["canonical_solution"])
            )

            self.samples.append(
                (
                    iid,
                    prompt,
                    full_solution.strip(),
                )
            )

        spec_texts = [
            make_structured_spec(prompt)
            for _, prompt, _ in self.samples
        ]

        texts = (
            [x for _, x, _ in self.samples]
            + [y for _, _, y in self.samples]
            + spec_texts
        )
        special_tag_text = (
            "<SPEC> </SPEC> "
            "<SIGNATURE> </SIGNATURE> "
            "<DESCRIPTION> </DESCRIPTION>"
        )
        texts = texts + [special_tag_text] * 100

        self.tok = SubwordTokenizer(texts, vocab_size=spm_vocab_size)
        self.max_in_len, self.max_out_len = max_in_len, max_out_len

    def as_tensors(self) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        ids: List[str] = []
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []

        for iid, x, y in self.samples:
            ids.append(iid)
            xs.append(self.tok.encode(x, add_bos_eos=False, max_len=self.max_in_len))
            ys.append(
                self.tok.encode(
                    y,
                    add_bos_eos=True,
                    max_len=self.max_out_len
                )
            )

        X = pad_sequence(xs, batch_first=True, padding_value=self.tok.pad)
        Y = pad_sequence(ys, batch_first=True, padding_value=self.tok.pad)
        return ids, X, Y

    def as_tensors_with_spec_targets(
        self,
        spec_max_len: int
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:

        ids, X, Y = self.as_tensors()
        Ps = []

        for _, prompt, _ in self.samples:
            spec_text = make_structured_spec(prompt)
            spec_text = _clean_spec_text(spec_text)
        
            Ps.append(
                self.tok.encode(
                    spec_text,
                    add_bos_eos=True,
                    max_len=spec_max_len
                )
            )

        P = pad_sequence(Ps, batch_first=True, padding_value=self.tok.pad)
        return ids, X, Y, P

# ============================================================
# Core model building blocks
# ============================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int = 512, n_heads: int = 8,
                    n_layers: int = 6, max_len: int = 1024, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tok_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim) * 0.01)
        layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor):
        B, T = x.shape

        h = (
            self.tok_embedding(x)
            + self.pos_embedding[:, :T, :]
        )

        mask = (x == self.pad_token_id)

        mem = self.encoder(
            h,
            src_key_padding_mask=mask,
        )

        valid = (~mask).float()

        denom = (
            valid.sum(dim=1, keepdim=True)
            .clamp_min(1.0)
        )

        pooled = (
            (mem * valid.unsqueeze(-1))
            .sum(dim=1)
            / denom
        )

        return mem, pooled, mask    

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int = 512, n_heads: int = 8,
                    n_layers: int = 6, max_len: int = 1024, pad_idx: int = PAD,
                    tok_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_embedding = tok_embedding if tok_embedding is not None else nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, model_dim) * 0.01)
        layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def _subsequent_mask(self, L: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
    
    def forward(
        self,
        y_in,
        memory,
        src_key_padding_mask,
    ):

        B, Lt = y_in.shape

        y_emb = (
            self.tok_embedding(y_in)
            + self.pos_embedding[:, :Lt, :]
        )

        tgt_mask = self._subsequent_mask(
            Lt,
            y_in.device,
        )

        tgt_key_padding_mask = (
            y_in == self.pad_idx
        )

        return self.decoder(
            y_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )    
class Agent(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int, adapter_dim: int = 124):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, model_dim),
        )
        self.role_head = nn.Linear(model_dim, vocab_size)
        self.router_head = nn.Linear(model_dim, 1)

        self.validator = None
        self.repairer = repair_output_text  

    def project(self, states: torch.Tensor) -> torch.Tensor:
        h = self.adapter(states)
        return self.role_head(h)

@torch.no_grad()
def generate_with_validation_repair(
    *,
    model,
    tok,
    X,
    agent_id: int,
    max_len: int,
    validator,
    max_attempts: int,
):
    rows = []

    for b in range(X.size(0)):
        xb = X[b:b+1]

        out = _generate_static(
            model,
            xb,
            agent_id=agent_id,
            max_len=max_len,
            top_k=None,
            top_p=None,
            temperature=1.0,
            no_repeat_ngram_size=4 if agent_id == AGENT_SPECIFICATION else 3,
            min_len=8 if agent_id == AGENT_SPECIFICATION else 6,
        )

        txt = tok.decode([
            int(t) for t in out[0].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        attempt = 0

        agent_validator = model.routing.agents[agent_id].validator or validator

        while not agent_validator(txt) and attempt < max_attempts:

            repaired_txt = repair_output_text(
                txt,
                agent_id=agent_id
            )

            if agent_validator(repaired_txt):
                txt = repaired_txt
                break

            out = _generate_static(
                model,
                xb,
                agent_id=agent_id,
                max_len=max_len,
                top_k=None,
                top_p=None,
                temperature=1.0,
                no_repeat_ngram_size=4 if agent_id == AGENT_SPECIFICATION else 3,
                min_len=8 if agent_id == AGENT_SPECIFICATION else 6,
            )

            txt = tok.decode([
                int(t)
                for t in out[0].tolist()
                if int(t) not in (tok.pad, tok.bos, tok.eos)
            ])

            attempt += 1

        rows.append(tok.encode(txt, add_bos_eos=True, max_len=max_len))

    return pad_sequence(rows, batch_first=True, padding_value=tok.pad).to(X.device)

def repair_output_text(text: str, *, agent_id: int) -> str:
    text = _clean_spec_text(text)

    if agent_id == AGENT_SPECIFICATION:
        return _postprocess_spec(text)

    return text.strip()

@torch.no_grad()
def teacher_forced_validator_signal(
    *,
    tok,
    logits: torch.Tensor,
    agent_id: int,
) -> Tuple[int, int, float]:
    """
    Non-differentiable Validator_i(Yhat_i) call for training-time
    structural validity tracking.

    Returns:
        valid_count, total_count, validity_rate
    """

    pred_ids = logits.argmax(dim=-1)

    validator = (
        validate_spec_output
        if agent_id == AGENT_SPECIFICATION
        else validate_impl_output
    )

    valid = 0
    total = pred_ids.size(0)

    for b in range(total):
        txt = tok.decode([
            int(t)
            for t in pred_ids[b].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        if validator(txt):
            valid += 1

    rate = valid / max(total, 1)

    return valid, total, rate

class StrictPipeline(nn.Module):
    """
    Strict A→B pipeline on static role heads:
        spec = Agent A(specification) generates from full X
        patch = Agent B(implementation) generates ONLY from propagated specification context
    """
    def __init__(self, agents: nn.ModuleList):
        super().__init__()
        self.agents = agents

    @torch.no_grad()
    def run(
        self,
        model,
        tok,
        X,
        *,
        spec_max_len,
        out_max_len,
        max_in_len,
    ):

        spec_display_ids = generate_with_validation_repair(
            model=model,
            tok=tok,
            X=X,
            agent_id=AGENT_SPECIFICATION,
            max_len=spec_max_len,
            validator=validate_spec_output,
            max_attempts=CFG.max_repair_attempts,
        )

        impl_input = build_spec_plus_anchor_context(
            tok,
            spec_display_ids,
            raw_x=None,
            max_in_len=max_in_len
        ).to(X.device)

        patch_ids = generate_with_validation_repair(
            model=model,
            tok=tok,
            X=impl_input,
            agent_id=AGENT_IMPLEMENTATION,
            max_len=out_max_len,
            validator=validate_impl_output,
            max_attempts=CFG.max_repair_attempts,
        )
        return spec_display_ids, patch_ids
        
class AssignmentModule:
    def __init__(self, n_agents: int): self.n_agents = n_agents
    def __call__(self, user_id: int) -> int:
        if isinstance(user_id, torch.Tensor): return int((user_id % self.n_agents).item())
        return int(user_id) % self.n_agents

class RoutingModule(nn.Module):
    """Static routing via AssignmentModule + Strict A→B pipeline."""
    def __init__(self, agents: nn.ModuleList):
        super().__init__()
        self.agents = agents
        self.assign = AssignmentModule(n_agents=len(agents))
        self.pipeline = StrictPipeline(agents)

    def project_role(self, dec_states: torch.Tensor, *, agent_id: int) -> torch.Tensor:
        return self.agents[agent_id].project(dec_states)

    @torch.no_grad()
    def run_pipeline(self, model: "AgenticTransformerSeq2Seq", tok: "SubwordTokenizer", X: torch.Tensor,
                        *, spec_max_len: int, out_max_len: int, max_in_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pipeline.run(model, tok, X, spec_max_len=spec_max_len, out_max_len=out_max_len, max_in_len=max_in_len)

class AgenticTransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, n_agents: int = 2, model_dim: int = 512,
                    n_heads: int = 8, n_layers_enc: int = 6, n_layers_dec: int = 6,
                    max_len: int = 1024, pad_idx: int = PAD):
        super().__init__()
        self.encoder = Encoder(vocab_size, model_dim, n_heads, n_layers_enc, max_len, pad_idx)
        self.decoder = Decoder(vocab_size, model_dim, n_heads, n_layers_dec, max_len, pad_idx,
                                tok_embedding=self.encoder.tok_embedding)
        agents = nn.ModuleList([Agent(model_dim, vocab_size) for _ in range(n_agents)])
        agents[AGENT_SPECIFICATION].validator = validate_spec_output
        agents[AGENT_IMPLEMENTATION].validator = validate_impl_output
        self.routing = RoutingModule(agents)
        self.pad_idx = pad_idx

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode_states(self, y_in: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor):
        return self.decoder(y_in, memory, src_key_padding_mask)

    def forward_role(self, x: torch.Tensor, y_in: torch.Tensor, *, agent_id: int):
        mem, _cls, src_mask = self.encode(x)
        dec_states = self.decode_states(y_in, mem, src_mask)
        return self.routing.project_role(dec_states, agent_id=agent_id)

# ============================================================
# Decoding & generation (static inference path)
# ============================================================
@torch.no_grad()
def _generate_static(
    model: AgenticTransformerSeq2Seq,
    X: torch.Tensor,
    *,
    agent_id: int,
    max_len: int,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
    no_repeat_ngram_size: int = 0,
    min_len: int = 0
) -> torch.Tensor:
    """
    Static autoregressive generation.

    IMPORTANT:
    This function intentionally does NOT call model.eval().
    The caller is responsible for setting train/eval mode.
    This prevents accidental mode leakage during Stage-1 training
    when generation is used to construct implementation context.
    """

    memory, _cls, src_mask = model.encode(X)

    B = X.size(0)

    vocab_size = (
        model.encoder
        .tok_embedding
        .num_embeddings
    )

    ys = torch.full(
        (B, 1),
        BOS,
        dtype=torch.long,
        device=X.device
    )

    finished = torch.zeros(
        B,
        dtype=torch.bool,
        device=X.device
    )

    for _t in range(1, max_len):

        dec = model.decode_states(
            ys,
            memory,
            src_mask
        )

        step_logits = (
            model.routing.agents[agent_id]
            .project(dec[:, -1:])
            .squeeze(1)
        )

        # ----------------------------------------------------
        # Never generate special tokens
        # ----------------------------------------------------

        step_logits[:, UNK] = float("-inf")
        step_logits[:, PAD] = float("-inf")
        step_logits[:, BOS] = float("-inf")

        # ----------------------------------------------------
        # Block EOS until minimum length reached
        # ----------------------------------------------------

        if ys.size(1) < max(1, min_len):
            step_logits[:, EOS] = float("-inf")

        # ----------------------------------------------------
        # No-repeat n-gram constraint
        # ----------------------------------------------------

        if no_repeat_ngram_size > 0:

            banned = _no_repeat_ngram_mask(
                ys,
                no_repeat_ngram_size,
                vocab_size
            )

            step_logits = step_logits.masked_fill(
                banned,
                float("-inf")
            )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        if temperature != 1.0:

            step_logits = (
                step_logits
                / max(temperature, 1e-8)
            )

        # ----------------------------------------------------
        # Sampling or greedy
        # ----------------------------------------------------

        use_sampling = (
            (top_k is not None and top_k > 0)
            or
            (top_p is not None and 0.0 < top_p < 1.0)
        )

        if use_sampling:

            filtered_logits = _top_k_top_p_filtering(
                step_logits.clone(),
                top_k,
                top_p
            )

            next_tok = (
                torch.distributions
                .Categorical(logits=filtered_logits)
                .sample()
                .unsqueeze(1)
            )

        else:

            next_tok = torch.argmax(
                step_logits,
                dim=-1,
                keepdim=True
            )

        ys = torch.cat(
            [ys, next_tok],
            dim=1
        )

        finished |= (
            next_tok.squeeze(1)
            == EOS
        )

        if finished.all():
            break

    return ys

# --- Sampling utilities ---
def _no_repeat_ngram_mask(ys: torch.Tensor, n: int, vocab_size: int) -> torch.Tensor:
    if n <= 0: return torch.zeros((ys.size(0), vocab_size), dtype=torch.bool, device=ys.device)
    B, L = ys.shape
    mask = torch.zeros((B, vocab_size), dtype=torch.bool, device=ys.device)
    if L < n: return mask
    for b in range(B):
        seq = ys[b].tolist()
        prefix2next = {}
        for i in range(L - n + 1):
            prefix = tuple(seq[i:i + n - 1]); nxt = seq[i + n - 1]
            prefix2next.setdefault(prefix, set()).add(nxt)
        last_prefix = tuple(seq[-(n - 1):]) if n > 1 else tuple()
        banned = prefix2next.get(last_prefix, set())
        if banned: mask[b, list(banned)] = True
    return mask

def _top_k_top_p_filtering(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[..., -1].unsqueeze(-1)
        logits = torch.where(
            logits < thresh,
            torch.full_like(logits, float("-inf")),
            logits
        )

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        to_mask = cum_probs > top_p
        to_mask[..., 1:] = to_mask[..., :-1].clone()
        to_mask[..., 0] = False

        sorted_logits = sorted_logits.masked_fill(to_mask, float("-inf"))

        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, sorted_idx, sorted_logits)
        logits = filtered_logits

    return logits
# ============================================================
# Constraint loss
# ============================================================

class ConstraintLoss(nn.Module):

    def __init__(
        self,
        pad_idx,
        eos_idx,
        diff_token_ids=None,
        repetition_weight=0.30,
        eos_weight=0.30,
        structure_weight=0.40,
    ):

        super().__init__()

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

        self.diff_token_ids = set(diff_token_ids or [])

        self.repetition_weight = repetition_weight
        self.eos_weight = eos_weight
        self.structure_weight = structure_weight

    def forward(
        self,
        logits,
        targets,
    ):

        probs = torch.softmax(logits, dim=-1)

        eos_probs = probs[..., self.eos_idx]

        valid_mask_eos = (
            targets != self.pad_idx
        ).float()

        eos_target = (
            targets == self.eos_idx
        ).float()

        eos_loss = F.binary_cross_entropy(
            eos_probs,
            eos_target,
            reduction="none"
        )

        eos_loss = (
            eos_loss * valid_mask_eos
        ).sum() / valid_mask_eos.sum().clamp_min(1.0)

        # ====================================================
        # differentiable repetition suppression
        # ====================================================

        cur_probs = probs[:, 1:, :]
        prev_probs = probs[:, :-1, :]

        repeat_similarity = (
            cur_probs * prev_probs
        ).sum(dim=-1)

        valid_mask = (
            targets[:, 1:] != self.pad_idx
        ).float()

        repetition_loss = (
            repeat_similarity * valid_mask
        ).sum() / valid_mask.sum().clamp_min(1.0)

        # ====================================================
        # structure-token encouragement
        # ====================================================

        structure_loss = torch.tensor(
            0.0,
            device=logits.device
        )

        if self.diff_token_ids:

            probs = torch.softmax(
                logits,
                dim=-1
            )

            struct_probs = []

            for tid in self.diff_token_ids:

                struct_probs.append(
                    probs[..., tid]
                )

            struct_probs = torch.stack(
                struct_probs,
                dim=-1
            )

            structure_loss = -struct_probs.mean()

        # ====================================================
        # final
        # ====================================================

        total_loss = (
            self.repetition_weight * repetition_loss
            + self.eos_weight * eos_loss
            + self.structure_weight * structure_loss
        )

        return total_loss
    
# ============================================================
# Training utilities: losses, metrics, targets
# ============================================================
class SeqCELoss(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, L, V = logits.shape
        return self.ce(logits.reshape(B*L, V), targets.reshape(B*L))

def shift_targets(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return y[:, :-1], y[:, 1:]

def build_mixed_decoder_inputs(
    gold_y_in: torch.Tensor,
    pred_y_in: torch.Tensor,
    teacher_force_ratio: float,
    *,
    bos_idx: int = BOS,
    pad_idx: int = PAD,
) -> torch.Tensor:
    mixed = gold_y_in.clone()

    replace_mask = (
        torch.rand(gold_y_in.shape, device=gold_y_in.device)
        > teacher_force_ratio
    )

    # never replace BOS or PAD positions
    replace_mask[:, 0] = False
    replace_mask = replace_mask & (gold_y_in != pad_idx)

    mixed[replace_mask] = pred_y_in[replace_mask]
    mixed[:, 0] = bos_idx

    return mixed    

def train_spec_supervised(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    P_train: torch.Tensor,
    *,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    unfreeze_backbone: bool = True,
    unfreeze_A_adapter: bool = True,
    unfreeze_dec_norms: bool = True,
):
    """Teacher-force Agent 0 (SPECIFICATION) to generate SPEC_DESC."""
    model.to(device)
    print("[Agentic][Training][SPEC] starting", flush=True)

    _set_trainable_strict_agent(
        model,
        agent_id=AGENT_SPECIFICATION,
        unfreeze_backbone=unfreeze_backbone,
        unfreeze_adapter=unfreeze_A_adapter,
        unfreeze_dec_norms=unfreeze_dec_norms,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    N = X_train.size(0)
    for ep in range(1, epochs + 1):
        model.train()
        sum_loss, tok_correct, tok_total = 0.0, 0, 0
        for i in range(0, N, batch_size):
            xb = X_train[i:i+batch_size].to(device)
            pb = P_train[i:i+batch_size].to(device)
            y_in, y_tgt = shift_targets(pb)
            # --------------------------------------------------
            # Scheduled sampling
            # --------------------------------------------------

            logits = model.forward_role(
                xb,
                y_in,
                agent_id=AGENT_SPECIFICATION
            )
            loss = loss_fn(logits, y_tgt)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask  = (y_tgt != model.pad_idx)
                tok_correct += ((preds == y_tgt) & mask).sum().item()
                tok_total   += mask.sum().item()
                sum_loss    += float(loss.detach()) * xb.size(0)

        print(
            f"[Agentic][Train][Spec] "
            f"epoch={ep}/{epochs} "
            f"spec_ce={sum_loss/float(N):.3f} "
            f"spec_acc={(tok_correct/max(tok_total,1)):.3f}"
        )
    print("[Agentic][Training][SPEC] done ✅", flush=True)

def fine_tune_static(
    model: AgenticTransformerSeq2Seq,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    user_id: int,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = True,
    unfreeze_decoder_tail_blocks: int = 1,
    idxs: Optional[torch.Tensor] = None,
    device: str = DEVICE,
    tok: Optional["SubwordTokenizer"] = None,
    P: Optional[torch.Tensor] = None,
    gist_ctx_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    X_gist: Optional[torch.Tensor] = None,
    max_in_len: Optional[int] = None,
    patience: int = 2
):
    if user_id == AGENT_SPECIFICATION and P is None:
        raise ValueError("fine_tune_static(spec): P (gold SPEC_DESC) is required.")

    model.to(device)

    _set_ft_requires_grad(
        model,
        user_id=user_id,
        unfreeze_adapters=unfreeze_adapters,
        unfreeze_dec_norms=unfreeze_dec_norms
    )

    # --------------------------------------------------
    # Never allow SPEC agent to be updated after Stage 0
    # --------------------------------------------------

    if user_id == AGENT_IMPLEMENTATION:

        for p in model.routing.agents[AGENT_SPECIFICATION].parameters():
            p.requires_grad = False

    if unfreeze_decoder_tail_blocks and unfreeze_decoder_tail_blocks > 0:
        _unfreeze_decoder_tail(model, n_last_blocks=int(unfreeze_decoder_tail_blocks))

    params = [p for p in model.parameters() if p.requires_grad]

    opt = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    print("\n[Agentic][FT] Trainable parameter summary")

    trainable = 0
    total = 0

    for name, p in model.named_parameters():

        total += p.numel()

        if p.requires_grad:

            trainable += p.numel()

            print(
                f"{name} | "
                f"shape={tuple(p.shape)} | "
                f"params={p.numel()}"
            )

    print(
        f"\nTrainable={trainable:,} "
        f"/ Total={total:,} "
        f"({100.0 * trainable / max(total,1):.4f}%)"
    )
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    constraint_fn = ConstraintLoss(
        pad_idx=model.pad_idx,
        eos_idx=EOS
    )

    xb_all = X if idxs is None else X[idxs]
    tgt_all = (Y if idxs is None else Y[idxs]) if user_id == AGENT_IMPLEMENTATION else (P if idxs is None else P[idxs])

    N = xb_all.size(0)
    max_in_len = int(max_in_len or xb_all.size(1))

    dev_frac = max(1, int(0.1 * N))
    xb_tr, xb_dev = xb_all[:-dev_frac], xb_all[-dev_frac:]
    tb_tr, tb_dev = tgt_all[:-dev_frac], tgt_all[-dev_frac:]

    best_dev_ce = float("inf")
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        correct_train, total_train = 0, 0
        valid_train, valid_total = 0, 0

        if user_id == AGENT_IMPLEMENTATION:

            if X_gist is not None:

                gist_all = (
                    X_gist
                    if idxs is None
                    else X_gist[idxs]
                )

                gist_tr = gist_all[:-dev_frac]
                gist_dev = gist_all[-dev_frac:]

                X_ctx_tr = gist_tr.to(device)[:, :max_in_len]
                X_ctx_dev = gist_dev.to(device)[:, :max_in_len]

            else:

                with torch.no_grad():

                    spec_tr = gist_ctx_fn(
                        xb_tr.to(device)
                    ).cpu()

                    spec_dev = gist_ctx_fn(
                        xb_dev.to(device)
                    ).cpu()

                X_ctx_tr = spec_tr.to(device)[:, :max_in_len]
                X_ctx_dev = spec_dev.to(device)[:, :max_in_len]
        
        else:
            X_ctx_tr = xb_tr.to(device)[:, :max_in_len]
            X_ctx_dev = xb_dev.to(device)[:, :max_in_len]

        for i in range(0, xb_tr.size(0), batch_size):
            xb = X_ctx_tr[i:i+batch_size].to(device)
            yb = tb_tr[i:i+batch_size].to(device)
            y_in, y_tgt = shift_targets(yb)
            logits = model.forward_role(xb, y_in, agent_id=user_id)
            seq_loss = loss_fn(logits, y_tgt)

            constraint_loss = constraint_fn(
                logits,
                y_tgt
            )

            with torch.no_grad():
                v_ok, v_total, _ = teacher_forced_validator_signal(
                    tok=tok,
                    logits=logits,
                    agent_id=user_id,
                )
                valid_train += v_ok
                valid_total += v_total

            lambda_constraint = CFG.lambda_constraint

            loss = (
                seq_loss
                + lambda_constraint * constraint_loss
            )

            preds = logits.argmax(-1)
            correct_train += (preds == y_tgt).masked_select(y_tgt != model.pad_idx).sum().item()
            total_train += (y_tgt != model.pad_idx).sum().item()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            ep_loss += float(loss.detach()) * xb.size(0)

        train_acc = correct_train / max(total_train, 1)
        train_ce = ep_loss / float(max(len(xb_tr), 1))

        model.eval()
        with torch.no_grad():
            y_in_dev, y_tgt_dev = shift_targets(tb_dev.to(device))
            logits_dev = model.forward_role(X_ctx_dev.to(device), y_in_dev, agent_id=user_id)
            dev_ce = float(loss_fn(logits_dev, y_tgt_dev).item())

            preds_dev = logits_dev.argmax(-1)
            correct_dev = (preds_dev == y_tgt_dev).masked_select(y_tgt_dev != model.pad_idx).sum().item()
            total_dev = (y_tgt_dev != model.pad_idx).sum().item()
            dev_acc = correct_dev / max(total_dev, 1)

        print(
            f"[Agentic][Train][FT] "
            f"epoch={ep}/{epochs} "
            f"train_ce={train_ce:.3f} "
            f"train_acc={train_acc:.3f} "
            f"val_ce={dev_ce:.3f} "
            f"val_acc={dev_acc:.3f} "
            f"train_valid={valid_train/max(valid_total,1):.3f}"
        )

        if dev_ce + 1e-4 < best_dev_ce:
            best_dev_ce = dev_ce
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("[Agentic][Static Routing] Early stopping triggered")
                break

# ============================================================
# Training helpers (freezing)
# ============================================================
def _set_ft_requires_grad(model: AgenticTransformerSeq2Seq, *, user_id: int, unfreeze_adapters: bool, unfreeze_dec_norms: bool):
    for p in model.parameters(): p.requires_grad = False
    if unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name: p.requires_grad = True
    idx = user_id % len(model.routing.agents)
    ag = model.routing.agents[idx]
    for name, p in ag.named_parameters():
        if name.startswith(("role_head", "router_head")):
            p.requires_grad = True
        elif unfreeze_adapters and name.startswith("adapter"):
            p.requires_grad = True

def _set_trainable_strict_agent(
    model: AgenticTransformerSeq2Seq,
    *,
    agent_id: int = AGENT_IMPLEMENTATION,
    unfreeze_backbone: bool = True,
    unfreeze_adapter: bool = True,
    unfreeze_dec_norms: bool = True
):
    for p in model.parameters():
        p.requires_grad = False
    ag = model.routing.agents[agent_id]
    for name, p in ag.named_parameters():
        if name.startswith(("role_head", "router_head")):
            p.requires_grad = True
        elif unfreeze_adapter and name.startswith("adapter"): p.requires_grad = True
    if unfreeze_backbone:
        for p in model.encoder.parameters(): p.requires_grad = True
        for p in model.decoder.parameters(): p.requires_grad = True
    elif unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name: p.requires_grad = True

def _set_trainable_stage1_joint(
    model: AgenticTransformerSeq2Seq,
    *,
    unfreeze_backbone: bool = False,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = False,
):
    """
    Stage 1 pipeline learning.

    Stage 0:
        Prompt -> Specification

    Stage 1:
        Generated Specification -> Implementation

    SPEC agent is frozen.
    Shared encoder/decoder are frozen by default.
    Only IMPL-specific parameters are trainable.
    """

    if unfreeze_backbone:
        raise ValueError(
            "Stage 1 must keep the shared backbone frozen. "
            "Use unfreeze_backbone=False."
        )

    for p in model.parameters():
        p.requires_grad = False

    impl_agent = model.routing.agents[AGENT_IMPLEMENTATION]

    for name, p in impl_agent.named_parameters():
        if name.startswith("role_head"):
            p.requires_grad = True
        elif name.startswith("router_head"):
            p.requires_grad = True
        elif unfreeze_adapters and name.startswith("adapter"):
            p.requires_grad = True

    if unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name:
                p.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "[Stage1] "
        f"trainable={trainable:,} "
        f"/ total={total:,} "
        f"({100.0 * trainable / max(total,1):.4f}%)"
    )
    print("[Stage1] Frozen: Encoder + Decoder + Specification Agent")
    print("[Stage1] Trainable: Implementation Agent")
# ============================================================
# Unified reporting
# ============================================================

# ============================================================
# Adaptation-cost / efficiency metrics
# ============================================================

def compute_agentic_efficiency_stats(
    model,
    *,
    active_agent_id=AGENT_IMPLEMENTATION,
):

    total_params = 0
    trainable_params = 0

    for p in model.parameters():

        n = p.numel()

        total_params += n

        if p.requires_grad:
            trainable_params += n

    # --------------------------------------------------------
    # Shared backbone
    # --------------------------------------------------------

    encoder_params = sum(
        p.numel()
        for p in model.encoder.parameters()
    )

    decoder_params = sum(
        p.numel()
        for p in model.decoder.parameters()
    )

    shared_params = (
        encoder_params
        + decoder_params
    )

    # --------------------------------------------------------
    # Per-agent parameter counts
    # --------------------------------------------------------

    per_agent_params = []

    for agent in model.routing.agents:

        count = sum(
            p.numel()
            for p in agent.parameters()
        )

        per_agent_params.append(count)

    active_agent_params = (
        per_agent_params[active_agent_id]
    )

    # --------------------------------------------------------
    # Inference-active parameters
    # --------------------------------------------------------

    active_inference_params = (
        shared_params
        + active_agent_params
    )

    active_parameter_ratio = (
        active_inference_params
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

        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "shared_params": shared_params,

        "per_agent_params": per_agent_params,

        "active_agent_id": active_agent_id,
        "active_agent_params": active_agent_params,

        "active_inference_params": active_inference_params,
        "active_parameter_ratio": active_parameter_ratio,
    }

def print_round_header(round_id: int):

    print("\n" + "=" * 60)
    print(f"Round {round_id}")
    print("=" * 60 + "\n")


def print_stage1_epoch(
    epoch,
    issue_ce,
    issue_acc,
    code_ce,
    code_acc,
):

    print(
        f"[Agentic][Training][Epoch {epoch}] "
        f"SPEC: CE={issue_ce:.3f} | tok_acc={issue_acc:.3f}  ||  "
        f"IMPLEMENTATION: CE={code_ce:.3f} | tok_acc={code_acc:.3f}"
    )


def print_pipeline_lift(
    ce_no_spec,
    ce_spec,
    acc_no_spec,
    acc_spec,
):

    delta_ce = ce_spec - ce_no_spec

    print("\n------------------------------------------------------------")
    print("[Agentic][Eval][PIPELINE-LIFT]")
    print("------------------------------------------------------------\n")

    print(
        "Teacher-forced implementation comparison "
        "(with propagated spec vs no-spec baseline)\n"
    )

    print(
        f"IMPLEMENTATION CE(no-spec)={ce_no_spec:.3f}"
    )

    print(
        f"IMPLEMENTATION CE(with spec)={ce_spec:.3f}"
    )

    print(f"ΔCE={delta_ce:.3f}\n")

    print(f"acc(no-spec)={acc_no_spec:.3f}")
    print(f"acc(with spec)={acc_spec:.3f}")


def print_spec_stats(
    sampled,
    avg_tokens,
    avg_lines,
    code_leak_lines,
):

    print("\n------------------------------------------------------------")
    print("[Agentic][Eval][SPEC-STATS]")
    print("------------------------------------------------------------\n")

    print(f"sampled={sampled}")
    print(f"avg_tokens={avg_tokens:.2f}")
    print(f"avg_lines={avg_lines:.2f}")
    print(f"code_leak_lines={code_leak_lines:.2f}")


def print_output_validity(
    patch_validity,
    syntax_validity,
    generation_validity,
    accepted_rate,
):

    print("\n------------------------------------------------------------")
    print("[Agentic][Inference][OUTPUT-VALIDITY]")
    print("------------------------------------------------------------\n")

    print(
        f"patch_structure_validity="
        f"{patch_validity:.4f}"
    )

    print(
        f"python_syntax_validity="
        f"{syntax_validity:.4f}"
    )

    print(
        f"generation_validity_rate="
        f"{generation_validity:.4f}"
    )

    print(
        f"accepted_generation_rate="
        f"{accepted_rate:.4f}"
    )


def print_ft_header(
    stage_name,
    description,
):

    print("\n------------------------------------------------------------")
    print(f"[Agentic][Training] {stage_name}")
    print("------------------------------------------------------------\n")

    print(description)
    print()


def print_before_ft(
    label,
    ce,
    acc,
):

    print(
        f"[Agentic][Eval][{label}@Before FT]"
        f"CE={ce:.3f} | tok_acc={acc:.3f}"
    )


def print_after_ft(
    label,
    before_ce,
    after_ce,
    before_acc,
    after_acc,
):

    dce = after_ce - before_ce
    dacc = after_acc - before_acc

    dce_pct = (
        100.0 * dce / max(abs(before_ce), 1e-8)
    )

    dacc_pct = (
        100.0 * dacc / max(abs(before_acc), 1e-8)
    )

    print(
        f"[Agentic][Eval][{label}@After FT]"
        f"CE={after_ce:.3f} | tok_acc={after_acc:.3f} | "
        f"ΔCE={dce:+.3f} ({dce_pct:+.2f}%) | "
        f"Δacc={dacc:+.3f} ({dacc_pct:+.2f}%)"
    )


def print_ft_epoch(
    role_name,
    epoch,
    train_ce,
    train_acc,
    dev_ce,
    dev_acc,
):

    print(
        f"[Agentic][Static Routing]"
        f"[{role_name} FT] "
        f"Epoch {epoch} | "
        f"TrainCE={train_ce:.3f} | "
        f"TrainAcc={train_acc:.3f} | "
        f"DevCE={dev_ce:.3f} | "
        f"DevAcc={dev_acc:.3f}"
    )


def print_pipeline_sample(
    idx,
    spec_text,
    impl_text,
):

    print("\n=== Example {} ===\n".format(idx))

    print("[SPEC]")
    print(spec_text.strip())

    print("\n[IMPLEMENTATION]")
    print(impl_text.strip())            

def print_agentic_efficiency_stats(
    model,
    *,
    stage_name,
    active_agent_id=AGENT_IMPLEMENTATION,
):

    stats = compute_agentic_efficiency_stats(
        model,
        active_agent_id=active_agent_id,
    )

    print("\n------------------------------------------------------------")
    print(f"[Agentic][Adaptation-Cost][{stage_name}]")
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
        f"encoder_params="
        f"{stats['encoder_params']}"
    )

    print(
        f"decoder_params="
        f"{stats['decoder_params']}"
    )

    print(
        f"shared_params="
        f"{stats['shared_params']}"
    )

    print(
        f"active_agent_params="
        f"{stats['active_agent_params']}"
    )

    print(
        f"active_inference_params="
        f"{stats['active_inference_params']}"
    )

    print(
        f"active_parameter_ratio="
        f"{stats['active_parameter_ratio']:.6f}"
    )

    for idx, count in enumerate(stats["per_agent_params"]):

        print(
            f"agent_{idx}_params="
            f"{count}"
        )
        
def print_trainable_modules(model):

    print("\n------------------------------------------------------------")
    print("[Agentic][Trainable-Modules]")
    print("------------------------------------------------------------\n")

    for name, p in model.named_parameters():

        if p.requires_grad:

            print(
                f"{name} | "
                f"shape={tuple(p.shape)} | "
                f"params={p.numel()}"
            )

@torch.no_grad()
def build_stage1_impl_context(
    model,
    tok,
    xb,
    _pb,
    *,
    ep: int,
    total_epochs: int,
    max_in_len: int,
    device: str = DEVICE,
):
    """
    Build Stage-1 implementation input.

    Stage 1 always uses generated SPEC context:

        Prompt -> frozen SPEC agent -> generated SPEC -> IMPL

    The gold SPEC tensor _pb is kept only for signature compatibility.
    """

    spec_rows = generate_with_validation_repair(
        model=model,
        tok=tok,
        X=xb,
        agent_id=AGENT_SPECIFICATION,
        max_len=CFG.spec_decode_len,
        validator=validate_spec_output,
        max_attempts=CFG.max_repair_attempts,
    )

    spec_ctx = build_spec_plus_anchor_context(
        tok,
        spec_rows,
        raw_x=None,
        max_in_len=max_in_len,
    ).to(device)

    return spec_ctx

def train_stage1_interleaved(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    P_train: torch.Tensor,
    *,
    tok: "SubwordTokenizer",
    spec_max_len: int = 124,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    unfreeze_backbone: bool = False,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = False,
    max_in_len: Optional[int] = None,
):
    """
    Stage 1 pipeline learning:

        Prompt
            ->
        Generated SPEC
            ->
        Implementation

    SPEC is used only to generate context.
    No SPEC loss is optimized here.
    """

    assert AGENT_SPECIFICATION == 0
    assert AGENT_IMPLEMENTATION == 1

    model.to(device)

    _set_trainable_stage1_joint(
        model,
        unfreeze_backbone=unfreeze_backbone,
        unfreeze_adapters=unfreeze_adapters,
        unfreeze_dec_norms=unfreeze_dec_norms,
    )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = optim.Adam(params, lr=lr)

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx
    )

    constraint_fn = ConstraintLoss(
        pad_idx=model.pad_idx,
        eos_idx=EOS
    )

    N = X_train.size(0)

    max_in_len = int(
        max_in_len or X_train.size(1)
    )

    for ep in range(1, epochs + 1):
        model.train()

        impl_loss_sum = 0.0
        impl_seq_sum = 0.0
        impl_constraint_sum = 0.0

        impl_correct = 0
        impl_total = 0

        impl_valid_count = 0
        impl_valid_total = 0

        for i in range(0, N, batch_size):
            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)
            pb = P_train[i:i + batch_size].to(device)

            with torch.no_grad():
                spec_ctx = build_stage1_impl_context(
                    model,
                    tok,
                    xb,
                    pb,
                    ep=ep,
                    total_epochs=epochs,
                    max_in_len=max_in_len,
                    device=device,
                )

            y_in_c, y_tgt_c = shift_targets(yb)

            logits_c = model.forward_role(
                spec_ctx,
                y_in_c,
                agent_id=AGENT_IMPLEMENTATION,
            )

            seq_loss_c = loss_fn(
                logits_c,
                y_tgt_c,
            )

            constraint_c = constraint_fn(
                logits_c,
                y_tgt_c,
            )

            loss_c = (
                seq_loss_c
                + CFG.lambda_constraint * constraint_c
            )

            with torch.no_grad():
                v_ok, v_total, _ = teacher_forced_validator_signal(
                    tok=tok,
                    logits=logits_c,
                    agent_id=AGENT_IMPLEMENTATION,
                )

                impl_valid_count += v_ok
                impl_valid_total += v_total

                preds_c = logits_c.argmax(dim=-1)
                mask_c = y_tgt_c != model.pad_idx

                impl_correct += (
                    ((preds_c == y_tgt_c) & mask_c)
                    .sum()
                    .item()
                )

                impl_total += mask_c.sum().item()

            impl_seq_sum += float(seq_loss_c.detach()) * xb.size(0)
            impl_constraint_sum += float(constraint_c.detach()) * xb.size(0)
            impl_loss_sum += float(loss_c.detach()) * xb.size(0)

            opt.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        print(
            f"[Agentic][Train][Stage1] "
            f"epoch={ep}/{epochs} "
            f"impl_ce={impl_loss_sum/float(N):.3f} "
            f"impl_acc={impl_correct/max(impl_total,1):.3f} "
            f"impl_valid={impl_valid_count/max(impl_valid_total,1):.3f}"
        )

        print(
            f"[Constraint] "
            f"impl_seq={impl_seq_sum/float(N):.3f} "
            f"impl_constraint={impl_constraint_sum/float(N):.3f} "
            f"ratio={(impl_constraint_sum/max(impl_seq_sum,1e-8)):.3f}"
        )

@torch.no_grad()
def _eval_impl_ce_acc(
    model: "AgenticTransformerSeq2Seq",
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    device: str = DEVICE
    ) -> Tuple[float, float]:
    """Teacher-forced CE/accuracy for the Impl agent on input X vs gold Y."""
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    y_in, y_tgt = shift_targets(Y.to(device))
    logits = model.forward_role(X.to(device), y_in, agent_id=AGENT_IMPLEMENTATION)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / (mask.float().sum() + 1e-8)).item())
    return ce, acc

@torch.no_grad()
def eval_pipeline(
    model: AgenticTransformerSeq2Seq,
    tok: SubwordTokenizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    spec_max_len: int,
    max_in_len: int,
    device: str = DEVICE
):
    """
    Evaluate the actual deployed pipeline:

        Prompt
          ->
        Generated Specification
          ->
        Implementation

    No oracle specification.
    No raw-prompt implementation shortcut.
    """

    model.to(device)
    model.eval()

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx
    )

    y_in, y_tgt = shift_targets(
        Y.to(device)
    )

    spec_ctx = build_spec_context(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        max_in_len=max_in_len,
        device=device
    ).to(device)

    logits = model.forward_role(
        spec_ctx,
        y_in,
        agent_id=AGENT_IMPLEMENTATION
    )

    ce = float(
        loss_fn(
            logits,
            y_tgt
        ).item()
    )

    acc = float(
        (
            (
                (logits.argmax(-1) == y_tgt)
                &
                (y_tgt != model.pad_idx)
            ).float().sum()
        )
        /
        (
            (y_tgt != model.pad_idx)
            .float()
            .sum()
            .clamp_min(1.0)
        )
    )

    print("\n--------------------------------------------------")
    print("[Agentic][PIPELINE EVALUATION]")
    print("--------------------------------------------------")

    print(
        f"GeneratedSpec->Impl  "
        f"CE={ce:.3f} "
        f"ACC={acc:.3f}"
    )

    return ce, acc

def validate_spec_output(text: str) -> bool:

    text = _clean_spec_text(text)

    required_sections = [
        "<SPEC>",
        "</SPEC>",
        "<SIGNATURE>",
        "</SIGNATURE>",
        "<DESCRIPTION>",
        "</DESCRIPTION>",
    ]

    for sec in required_sections:
        if sec not in text:
            return False

    sig_match = re.search(
        r"<SIGNATURE>(.*?)</SIGNATURE>",
        text,
        re.DOTALL
    )

    desc_match = re.search(
        r"<DESCRIPTION>(.*?)</DESCRIPTION>",
        text,
        re.DOTALL
    )

    if not sig_match or not desc_match:
        return False

    signature = sig_match.group(1).strip()
    description = desc_match.group(1).strip()

    if not re.search(
        r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(",
        signature
    ):
        return False

    if len(description.split()) < 5:
        return False

    # Check only the description, not the whole SPEC.
    if _is_noisy_spec_description(description):
        return False

    return True

@torch.no_grad()
def evaluate_spec_generation_validity(
    model,
    tok,
    X,
    *,
    spec_max_len,
    device=DEVICE,
    n_samples=None,
):
    model.to(device)
    model.eval()

    if n_samples is not None:
        X = X[:min(n_samples, X.size(0))]

    spec_rows = generate_valid_spec_rows(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        device=device,
        fallback_to_prompt_docstring=False,
    )

    valid = 0
    total = X.size(0)

    for i in range(total):
        spec_txt = tok.decode([
            int(t)
            for t in spec_rows[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        if validate_spec_output(spec_txt):
            valid += 1

    rate = valid / max(total, 1)

    print(
        f"[SPEC VALIDITY] "
        f"valid={valid}/{total} "
        f"rate={rate:.4f}"
    )

    return rate

def validate_impl_output(text: str) -> bool:

    code = text.strip()

    if not code:
        return False

    try:
        ast.parse(code)
        return True
    except Exception:
        pass

    wrapped = (
        "def generated_function():\n"
        + "\n".join(
            "    " + ln
            for ln in code.splitlines()
            if ln.strip()
        )
    )

    try:
        ast.parse(wrapped)
        return True
    except Exception:
        return False

def normalize_humaneval_impl(
    code: str,
    spec_txt: str
) -> str:

    code = code.strip()

    anchor = extract_prompt_anchor(spec_txt)

    body_lines = []

    for ln in code.splitlines():

        ln = ln.strip()

        if not ln:
            continue

        if ln.startswith("def "):
            continue

        if (
            "Task:" in ln
            or "Constraints:" in ln
            or "Examples:" in ln
        ):
            continue

        body_lines.append(ln)

    if not body_lines:
        return code

    repaired = (
        anchor
        + "\n"
        + "\n".join(
            "    " + ln
            for ln in body_lines
        )
    )

    try:
        ast.parse(repaired)
        return repaired

    except Exception:
        return code

    
def repair_body_fragment(
    code: str,
    spec_txt: str
) -> str:

    anchor = extract_prompt_anchor(spec_txt)

    candidates = []

    for ln in code.splitlines():

        ln = ln.strip()

        if not ln:
            continue

        if ln.startswith("return "):
            candidates.append(ln)

        elif ln.startswith(("if ", "for ", "while ")):
            candidates.append(ln)

        elif "=" in ln and "==" not in ln:
            candidates.append(ln)

    if not candidates:
        return code

    repaired = (
        anchor
        + "\n"
        + "\n".join(
            "    " + ln
            for ln in candidates
        )
    )

    try:
        ast.parse(repaired)
        return repaired

    except Exception:
        return code
        
def is_trivial_pass_patch(code: str) -> bool:

    txt = code.strip()

    try:
        tree = ast.parse(txt)
    except Exception:
        return False

    if len(tree.body) != 1:
        return False

    fn = tree.body[0]

    if not isinstance(fn, ast.FunctionDef):
        return False

    meaningful = [
        node for node in fn.body
        if not isinstance(node, ast.Expr)
    ]

    # only pass
    if (
        len(meaningful) == 1
        and isinstance(meaningful[0], ast.Pass)
    ):
        return True

    # only return None
    if (
        len(meaningful) == 1
        and isinstance(meaningful[0], ast.Return)
    ):
        return meaningful[0].value is None

    # only raise NotImplementedError
    if (
        len(meaningful) == 1
        and isinstance(meaningful[0], ast.Raise)
    ):
        exc = meaningful[0].exc

        if isinstance(exc, ast.Call):
            if getattr(exc.func, "id", "") == "NotImplementedError":
                return True

        if isinstance(exc, ast.Name):
            if exc.id == "NotImplementedError":
                return True
            
    return False
    
@torch.no_grad()
def validate_and_repair_stage_output(
    *,
    model,
    tok,
    stage_input,
    stage_output,
    agent_id,
    max_len,
    max_attempts,
    max_in_len,
    validator
):
    repaired_rows = []

    for i in range(stage_output.size(0)):

        txt = tok.decode([
            t for t in stage_output[i].tolist()
            if t not in (tok.pad, tok.bos, tok.eos)
        ])

        txt = _clean_spec_text(txt)

        repaired = txt

        for attempt in range(max_attempts):

            if validator(repaired):
                break

            regenerated = _generate_static(
                model,
                stage_input[i:i+1],
                agent_id=agent_id,
                max_len=max_len,
                top_k=None,
                top_p=None,
                temperature=1.0,        
            )

            repaired = tok.decode([
                t for t in regenerated[0].tolist()
                if t not in (tok.pad, tok.bos, tok.eos)
            ])

            if agent_id == AGENT_SPECIFICATION:
                repaired = _clean_spec_text(repaired)
        ids = tok.encode(
            repaired,
            add_bos_eos=True,
            max_len=max_len
        ).to(stage_output.device)

        repaired_rows.append(ids[:max_len])

    return pad_sequence(
        repaired_rows,
        batch_first=True,
        padding_value=tok.pad
    )
# ============================================================
# Runtime inference with validation
# ============================================================

def function_name_matches_spec(code: str, spec_txt: str) -> bool:
    expected = extract_prompt_anchor(spec_txt)

    m_expected = re.search(
        r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        expected
    )

    m_actual = re.search(
        r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        code
    )

    if not m_expected or not m_actual:
        return False

    return m_expected.group(1) == m_actual.group(1)


def has_undefined_names(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return True

    defined = set()
    used = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            for arg in node.args.args:
                defined.add(arg.arg)

        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)

    allowed_builtins = {
        "len", "range", "sorted", "sum", "min", "max",
        "abs", "str", "int", "float", "list", "set",
        "dict", "enumerate", "zip", "all", "any"
    }

    undefined = used - defined - allowed_builtins

    return len(undefined) > 0

def repair_humaneval_body_to_function(
    *,
    anchor: str,
    raw_body: str,
) -> str:

    body = raw_body.strip()

    body = re.sub(r"<[^>]+>", " ", body)
    body = body.replace("```python", "").replace("```", "")
    body = re.sub(r"\s+", " ", body).strip()

    body = re.sub(
        r"^def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:",
        "",
        body
    ).strip()

    candidates = []

    for piece in re.split(r";|\n", body):

        s = piece.strip()

        if not s:
            continue

        if s.startswith(("return ", "if ", "for ", "while ", "try:", "with ")):
            candidates.append(s)

        elif "=" in s and "==" not in s:
            candidates.append(s)

    trial_bodies = []

    if candidates:
        trial_bodies.append(candidates)

    if body:
        trial_bodies.append([f"return {body}"])

    for lines in trial_bodies:

        code = (
            anchor
            + "\n"
            + "\n".join(
                "    " + ln
                for ln in lines
            )
        )

        try:
            ast.parse(code)
            return code
        except Exception:
            pass

    fallback = fallback_function_from_signature(anchor)

    return fallback

def fallback_function_from_signature(anchor: str) -> str:

    if "-> bool" in anchor:
        value = "False"
    elif "-> int" in anchor:
        value = "0"
    elif "-> float" in anchor:
        value = "0.0"
    elif "-> str" in anchor:
        value = "''"
    elif "List[" in anchor or "-> list" in anchor:
        value = "[]"
    else:
        value = "None"

    return (
        anchor
        + "\n"
        + f"    return {value}"
    )

@torch.no_grad()
def generate_validated_samples(
    model,
    tok,
    ids,
    X,
    *,
    output_dir: str,
    sample_prefix: str,
    spec_max_len: int,
    out_max_len: int,
    max_in_len: int,
    n_samples: int = 10,
    max_repair_attempts: int = 3,
    device: str = DEVICE
):

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)
    model.eval()

    n_samples = min(n_samples, X.size(0))

    generated_patches = []
    results = []

    predictions = []

    for i in range(n_samples):

        instance_id = ids[i]

        x = X[i:i+1].to(device)

        raw_prompt_txt = tok.decode([
            int(t)
            for t in x[0].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        # --------------------------------------------------
        # Pipeline
        # --------------------------------------------------

        spec_ids, patch_ids = model.routing.run_pipeline(
            model,
            tok,
            x,
            spec_max_len=spec_max_len,
            out_max_len=out_max_len,
            max_in_len=max_in_len
        )

        spec_txt = tok.decode([
            int(t)
            for t in spec_ids[0].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        patch_txt = tok.decode([
            int(t)
            for t in patch_ids[0].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        # --------------------------------------------------
        # Repair
        # --------------------------------------------------

        anchor = extract_prompt_anchor(raw_prompt_txt)

        repaired_txt = patch_txt.strip()

        if (
            not repaired_txt.startswith("def ")
            or not validate_impl_output(repaired_txt)
        ):
            repaired_txt = repair_humaneval_body_to_function(
                anchor=anchor,
                raw_body=repaired_txt,
            )

        syntax_valid = (
            validate_impl_output(repaired_txt)
            and "NotImplementedError" not in repaired_txt
        )

        lexical_overlap = lexical_overlap_score(
            spec_txt,
            repaired_txt
        )

        trivial_pass = is_trivial_pass_patch(
            repaired_txt
        )

        name_valid = function_name_matches_spec(
            repaired_txt,
            raw_prompt_txt
        )

        undefined_valid = not has_undefined_names(
            repaired_txt
        )

        patch_valid = (
            syntax_valid
            and name_valid
            and undefined_valid
            and not trivial_pass
            and "return None" not in repaired_txt
            and "return 0" not in repaired_txt
            and "return ''" not in repaired_txt
            and "return []" not in repaired_txt
            and "NotImplementedError" not in repaired_txt
        )

        final_valid = patch_valid

        generated_patches.append(
            repaired_txt
        )

        results.append({
            "instance_id": instance_id,
            "accepted": final_valid,
            "patch_valid": patch_valid,
            "syntax_valid": syntax_valid,
            "lexical_overlap": lexical_overlap,
        })

        # ==================================================
        # SWE-style export record
        # ==================================================

        predictions.append({
            "instance_id": instance_id,

            "prompt": raw_prompt_txt,

            "issue_gist": spec_txt,

            "generated_patch": repaired_txt,

            "accepted": bool(final_valid),

            "patch_valid": bool(patch_valid),

            "syntax_valid": bool(syntax_valid),

            "lexical_overlap": float(lexical_overlap),
        })

        # ==================================================
        # Human-readable sample file
        # ==================================================

        out_path = os.path.join(
            output_dir,
            f"{sample_prefix}-sample{i+1}.txt"
        )

        with open(out_path, "w", encoding="utf-8") as f:

            f.write(
                f"INSTANCE_ID: {instance_id}\n"
            )

            f.write(
                f"ACCEPTED: {final_valid}\n"
            )

            f.write(
                f"PATCH_VALID: {patch_valid}\n"
            )

            f.write(
                f"SYNTAX_VALID: {syntax_valid}\n"
            )

            f.write(
                f"LEXICAL_OVERLAP: "
                f"{lexical_overlap:.4f}\n\n"
            )

            f.write("===== PROMPT =====\n")
            f.write(raw_prompt_txt)

            f.write("\n\n===== ISSUE_GIST =====\n")
            f.write(spec_txt)

            f.write("\n\n===== GENERATED_PATCH =====\n")
            f.write(repaired_txt)

    # ======================================================
    # Aggregate metrics
    # ======================================================

    syntax_valid_rate = (
        sum(r["syntax_valid"] for r in results)
        / max(len(results), 1)
    )

    patch_valid_rate = (
        sum(r["patch_valid"] for r in results)
        / max(len(results), 1)
    )

    accepted_rate = (
        sum(r["accepted"] for r in results)
        / max(len(results), 1)
    )

    consistency = pairwise_consistency(
        generated_patches
    )

    print(
        "\n[Inference Summary] "
        f"accepted_rate={accepted_rate:.4f} "
        f"| patch_valid_rate={patch_valid_rate:.4f} "
        f"| syntax_valid_rate={syntax_valid_rate:.4f} "
        f"| consistency={consistency:.4f}"
    )

    pred_path = os.path.join(
        output_dir,
        "predictions.jsonl"
    )

    with open(pred_path, "w", encoding="utf-8") as f:

        for row in predictions:
            f.write(
                json.dumps(row)
                + "\n"
            )

    print(
        f"[JSONL Export] saved -> {pred_path}"
    )
    
# ============================================================
# Diagnostics / reporting
# ============================================================
@torch.no_grad()
def _print_agent_role_outputs_after(model: AgenticTransformerSeq2Seq, X: torch.Tensor, y: torch.Tensor,
                                    *, n_tokens: int = 3, n_agents: Optional[int] = None,
                                    device: str = DEVICE) -> None:
    """Debug helper: prints a small slice of role-head logits for each agent on the last decode step."""
    model.to(device); model.eval()
    nA = len(model.routing.agents) if n_agents is None else n_agents
    y_in, _ = shift_targets(y.to(device))
    mem, cls, src_mask = model.encode(X.to(device))
    dec_states = model.decode_states(y_in, mem, src_mask)
    last = dec_states[:, -1:]
    for a in range(nA):
        logits = model.routing.agents[a].project(last)
        vec = logits[0, 0, :n_tokens].detach().cpu().numpy()
        print(f"[{agent_pretty_name(a)}] role_head logits[:{n_tokens}] -> {vec}")

# ============================================================
# Validation / repair utilities
# ============================================================

def tokenize_consistency_text(text: str):
    return tokenize_lexical_overlap_text(text)

def tokenize_lexical_overlap_text(text: str):

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower()
    )


def lexical_overlap_score(
    spec_txt: str,
    patch_txt: str
) -> float:

    spec_tokens = set(
        tokenize_lexical_overlap_text(spec_txt)
    )

    patch_tokens = set(
        tokenize_lexical_overlap_text(patch_txt)
    )

    if not spec_tokens or not patch_tokens:
        return 0.0

    overlap = spec_tokens.intersection(
        patch_tokens
    )

    return len(overlap) / max(len(spec_tokens), 1)

def generation_validity_rate(results):

    if not results:
        return 0.0

    valid = sum(
        1 for r in results
        if r["final_valid"]
    )

    return valid / len(results)

def pairwise_consistency(
    samples: List[str]
) -> float:

    if len(samples) <= 1:
        return 1.0

    scores = []

    for i in range(len(samples)):

        for j in range(i + 1, len(samples)):

            a = set(
                tokenize_consistency_text(samples[i])
            )

            b = set(
                tokenize_consistency_text(samples[j])
            )

            if not a or not b:
                continue

            overlap = len(a.intersection(b))
            union = len(a.union(b))

            scores.append(
                overlap / max(union, 1)
            )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)

# ============================================================
# Small tensor helpers
# ============================================================

@torch.no_grad()
def spec_analysis_stats(
    model: AgenticTransformerSeq2Seq,
    tok: SubwordTokenizer,
    X: torch.Tensor,
    *,
    spec_max_len: int,
    device: str = DEVICE
) -> Dict[str, float]:

    decoded = []

    for i in range(min(4, X.size(0))):
        raw_txt = tok.decode([
            int(t)
            for t in X[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        spec_txt = make_structured_spec(raw_txt)
        spec_txt = _clean_spec_text(spec_txt)
        decoded.append(spec_txt)

    lengths = [len(s.split()) for s in decoded]
    line_counts = [s.count("\n") + 1 for s in decoded]

    impl_leak_lines = sum(
        1 for s in decoded
        for ln in s.splitlines()
        if ("diff --git" in ln)
        or ln.strip().startswith("class ")
        or ("```" in ln)
    )

    return {
        "sampled": float(len(decoded)),
        "avg_tokens": float(np.mean(lengths) if lengths else 0.0),
        "avg_lines": float(np.mean(line_counts) if line_counts else 0.0),
        "impl_leak_lines": float(impl_leak_lines),
    }

@torch.no_grad()
def spec_to_impl_alignment_sample(model: AgenticTransformerSeq2Seq, tok: SubwordTokenizer, X: torch.Tensor,
                                    *, spec_max_len: int, out_max_len: int, max_in_len: int, k: int = 3,
                                    device: str = DEVICE) -> None:
    """Print K examples: spec (A) and patch (B) to eyeball alignment."""
    model.to(device); model.eval()
    Xk = X[:k].to(device)
    spec_ids, patch_ids = model.routing.run_pipeline(
        model, tok, Xk, spec_max_len=spec_max_len, out_max_len=out_max_len, max_in_len=max_in_len
    )
    for i in range(min(k, Xk.size(0))):
        spec = tok.decode([t for t in spec_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)])
        patch = tok.decode([t for t in patch_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)])
        print(f"\n=== Example {i} ===")
        print("[SPEC]\n", spec[:800])
        print("\n[PATCH]\n", patch[:800])

def _first_sentence(txt: str) -> str:
    # crude first sentence splitter; falls back to first ~30 words
    txt = re.sub(r"\s+", " ", txt).strip()
    m = re.search(r"(.+?[.!?])(\s|$)", txt)
    if m:
        return m.group(1)
    # fallback: ~30 words
    parts = txt.split()
    return " ".join(parts[:30]) if parts else ""

def _first_n_words(txt: str, n: int = 100) -> str:
    parts = txt.split()[:n]
    return ' '.join(parts)

def _first_k_sentences(txt: str, k: int = 3) -> str:
    # Split on sentence boundaries, preserving punctuation
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', txt)[:k]
    return ' '.join(sentences).strip()

def make_structured_spec(prompt: str) -> str:
    """
    Compact specification.

    HumanEval contains only:
        - function signature
        - task description (docstring)

    The specification agent learns:

        Prompt
            ->
        Structured Prompt

    not

        Prompt
            ->
        Synthetic document
    """

    txt = str(prompt).strip()

    signature = extract_prompt_anchor(txt)

    doc_match = re.search(
        r'"""(.*?)"""',
        txt,
        re.DOTALL
    )

    description = (
        doc_match.group(1).strip()
        if doc_match else ""
    )

    description = re.sub(
        r"\s+",
        " ",
        description
    ).strip()

    return (
        "<SPEC>\n"
        "<SIGNATURE>\n"
        f"{signature}\n"
        "</SIGNATURE>\n"
        "<DESCRIPTION>\n"
        f"{description}\n"
        "</DESCRIPTION>\n"
        "</SPEC>"
    )

def extract_prompt_anchor(prompt: str) -> str:
    """
    Extract ONLY a valid Python function signature.

    Prevents docstrings / task text / examples from leaking into
    the generated function header.
    """

    txt = str(prompt)

    m = re.search(
        r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:",
        txt
    )

    if m:
        return m.group(0).strip()

    m2 = re.search(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        txt
    )

    if m2:
        fn_name = m2.group(1)
        return f"def {fn_name}(*args):"

    return "def generated_function(*args):"

def force_signature_from_prompt(
    generated_spec: str,
    prompt_text: str
) -> str:
    """
    Replace whatever signature Agent A generated
    with the original HumanEval signature.
    """

    gold_sig = extract_prompt_anchor(prompt_text)

    generated_spec = _clean_spec_text(
        generated_spec
    )

    # --------------------------------------------------
    # Existing signature section
    # --------------------------------------------------

    if (
        "<SIGNATURE>" in generated_spec
        and "</SIGNATURE>" in generated_spec
    ):

        generated_spec = re.sub(
            r"<SIGNATURE>.*?</SIGNATURE>",
            (
                "<SIGNATURE>\n"
                f"{gold_sig}\n"
                "</SIGNATURE>"
            ),
            generated_spec,
            flags=re.DOTALL
        )

        return generated_spec

    # --------------------------------------------------
    # Missing signature section
    # --------------------------------------------------

    if "<SPEC>" in generated_spec:

        generated_spec = generated_spec.replace(
            "<SPEC>",
            (
                "<SPEC>\n"
                "<SIGNATURE>\n"
                f"{gold_sig}\n"
                "</SIGNATURE>\n"
            ),
            1
        )

    return generated_spec

@torch.no_grad()
def force_signature_rows_from_prompts(
    tok,
    spec_rows: torch.Tensor,
    prompt_rows: torch.Tensor,
    *,
    max_len: int,
    device: str = DEVICE,
) -> torch.Tensor:

    fixed_rows = []

    for i in range(spec_rows.size(0)):

        prompt_txt = tok.decode([
            int(t)
            for t in prompt_rows[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        spec_txt = tok.decode([
            int(t)
            for t in spec_rows[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        gold_spec = make_structured_spec(prompt_txt)

        generated_desc = ""

        m = re.search(
            r"<DESCRIPTION>(.*?)</DESCRIPTION>",
            spec_txt,
            re.DOTALL
        )

        if m:
            generated_desc = m.group(1).strip()

        gold_spec = re.sub(
            r"<DESCRIPTION>.*?</DESCRIPTION>",
            (
                "<DESCRIPTION>\n"
                + generated_desc +
                "\n</DESCRIPTION>"
            ),
            gold_spec,
            flags=re.DOTALL
        )

        spec_txt = gold_spec

        fixed_rows.append(
            tok.encode(
                spec_txt,
                add_bos_eos=True,
                max_len=max_len
            )
        )

    return pad_sequence(
        fixed_rows,
        batch_first=True,
        padding_value=tok.pad
    ).to(device)


@torch.no_grad()
def generate_valid_spec_rows(
    model,
    tok,
    X,
    *,
    spec_max_len,
    device=DEVICE,
    fallback_to_prompt_docstring: bool = True,
):
    model.to(device)
    model.eval()

    raw_ids = _generate_static(
        model,
        X.to(device),
        agent_id=AGENT_SPECIFICATION,
        max_len=spec_max_len,
        top_k=None,
        top_p=None,
        temperature=1.0,
        no_repeat_ngram_size=4,
        min_len=8,
    )

    fixed_rows = []

    for i in range(X.size(0)):

        prompt_txt = tok.decode([
            int(t)
            for t in X[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        raw_spec_txt = tok.decode([
            int(t)
            for t in raw_ids[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        gold_sig = extract_prompt_anchor(prompt_txt)

        desc = ""

        m = re.search(
            r"<DESCRIPTION>(.*?)</DESCRIPTION>",
            raw_spec_txt,
            re.DOTALL
        )

        if m:
            desc = m.group(1).strip()

        desc = _clean_spec_text(desc)
        desc = re.sub(r"\s+", " ", desc).strip()

        if fallback_to_prompt_docstring and _is_noisy_spec_description(desc):
            gold_spec = make_structured_spec(prompt_txt)
            m2 = re.search(
                r"<DESCRIPTION>(.*?)</DESCRIPTION>",
                gold_spec,
                re.DOTALL
            )
            desc = m2.group(1).strip() if m2 else ""

        spec_txt = (
            "<SPEC>\n"
            "<SIGNATURE>\n"
            f"{gold_sig}\n"
            "</SIGNATURE>\n"
            "<DESCRIPTION>\n"
            f"{desc}\n"
            "</DESCRIPTION>\n"
            "</SPEC>"
        )

        if fallback_to_prompt_docstring and not validate_spec_output(spec_txt):
            spec_txt = make_structured_spec(prompt_txt)

        fixed_rows.append(
            tok.encode(
                spec_txt,
                add_bos_eos=True,
                max_len=spec_max_len
            )
        )

    return pad_sequence(
        fixed_rows,
        batch_first=True,
        padding_value=tok.pad
    ).to(device)


def build_spec_plus_anchor_context(
    tok,
    spec_rows,
    raw_x=None,
    *,
    max_in_len,
):
    rows = []

    for i in range(spec_rows.size(0)):
        row = spec_rows[i]

        trimmed = []
        for t in row.tolist():
            t = int(t)

            if t == tok.pad:
                continue

            trimmed.append(t)

            if t == tok.eos:
                break

        if not trimmed:
            trimmed = [tok.bos, tok.eos]

        rows.append(
            torch.tensor(
                trimmed[:max_in_len],
                dtype=torch.long
            )
        )

    return pad_sequence(
        rows,
        batch_first=True,
        padding_value=tok.pad
    )

@torch.no_grad()
def build_spec_context(
    model,
    tok,
    X,
    *,
    spec_max_len,
    max_in_len,
    device=DEVICE,
):

    X = X.to(device)

    spec_ids = generate_with_validation_repair(
        model=model,
        tok=tok,
        X=X.to(device),
        agent_id=AGENT_SPECIFICATION,
        max_len=spec_max_len,
        validator=validate_spec_output,
        max_attempts=CFG.max_repair_attempts,
    )

    lens = [
        int((row != tok.pad).sum().item())
        for row in spec_ids
    ]

    print(
        f"[SPEC LEN] "
        f"avg={np.mean(lens):.1f} "
        f"min={min(lens)} "
        f"max={max(lens)}"
    )

    return build_spec_plus_anchor_context(
        tok,
        spec_ids,
        raw_x=None,
        max_in_len=max_in_len
    ).to(device)
# Diagnostic only. Do not use in normal pipeline evaluation/inference.

def _clean_spec_text(txt: str) -> str:
    # keep simple printable range; strip emojis/control chars
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", txt).strip()

def _postprocess_spec(txt: str) -> str:
    txt = _clean_spec_text(txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

def _is_noisy_gist(txt: str) -> bool:
    if not txt:
        return True

    txt = _clean_spec_text(txt).strip()
    words = txt.split()

    # Too short
    if len(words) < 8:
        return True

    # Too low alnum density
    alnum = sum(ch.isalnum() for ch in txt)
    if (alnum / max(len(txt), 1)) < 0.35:
        return True

    # Obvious code / patch leakage
    bad_markers = (
        "diff --git", "```", "@@", "+++", "---",
        "class ", "import ", "return ", "://",
        "/pytorch", "/prefect"
    )
    if any(b in txt for b in bad_markers):
        return True

    # Too many strange symbols
    nonword = re.sub(r"[A-Za-z0-9\s]", "", txt)
    if (len(nonword) / max(len(txt), 1)) >= 0.25:
        return True

    # Repetitive junk patterns
    junk_patterns = [
        "def (:",
        ": : :",
        "function function",
        "ation: :",
        "( :",
        "., .",
    ]
    if any(p in txt for p in junk_patterns):
        return True

    # For HumanEval-like specs, require at least one meaningful alphabetic span
    alpha_words = [w for w in words if re.search(r"[A-Za-z]{3,}", w)]
    if len(alpha_words) < 5:
        return True

    return False

def _is_noisy_spec_description(txt: str) -> bool:
    """
    Noise check for SPEC descriptions only.

    Unlike _is_noisy_gist(), this allows normal HumanEval words
    such as 'return', examples, function calls, and assertions.
    """

    if not txt:
        return True

    txt = _clean_spec_text(txt).strip()
    words = txt.split()

    if len(words) < 5:
        return True

    alnum = sum(ch.isalnum() for ch in txt)
    if (alnum / max(len(txt), 1)) < 0.30:
        return True

    bad_markers = (
        "diff --git",
        "```",
        "@@",
        "+++",
        "---",
        "/pytorch",
        "/prefect",
    )

    if any(b in txt for b in bad_markers):
        return True

    repeated = re.search(
        r"\b(\w+)(\s+\1){3,}\b",
        txt.lower()
    )

    if repeated:
        return True

    strange = re.search(
        r"(>>>>>>|<<<<<<|======|----delimited|to_to_to|oneoneone|intintint)",
        txt
    )

    if strange:
        return True

    alpha_words = [
        w for w in words
        if re.search(r"[A-Za-z]{3,}", w)
    ]

    if len(alpha_words) < 4:
        return True

    return False

def _decode_row_no_pad(tok: "SubwordTokenizer", row: torch.Tensor) -> str:
    ids = [int(t) for t in row.tolist() if int(t) != tok.pad]
    return tok.decode(ids)

# Helper to precompute agent-generated specification context inputs =======================
@torch.no_grad()
def build_agent_spec_context_inputs(
    model,
    tok,
    X,
    *,
    spec_max_len,
    max_in_len,
    device=DEVICE
):
    return build_spec_context(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        max_in_len=max_in_len,
        device=device
    )

def _unfreeze_decoder_tail(model: AgenticTransformerSeq2Seq, n_last_blocks: int = 1):
    # Unfreeze final N transformer decoder layers + all decoder LayerNorms
    if hasattr(model.decoder, "decoder"):
        # PyTorch TransformerDecoder with 'layers'
        layers = getattr(model.decoder.decoder, "layers", [])
    else:
        layers = []
    # Unfreeze norms everywhere in decoder
    for name, p in model.decoder.named_parameters():
        if "norm" in name:
            p.requires_grad = True
    # Unfreeze last N full blocks
    if layers:
        for bl in layers[-n_last_blocks:]:
            for p in bl.parameters():
                p.requires_grad = True

def freeze_spec_agent(model):

    for p in model.routing.agents[
        AGENT_SPECIFICATION
    ].parameters():

        p.requires_grad = False

# =========================
# Spec agent: evaluation
# =========================
@torch.no_grad()
def _eval_spec_ce_acc(
    model: "AgenticTransformerSeq2Seq",
    X: torch.Tensor,
    P: torch.Tensor,
    *,
    device: str = DEVICE
) -> Tuple[float, float]:
    """
    Teacher-forced CE/accuracy for the Spec agent on input X vs gold SPEC_DESC targets P.
    """
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    y_in, y_tgt = shift_targets(P.to(device))
    logits = model.forward_role(X.to(device), y_in, agent_id=AGENT_SPECIFICATION)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())
    return ce, acc

@torch.no_grad()
def debug_spec_generation_gap(
    model,
    tok,
    X,
    P,
    *,
    spec_max_len: int,
    n_samples: int = 3,
    device: str = DEVICE
):
    """
    Minimal debug:
    checks whether gold SPEC target is learned under teacher forcing,
    but autoregressive generation still fails to reproduce it.
    """

    model.to(device)
    model.eval()

    ce, acc = _eval_spec_ce_acc(
        model,
        X,
        P,
        device=device
    )

    print("\n--------------------------------------------------")
    print("[Agentic][DEBUG][SPEC GENERATION GAP]")
    print("--------------------------------------------------")
    print(f"Teacher-forced SPEC CE={ce:.3f} | tok_acc={acc:.3f}")

    n = min(n_samples, X.size(0))

    for i in range(n):

        pred = _generate_static(
            model,
            X[i:i+1].to(device),
            agent_id=AGENT_SPECIFICATION,
            max_len=spec_max_len,
            top_k=None,
            top_p=None,
            temperature=1.0,
            no_repeat_ngram_size=4,
            min_len=8,
        )

        prompt_txt = tok.decode([
            int(t)
            for t in X[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        gold_txt = tok.decode([
            int(t)
            for t in P[i].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        pred_txt_raw = tok.decode([
            int(t)
            for t in pred[0].tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ])

        pred_txt_fixed = force_signature_from_prompt(
            pred_txt_raw,
            prompt_txt
        )

        gold_tokens = set(tokenize_lexical_overlap_text(gold_txt))
        pred_tokens = set(tokenize_lexical_overlap_text(pred_txt_fixed))

        overlap = (
            len(gold_tokens & pred_tokens)
            / max(len(gold_tokens), 1)
        )

        hit_eos = EOS in pred[0].tolist()
        raw_len = int((pred[0] != tok.pad).sum().item())

        print(f"\n[Spec Debug Sample {i}]")
        print(f"generation_len={raw_len} | hit_eos={hit_eos} | gold_pred_overlap={overlap:.3f}")
        print("[GOLD SPEC]")
        print(gold_txt[:700])
        print("\n[GENERATED SPEC - SIGNATURE FIXED]")
        print(pred_txt_fixed[:700])

def run_all(cfg: Config = CFG):
    set_seed(cfg.seed)

    # ==========================================================
    # Data
    # ==========================================================

    data = HumanEvalData(
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
    )

    ids, X, Y, P = data.as_tensors_with_spec_targets(
        spec_max_len=cfg.max_in_len
    )

    for i in range(min(3, len(data.samples))):
        print("\n====================")
        print("PROMPT")
        print(data.samples[i][1][:1000])

        print("\nSPEC TARGET")
        print(
            data.tok.decode([
                int(t)
                for t in P[i].tolist()
                if int(t) not in (
                    data.tok.pad,
                    data.tok.bos,
                    data.tok.eos,
                )
            ])
        )

    # ==========================================================
    # Deterministic split
    # ==========================================================

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
    P = P[perm]

    # ==========================================================
    # Deployment / Adaptation split
    # ==========================================================

    ADAPT_HOLDOUT = 33

    deploy_N = N - ADAPT_HOLDOUT

    X_deploy = X[:deploy_N]
    Y_deploy = Y[:deploy_N]
    P_deploy = P[:deploy_N]

    X_adapt = X[deploy_N:]
    Y_adapt = Y[deploy_N:]
    P_adapt = P[deploy_N:]

    split = int(deploy_N * 0.8)

    X_train = X_deploy[:split]
    X_test  = X_deploy[split:]

    Y_train = Y_deploy[:split]
    Y_test  = Y_deploy[split:]

    P_train = P_deploy[:split]
    P_test  = P_deploy[split:]

    print(
        f"[Info] DeployTrain={split} "
        f"DeployTest={deploy_N - split} "
        f"AdaptHoldout={ADAPT_HOLDOUT}"
    )

    # ==========================================================
    # Model
    # ==========================================================

    max_len_for_model = max(
        cfg.max_len_cap,
        cfg.max_in_len
        + max(
            cfg.max_out_len,
            cfg.spec_decode_len,
            cfg.impl_decode_len,
        )
        + 8,
    )

    model = AgenticTransformerSeq2Seq(
        vocab_size=data.tok.vocab_size,
        n_agents=cfg.n_agents,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_heads,
        n_layers_enc=cfg.n_layers_enc,
        n_layers_dec=cfg.n_layers_dec,
        max_len=max_len_for_model,
        pad_idx=data.tok.pad,
    )

    # ==========================================================
    # Initialization diagnostics
    # ==========================================================

    print("\n[Initialization Diagnostic]")

    debug_spec_generation_gap(
        model,
        data.tok,
        X_test,
        P_test,
        spec_max_len=cfg.spec_decode_len,
        n_samples=3,
        device=DEVICE,
    )

    spec_validity = evaluate_spec_generation_validity(
        model,
        data.tok,
        X_test,
        spec_max_len=cfg.spec_decode_len,
        device=DEVICE,
        n_samples=cfg.spec_gate_samples,
    )

    if spec_validity < cfg.spec_validity_threshold:
        print(
            f"[WARN] Fallback-free SPEC validity is low: "
            f"{spec_validity:.4f} < {cfg.spec_validity_threshold:.4f}. "
            f"Continuing with validation repair/fallback for pipeline stability."
        )

    print("\n[Fallback-Free SPEC Samples Before Training]")

    for i in range(min(3, X_test.size(0))):
        spec_rows = generate_valid_spec_rows(
            model,
            data.tok,
            X_test[i:i + 1].to(DEVICE),
            spec_max_len=cfg.spec_decode_len,
            device=DEVICE,
            fallback_to_prompt_docstring=False,
        )

        spec_txt = data.tok.decode([
            int(t)
            for t in spec_rows[0].tolist()
            if int(t) not in (
                data.tok.pad,
                data.tok.bos,
                data.tok.eos,
            )
        ])

        print("\n====================")
        print(f"[INIT SPEC SAMPLE {i}]")
        print(spec_txt[:1000])

    # ==========================================================
    # Stage 0 — SPEC supervision
    # Prompt -> Specification
    # ==========================================================

    print(
        "\n[Agentic][Training] "
        "Stage 0: SPEC supervision"
    )

    train_spec_supervised(
        model,
        X_train,
        P_train,
        epochs=cfg.ft_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=DEVICE,
        unfreeze_backbone=True,
        unfreeze_A_adapter=True,
        unfreeze_dec_norms=True,
    )

    print(
        "\n[Stage 0 Diagnostic] "
        "SPEC generation after supervision"
    )

    debug_spec_generation_gap(
        model,
        data.tok,
        X_test,
        P_test,
        spec_max_len=cfg.spec_decode_len,
        n_samples=3,
        device=DEVICE,
    )

    spec_validity = evaluate_spec_generation_validity(
        model,
        data.tok,
        X_test,
        spec_max_len=cfg.spec_decode_len,
        device=DEVICE,
        n_samples=cfg.spec_gate_samples,
    )

    print(
        f"[Stage 0] SPEC validity={spec_validity:.4f}"
    )

    # ==========================================================
    # Stage 1 — IMPL training from generated SPEC
    #
    # Important:
    # - SPEC agent remains frozen.
    # - Shared encoder/decoder remain frozen.
    # - Only IMPL agent adapter/head are trained.
    # ==========================================================

    print(
        "\n[Agentic][Training] "
        "Stage 1: Generated SPEC -> IMPL"
    )

    train_stage1_interleaved(
        model,
        X_train,
        Y_train,
        P_train,
        tok=data.tok,
        spec_max_len=cfg.spec_decode_len,
        epochs=cfg.pipe_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=DEVICE,
        unfreeze_backbone=False,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=False,
        max_in_len=cfg.max_in_len,
    )

    # ==========================================================
    # Save deployed checkpoint BEFORE post-deployment adaptation
    # ==========================================================

    save_deployed_checkpoint(
        cfg.deployed_checkpoint,
        model,
        cfg,
        data,
        meta={
            "meaning": "HumanEval deployed checkpoint after Stage 0 SPEC training and Stage 1 IMPL training",
            "train_size": int(X_train.size(0)),
            "test_size": int(X_test.size(0)),
            "adapt_size": int(X_adapt.size(0)),
            "stage": "after_stage1_before_stage2_adaptation",
        },
    )

    print(
        "\n[Stage 1 Diagnostic] "
        "SPEC generation after IMPL training"
    )

    debug_spec_generation_gap(
        model,
        data.tok,
        X_test,
        P_test,
        spec_max_len=cfg.spec_decode_len,
        n_samples=3,
        device=DEVICE,
    )

    print("\n[Probe] SPEC generation on TRAIN samples after Stage 1")

    for i in range(min(3, X_train.size(0))):
        pred = generate_valid_spec_rows(
            model,
            data.tok,
            X_train[i:i + 1].to(DEVICE),
            spec_max_len=cfg.spec_decode_len,
            device=DEVICE,
            fallback_to_prompt_docstring=True,
        )

        gold_txt = data.tok.decode([
            int(t)
            for t in P_train[i].tolist()
            if int(t) not in (
                data.tok.pad,
                data.tok.bos,
                data.tok.eos,
            )
        ])

        pred_txt = data.tok.decode([
            int(t)
            for t in pred[0].tolist()
            if int(t) not in (
                data.tok.pad,
                data.tok.bos,
                data.tok.eos,
            )
        ])

        prompt_txt = data.tok.decode([
            int(t)
            for t in X_train[i].tolist()
            if int(t) not in (
                data.tok.pad,
                data.tok.bos,
                data.tok.eos,
            )
        ])

        gold_sig = extract_prompt_anchor(prompt_txt)
        pred_sig = extract_prompt_anchor(pred_txt)

        print(f"[SIG MATCH] {gold_sig == pred_sig}")
        print("\n====================")
        print("[GOLD SPEC]")
        print(gold_txt[:1000])
        print("\n[PRED SPEC]")
        print(pred_txt[:1000])

    # ==========================================================
    # Pipeline evaluation after Stage 1
    # ==========================================================

    print("\n[Pipeline After Stage 1]")

    eval_pipeline(
        model,
        data.tok,
        X_test,
        Y_test,
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    print("\n[Agentic][Eval][SPEC-STATS] Analysis agent quick stats")

    stats = spec_analysis_stats(
        model,
        data.tok,
        X_test,
        spec_max_len=cfg.spec_decode_len,
        device=DEVICE,
    )

    print(stats)

    print("\n[Agentic][Inference][PIPELINE-SAMPLES] Spec↔Patch examples")

    spec_to_impl_alignment_sample(
        model,
        data.tok,
        X_test,
        spec_max_len=cfg.spec_decode_len,
        out_max_len=cfg.impl_decode_len,
        max_in_len=cfg.max_in_len,
        k=3,
        device=DEVICE,
    )

    # ==========================================================
    # DEPLOYED MODEL CHECKPOINT
    # ==========================================================

    print(
        "\n[Deployment] Stage 0 + Stage 1 complete. "
        "Checkpoint already saved. "
        "Skipping post-deployment adaptation."
    )

    return model, data, (ids, X, Y, P)

    # ==========================================================
    # Local SPEC eval helper
    # ==========================================================

    @torch.no_grad()
    def _eval_spec_ce_acc_local(
        m: AgenticTransformerSeq2Seq,
        Xenc: torch.Tensor,
        Ptg: torch.Tensor,
        *,
        device: str,
    ):
        m.to(device)
        m.eval()

        loss_fn = SeqCELoss(
            pad_idx=m.pad_idx
        )

        y_in, y_tgt = shift_targets(
            Ptg.to(device)
        )

        logits = m.forward_role(
            Xenc.to(device),
            y_in,
            agent_id=AGENT_SPECIFICATION,
        )

        ce = float(
            loss_fn(
                logits,
                y_tgt,
            ).item()
        )

        preds = logits.argmax(-1)
        mask = y_tgt != m.pad_idx

        acc = float(
            (
                ((preds == y_tgt) & mask)
                .float()
                .sum()
                / mask.float().sum().clamp_min(1.0)
            ).item()
        )

        return ce, acc

    # ==========================================================
    # Stage 2A — SPEC specialization
    # ==========================================================

    print(
        "\n[Agentic][Training] "
        "Stage 2A: Static specialization for SPEC agent "
        "(freeze backbone + IMPL agent; train SPEC agent on X -> P)"
    )

    iss_ce_before, iss_acc_before = _eval_spec_ce_acc_local(
        model,
        X_test,
        P_test,
        device=DEVICE,
    )

    print(
        f"[Agentic][Eval][SPEC@Before FT] "
        f"CE={iss_ce_before:.3f} | tok_acc={iss_acc_before:.3f}"
    )

    print("\n[Pipeline Before SPEC FT]")

    pipe_ce_before, pipe_acc_before = eval_pipeline(
        model,
        data.tok,
        X_test,
        Y_test,
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    fine_tune_static(
        model,
        X_adapt,
        Y_adapt,
        user_id=AGENT_SPECIFICATION,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=0,
        idxs=None,
        device=DEVICE,
        tok=data.tok,
        P=P_adapt,
        gist_ctx_fn=None,
        max_in_len=cfg.max_in_len,
        patience=5,
    )

    iss_ce_after, iss_acc_after = _eval_spec_ce_acc_local(
        model,
        X_test,
        P_test,
        device=DEVICE,
    )

    print(
        f"[Agentic][Eval][SPEC@After FT]"
        f"CE={iss_ce_after:.3f} | tok_acc={iss_acc_after:.3f} "
        f"| ΔCE={iss_ce_after - iss_ce_before:+.3f} "
        f"({(iss_ce_after - iss_ce_before) / max(abs(iss_ce_before), 1e-8) * 100:+.2f}%) "
        f"| Δacc={iss_acc_after - iss_acc_before:+.3f} "
        f"({(iss_acc_after - iss_acc_before) / max(abs(iss_acc_before), 1e-8) * 100:+.2f}%)"
    )

    print("\n[Fallback-Free SPEC Samples After SPEC FT]")

    for i in range(min(3, X_test.size(0))):
        spec_rows = generate_valid_spec_rows(
            model,
            data.tok,
            X_test[i:i + 1].to(DEVICE),
            spec_max_len=cfg.spec_decode_len,
            device=DEVICE,
            fallback_to_prompt_docstring=False,
        )

        spec_txt = data.tok.decode([
            int(t)
            for t in spec_rows[0].tolist()
            if int(t) not in (
                data.tok.pad,
                data.tok.bos,
                data.tok.eos,
            )
        ])

        print("\n====================")
        print(f"[POST-FT SPEC SAMPLE {i}]")
        print(spec_txt[:1000])

    print_agentic_efficiency_stats(
        model,
        stage_name="SPEC-FT",
        active_agent_id=AGENT_SPECIFICATION,
    )

    freeze_spec_agent(model)

    print("\n[Pipeline After SPEC FT]")

    pipe_ce_after, pipe_acc_after = eval_pipeline(
        model,
        data.tok,
        X_test,
        Y_test,
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    print(
        f"[SPEC Evolution Impact] "
        f"ΔCE={pipe_ce_after - pipe_ce_before:+.3f} "
        f"| ΔACC={pipe_acc_after - pipe_acc_before:+.3f}"
    )

    # ==========================================================
    # Stage 2B — IMPL specialization
    # ==========================================================

    print(
        "\n[Agentic][Training] "
        "Stage 2B: Static specialization for IMPL agent "
        "(freeze backbone + SPEC agent; train IMPL agent on generated SPEC input)"
    )

    spec_len = cfg.spec_decode_len

    X_test_impl = build_agent_spec_context_inputs(
        model,
        data.tok,
        X_test,
        spec_max_len=spec_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    ce_before, acc_before = _eval_impl_ce_acc(
        model,
        X_test_impl,
        Y_test,
        device=DEVICE,
    )

    print(
        f"[Agentic][Eval][IMPL-ONLY ADAPTATION][Before FT] "
        f"CE={ce_before:.3f} | tok_acc={acc_before:.3f}"
    )

    fine_tune_static(
        model,
        X_adapt,
        Y_adapt,
        user_id=AGENT_IMPLEMENTATION,
        epochs=3,
        batch_size=cfg.ft_batch,
        lr=3e-5,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=0,
        device=DEVICE,
        tok=data.tok,
        gist_ctx_fn=lambda xb: build_agent_spec_context_inputs(
            model,
            data.tok,
            xb,
            spec_max_len=spec_len,
            max_in_len=cfg.max_in_len,
            device=DEVICE,
        ),
        X_gist=None,
        max_in_len=cfg.max_in_len,
        patience=2,
    )

    X_test_impl_after = build_agent_spec_context_inputs(
        model,
        data.tok,
        X_test,
        spec_max_len=spec_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    ce_after, acc_after = _eval_impl_ce_acc(
        model,
        X_test_impl_after,
        Y_test,
        device=DEVICE,
    )

    print(
        f"[Agentic][Eval][IMPL-ONLY ADAPTATION][After FT] "
        f"CE={ce_after:.3f} | tok_acc={acc_after:.3f} "
        f"| ΔCE={ce_after - ce_before:+.3f} "
        f"| Δacc={acc_after - acc_before:+.3f}"
    )

    print("\n[Stage2 Diagnostic]")

    eval_pipeline(
        model,
        data.tok,
        X_test,
        Y_test,
        spec_max_len=cfg.spec_decode_len,
        max_in_len=cfg.max_in_len,
        device=DEVICE,
    )

    print_agentic_efficiency_stats(
        model,
        stage_name="IMPL-FT",
        active_agent_id=AGENT_IMPLEMENTATION,
    )

    # ==========================================================
    # Runtime inference
    # ==========================================================

    print("\n[Agentic][Inference] Generating validated samples")

    generate_validated_samples(
        model,
        data.tok,
        ids[split:],
        X_test,
        output_dir=cfg.out_dir,
        sample_prefix="humaneval",
        spec_max_len=cfg.spec_decode_len,
        out_max_len=cfg.impl_decode_len,
        max_in_len=cfg.max_in_len,
        n_samples=cfg.n_validation_samples,
        max_repair_attempts=cfg.max_repair_attempts,
        device=DEVICE,
    )

    return model, data, (ids, X, Y, P)
# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    print("[Main] Starting HumanEval experiment...")

    run_all()    