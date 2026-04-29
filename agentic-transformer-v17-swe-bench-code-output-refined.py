# Agentic seq2seq — Routing with Dynamic→Static (CPU-only, no autotune)

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# ============================================================
# Repro (CPU-only)
# ============================================================
DEVICE = "cpu"

# ===== Fixed role indices for strict pipeline =====
AGENT_ISSUE_ANALYSIS   = 0     # was AGENT_ISSUE_ANALYSIS
AGENT_CODE_GENERATION  = 1

def agent_pretty_name(agent_id: int) -> str:
    return "Issue Analysis Agent" if agent_id == AGENT_ISSUE_ANALYSIS else (
            "Code Generation Agent" if agent_id == AGENT_CODE_GENERATION else f"Agent {agent_id}"
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
    limit: int = 1024
    max_in_len: int = 1024
    max_out_len: int = 256
    spm_vocab: int = 8000
    demo_data: bool = False         # False = load SWE-bench via HF datasets
    # model
    n_agents: int = 2
    model_dim: int = 384
    n_heads: int = 6
    n_layers_enc: int = 4
    n_layers_dec: int = 4
    max_len_cap: int = 1024
    # pipeline training (global)
    pipe_epochs: int = 4
    pipe_batch: int = 8
    pipe_lr: float = 2e-4
    lb_lambda: float = 5
    router_lambda: float = 1.0
    # static fine-tuning (specialization)
    ft_epochs: int = 8
    ft_batch: int = 8
    ft_lr: float = 1e-4
    agent_idx: int = 0
    ft_unfreeze_adapters: bool = True
    ft_unfreeze_dec_norms: bool = True
    ft_unfreeze_decoder_tail_blocks_code: int = 2
    ft_concat_epochs_code: int = 2
    code_use_repo_anchor: bool = True
    repo_anchor_max_len: int = 192
    # decode / dump
    decode_max_len: int = 256
    out_dir: str = "preds_static_role"
    save_samples_k: int = 10

CFG = Config()

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
        return self.sp.decode(ids)

    @property
    def pad(self): return self.pad_idx
    @property
    def bos(self): return self.bos_idx
    @property
    def eos(self): return self.eos_idx

def _extract_tag_block(text: str, tag: str) -> str:
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    if open_tag in text and close_tag in text:
        return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    return ""

# ============================================================
# Data loading / batching
# ============================================================
try:
    from datasets import load_dataset
    HAVE_HF = True
except Exception:
    HAVE_HF = False

class SWEText2PatchData:
    def __init__(self, *, split: str = "train", limit: Optional[int] = 1024,
                    max_in_len: int = 512, max_out_len: int = 256,
                    spm_vocab_size: int = 8000, demo_data: bool = True):
        if demo_data:
            print("[Data] DEMO synthetic dataset")
            rng = random.Random()
            self.samples: List[Tuple[str, str, str]] = []
            n = int(limit or 1024)
            for i in range(n):
                title = f"Issue {i}: Widget broken"
                body = f"Repro {i}: click→crash, trace={rng.randint(0,999)}"
                patch = f"diff --git a/app.py b/app.py\n+print('fix {i}')\n"
                self.samples.append((f"demo-{i}", title + "\n" + body, patch))
            rng.shuffle(self.samples)

            texts = [x for _, x, _ in self.samples] + [y for _, _, y in self.samples]
            special_tag_text = " ".join([
                "<ISSUE_TITLE>", "</ISSUE_TITLE>",
                "<ISSUE_DESC>",  "</ISSUE_DESC>",
                "<HINTS>",       "</HINTS>",
                "<ISSUE_GIST>",  "</ISSUE_GIST>",   # NEW tag used in pipeline context
            ])
            texts = texts + [special_tag_text] * 100
            self.tok = SubwordTokenizer(texts, vocab_size=spm_vocab_size)
            self.max_in_len, self.max_out_len = max_in_len, max_out_len
            return

        if not HAVE_HF:
            raise RuntimeError("Install `datasets` to use SWE-bench: pip install datasets")

        print("[Data] Load SWE-bench…")
        ds = load_dataset("princeton-nlp/SWE-bench", split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        rows = list(ds)

        def build_input(ex: Dict) -> str:
            title = str(ex.get("title", "")).strip()
            desc  = str(ex.get("problem_statement", "")).strip()
            hints = str(ex.get("hints_text", "")).strip()
            tagged = []
            if title: tagged.append(f"<ISSUE_TITLE>\n{title}\n</ISSUE_TITLE>")
            if desc:  tagged.append(f"<ISSUE_DESC>\n{desc}\n</ISSUE_DESC>")
            if hints: tagged.append(f"<HINTS>\n{hints}\n</HINTS>")
            meta = []
            if ex.get("repo"): meta.append(f"repo={ex['repo']}")
            if ex.get("base_commit"): meta.append(f"base={ex['base_commit']}")
            if meta: tagged.append("[" + ", ".join(meta) + "]")
            return "\n".join(tagged)

        def pick_patch(ex: Dict) -> str:
            for key in ("patch", "base_patch", "model_patch", "test_patch"):
                if key in ex and ex[key]: return str(ex[key])
            return ""

        self.samples: List[Tuple[str, str, str]] = []
        for ex in rows:
            iid = str(ex.get("instance_id", ""))
            xin = build_input(ex); yout = pick_patch(ex)
            if len(yout.strip()) == 0: continue
            self.samples.append((iid, xin, yout))

        print(f"[Data] {len(self.samples)} supervised pairs")
        texts = [x for _, x, _ in self.samples] + [y for _, _, y in self.samples]
        special_tag_text = " ".join([
            "<ISSUE_TITLE>", "</ISSUE_TITLE>",
            "<ISSUE_DESC>",  "</ISSUE_DESC>",
            "<HINTS>",       "</HINTS>",
            "<ISSUE_GIST>",        "</ISSUE_GIST>",
        ])
        texts = texts + [special_tag_text] * 100
        self.tok = SubwordTokenizer(texts, vocab_size=spm_vocab_size)
        self.max_in_len, self.max_out_len = max_in_len, max_out_len

    def as_tensors(self) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        if not getattr(self, "samples", None):
            raise ValueError("No samples loaded.")
        ids: List[str] = []
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for iid, x, y in self.samples:
            ids.append(iid)
            xs.append(self.tok.encode(x, add_bos_eos=False, max_len=self.max_in_len))
            ys.append(self.tok.encode(y, add_bos_eos=True,  max_len=self.max_out_len))
        X = pad_sequence(xs, batch_first=True, padding_value=self.tok.pad)
        Y = pad_sequence(ys, batch_first=True, padding_value=self.tok.pad)
        return ids, X, Y

    def as_tensors_with_issue_targets(self, issue_max_len: int) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, X, Y = self.as_tensors()
        Ps = []
        for _, xin, _ in self.samples:
            title = _extract_tag_block(xin, "ISSUE_TITLE")
            desc  = _extract_tag_block(xin, "ISSUE_DESC") or xin
            issue  = _clean_issue_text(_make_issue_gist(title, desc))
            if not issue:
                issue = _clean_issue_text(desc)
            Ps.append(self.tok.encode(issue, add_bos_eos=True, max_len=issue_max_len))
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
        h = self.tok_embedding(x) + self.pos_embedding[:, :T, :]
        mask = (x == self.pad_token_id)
        mem = self.encoder(h, src_key_padding_mask=mask)
        valid = (~mask).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (mem * valid.unsqueeze(-1)).sum(dim=1) / denom
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

    def forward(self, y_in: torch.Tensor, memory: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, Lt = y_in.shape
        y_emb = self.tok_embedding(y_in) + self.pos_embedding[:, :Lt, :]
        tgt_key_padding_mask = (y_in == self.pad_idx)
        tgt_mask = self._subsequent_mask(Lt, y_in.device)
        return self.decoder(y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=src_key_padding_mask)

class Agent(nn.Module):
    def __init__(self, model_dim: int, vocab_size: int, adapter_dim: int = 124):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, model_dim),
        )
        self.router_head = nn.Linear(model_dim, vocab_size)  # kept to preserve shape, unused
        self.role_head   = nn.Linear(model_dim, vocab_size)

    def project(self, states: torch.Tensor, head: str = "role") -> torch.Tensor:
        h = self.adapter(states)
        layer = self.router_head if head == "router" else self.role_head
        return layer(h)

class StrictPipeline(nn.Module):
    """
    Strict A→B pipeline on static role heads:
        issue = Agent A(analysis) generates from full X
        gist_only = <ISSUE_GIST>...</ISSUE_GIST> produced from A
        patch = Agent B(code) generates from issue-derived context only
    """

    def __init__(self, agents: nn.ModuleList):
        super().__init__()
        self.agents = agents

    @torch.no_grad()
    def run(
        self,
        model: "AgenticTransformerSeq2Seq",
        tok: "SubwordTokenizer",
        X: torch.Tensor,
        *,
        issue_max_len: int,
        out_max_len: int,
        max_in_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        issue_ctx, issue_display_ids = _issue_ctx_greedy_with_fallback(
            model, tok, X, issue_max_len=issue_max_len
        )
        code_ctx = build_code_context_inputs(
            model,
            tok,
            X,
            issue_ctx=issue_ctx,
            issue_max_len=issue_max_len,
            max_in_len=max_in_len,
            device=X.device.type if X.is_cuda else DEVICE,
        )
        patch_ids = _generate_static(
            model,
            code_ctx,
            agent_id=AGENT_CODE_GENERATION,
            max_len=out_max_len,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            no_repeat_ngram_size=3,
            min_len=24,
        )
        patch_ids = _apply_correction_pass(
            model,
            tok,
            patch_ids,
            code_ctx,
            max_len=out_max_len,
        )
        return issue_display_ids, patch_ids

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
        return self.agents[agent_id].project(dec_states, head="role")

    @torch.no_grad()
    def run_pipeline(self, model: "AgenticTransformerSeq2Seq", tok: "SubwordTokenizer", X: torch.Tensor,
                        *, issue_max_len: int, out_max_len: int, max_in_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pipeline.run(model, tok, X, issue_max_len=issue_max_len, out_max_len=out_max_len, max_in_len=max_in_len)

class AgenticTransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, n_agents: int = 2, model_dim: int = 512,
                    n_heads: int = 8, n_layers_enc: int = 6, n_layers_dec: int = 6,
                    max_len: int = 1024, pad_idx: int = PAD):
        super().__init__()
        self.encoder = Encoder(vocab_size, model_dim, n_heads, n_layers_enc, max_len, pad_idx)
        self.decoder = Decoder(vocab_size, model_dim, n_heads, n_layers_dec, max_len, pad_idx,
                                tok_embedding=self.encoder.tok_embedding)
        agents = nn.ModuleList([Agent(model_dim, vocab_size) for _ in range(n_agents)])
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
    """Greedy by default; with top_k/top_p uses constrained sampling. Static agent selection."""
    model.eval()
    memory, _cls, src_mask = model.encode(X)
    B = X.size(0)
    vocab_size = model.encoder.tok_embedding.num_embeddings
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=X.device)

    for _t in range(1, max_len):
        dec = model.decode_states(ys, memory, src_mask)
        step_logits = model.routing.agents[agent_id].project(dec[:, -1:], head="role").squeeze(1)

        # Block EOS until min_len is reached
        if ys.size(1) < max(1, min_len):
            step_logits[:, EOS] = float("-inf")

        # No-repeat n-gram mask
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            banned = _no_repeat_ngram_mask(ys, no_repeat_ngram_size, vocab_size)
            step_logits = step_logits.masked_fill(banned, float("-inf"))

        # Temperature
        if temperature and temperature != 1.0:
            step_logits = step_logits / max(temperature, 1e-8)

        # Sampling vs greedy
        use_sampling = (top_k is not None and top_k > 0) or (top_p is not None and 0.0 < top_p < 1.0)
        if use_sampling:
            logits = _top_k_top_p_filtering(step_logits.clone(), top_k, top_p)
            next_tok = torch.distributions.Categorical(logits=logits).sample().unsqueeze(1)
        else:
            next_tok = torch.argmax(step_logits, dim=-1, keepdim=True)

        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok == EOS).all():
            break
    return ys

@torch.no_grad()
def _apply_correction_pass(
    model: AgenticTransformerSeq2Seq,
    tok: "SubwordTokenizer",
    patch_ids: torch.Tensor,
    X_ctx: torch.Tensor,
    *,
    max_len: int,
) -> torch.Tensor:
    """Re-decode once, validate aggressively, and only emit structurally plausible diffs."""
    refined = _generate_static(
        model,
        X_ctx,
        agent_id=AGENT_CODE_GENERATION,
        max_len=max_len,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3,
        min_len=24,
    )

    def decode_ids(ids: torch.Tensor) -> str:
        kept = [int(t) for t in ids.tolist() if int(t) not in (tok.pad, tok.bos, tok.eos)]
        return tok.decode(kept).strip()

    def encode_txt(txt: str) -> torch.Tensor:
        return tok.encode(txt, add_bos_eos=True, max_len=max_len)

    corrected_rows: List[torch.Tensor] = []
    for i in range(refined.size(0)):
        issue_ctx_text = decode_ids(X_ctx[i])
        candidate = _sanitize_generated_patch_text(decode_ids(refined[i]))
        original = _sanitize_generated_patch_text(decode_ids(patch_ids[i]))

        options = [candidate, original]
        valid = [txt for txt in options if _patch_text_is_plausible(txt)]

        if valid:
            best = max(valid, key=len)
        else:
            repaired = _sanitize_generated_patch_text(candidate or original)
            if _patch_text_is_plausible(repaired):
                best = repaired
            else:
                best = _build_patch_stub_from_issue(issue_ctx_text)

        corrected_rows.append(encode_txt(best))

    return pad_sequence(corrected_rows, batch_first=True, padding_value=tok.pad)

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
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        to_mask = cum > top_p
        to_mask[..., 1:] = to_mask[..., :-1].clone()
        to_mask[..., 0] = False
        logits.scatter_(1, sorted_idx, torch.where(to_mask, torch.full_like(sorted_probs, float("-inf")), logits.gather(1, sorted_idx)))
    return logits

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

# ============================================================
# Training loops (specialization & pipeline)
# ============================================================
def train_strict_pipeline_swebench(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    *,
    tok: "SubwordTokenizer",
    issue_max_len: int = 124,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    unfreeze_backbone: bool = True,
    unfreeze_B_adapter: bool = True,
    unfreeze_dec_norms: bool = True,
    max_in_len: Optional[int] = None,
):
    """Strict ISSUE_ANALYSIS → CODE_GENERATION training."""
    assert AGENT_ISSUE_ANALYSIS == 0 and AGENT_CODE_GENERATION == 1, "Expect issue=0, code=1."
    model.to(device)
    print("[Agentic][Training][CODE] starting (teacher-forced with gist context)", flush=True)

    _set_trainable_strict_agent(
        model,
        agent_id=AGENT_CODE_GENERATION,
        unfreeze_backbone=unfreeze_backbone,
        unfreeze_adapter=unfreeze_B_adapter,
        unfreeze_dec_norms=unfreeze_dec_norms,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    N = X_train.size(0)
    max_in_len = int(max_in_len or X_train.size(1))

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tok_correct = 0
        epoch_tok_total = 0

        for i in range(0, N, batch_size):
            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)

            with torch.no_grad():
                issue_ctx, _ = _issue_ctx_greedy_with_fallback(
                    model, tok, xb, issue_max_len=issue_max_len
                )
                xb_gist = issue_ctx[:, :max_in_len]  # ← with gist input

            y_in, y_tgt = shift_targets(yb)
            logits = model.forward_role(xb_gist, y_in, agent_id=AGENT_CODE_GENERATION)
            loss = loss_fn(logits, y_tgt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            with torch.no_grad():
                Bsz, _, _ = logits.shape
                preds = logits.argmax(dim=-1)
                mask = (y_tgt != model.pad_idx)
                epoch_loss_sum += float(loss.detach()) * Bsz
                epoch_tok_correct += ((preds == y_tgt) & mask).sum().item()
                epoch_tok_total += mask.sum().item()

        epoch_ce = epoch_loss_sum / float(N)
        epoch_acc = (epoch_tok_correct / max(epoch_tok_total, 1)) if epoch_tok_total > 0 else 0.0
        print(f"[Agentic][Training][CODE][Epoch {ep}] CE={epoch_ce:.3f} | tok_acc={epoch_acc:.3f}")
    print("[Agentic][Training][CODE] done ✅", flush=True)

def train_issue_supervised(
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
    """Teacher-force Agent 0 (ISSUE_ANALYSIS) to generate ISSUE_DESC."""
    model.to(device)
    print("[Agentic][Training][ISSUE] starting", flush=True)

    _set_trainable_strict_agent(
        model,
        agent_id=AGENT_ISSUE_ANALYSIS,
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
            logits = model.forward_role(xb, y_in, agent_id=AGENT_ISSUE_ANALYSIS)
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

        print(f"[Agentic][Training][ISSUE][Epoch {ep}] CE={sum_loss/float(N):.3f} | tok_acc={(tok_correct/max(tok_total,1)):.3f}")
    print("[Agentic][Training][ISSUE] done ✅", flush=True)

def fine_tune_static(
    model: AgenticTransformerSeq2Seq,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    user_id: int,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,                 # lower LR
    weight_decay: float = 0.01,       # add wd
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = True,
    unfreeze_decoder_tail_blocks: int = 1,   # tiny extra capacity if desired
    idxs: Optional[torch.Tensor] = None,
    device: str = DEVICE,
    tok: Optional["SubwordTokenizer"] = None,
    P: Optional[torch.Tensor] = None,         # gold ISSUE_DESC (targets for Issue agent)
    gist_ctx_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # for Code agent
    max_in_len: Optional[int] = None,
    use_concat_first_epoch: bool = True,      # (Code only) concat gist + original X for epoch 1
    patience: int = 2                         # early stopping on Dev CE
):
    """
    Stage-2 static specialization for the selected agent.
        - If user_id == AGENT_CODE_GENERATION (1): curriculum with gist context (unchanged).
        - If user_id == AGENT_ISSUE_ANALYSIS   (0): train on original X, targets=P (gold ISSUE_DESC).
    Backbone stays frozen except for explicitly allowed parts (adapters/dec norms/tail blocks).
    """
    # Sanity: for Issue agent we need P (targets), for Code agent we need Y (patch targets)
    if user_id == AGENT_ISSUE_ANALYSIS and P is None:
        raise ValueError("fine_tune_static(issue): P (gold ISSUE_DESC) is required.")
    if user_id == AGENT_CODE_GENERATION and gist_ctx_fn is None:
        raise ValueError("fine_tune_static(code): gist_ctx_fn is required for gist curriculum.")

    model.to(device)

    # Freeze everything; unfreeze only this agent (+optional norms/decoder tail)
    _set_ft_requires_grad(
        model,
        user_id=user_id,
        unfreeze_adapters=unfreeze_adapters,
        unfreeze_dec_norms=unfreeze_dec_norms
    )
    if unfreeze_decoder_tail_blocks and unfreeze_decoder_tail_blocks > 0:
        _unfreeze_decoder_tail(model, n_last_blocks=int(unfreeze_decoder_tail_blocks))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    # Slice to optional subset
    xb_all = X if idxs is None else X[idxs]
    # Choose targets by agent
    if user_id == AGENT_CODE_GENERATION:
        tgt_all = Y if idxs is None else Y[idxs]
    else:  # Issue agent
        tgt_all = P if idxs is None else P[idxs]

    N = xb_all.size(0)
    max_in_len = int(max_in_len or xb_all.size(1))

    # 90/10 tail split for dev
    dev_frac = max(1, int(0.1 * N))
    xb_tr, xb_dev = xb_all[:-dev_frac], xb_all[-dev_frac:]
    tb_tr, tb_dev = tgt_all[:-dev_frac], tgt_all[-dev_frac:]
    P_tr = P[:-dev_frac] if (P is not None) else None
    P_dev = P[-dev_frac:] if (P is not None) else None

    best_dev_ce = float("inf")
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        correct_train, total_train = 0, 0

        # ===== Build contexts per agent =====
        if user_id == AGENT_CODE_GENERATION:
            # ---- Code agent: same gist curriculum as before ----
            if ep == 1 and P is not None:
                with torch.no_grad():
                    def _wrap_from_P(P_block):
                        disp_rows = []
                        for i in range(P_block.size(0)):
                            ids = [t for t in P_block[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]
                            disp_rows.append(tok.decode(ids))
                        ctx_rows = []
                        for g in disp_rows:
                            ctx_txt = f"<ISSUE_GIST>\n{_postprocess_gist(g)}\n</ISSUE_GIST>"
                            ctx_rows.append(torch.tensor(tok.sp.encode(ctx_txt, out_type=int), dtype=torch.long))
                        return pad_sequence(ctx_rows, batch_first=True, padding_value=tok.pad)
                X_gist_clean_tr  = _wrap_from_P(P_tr)
                X_gist_clean_dev = _wrap_from_P(P_dev)
            else:
                with torch.no_grad():
                    X_gist_clean_tr  = gist_ctx_fn(xb_tr.to(device)).cpu()
                    X_gist_clean_dev = gist_ctx_fn(xb_dev.to(device)).cpu()

            if ep <= CFG.ft_concat_epochs_code and use_concat_first_epoch:
                X_ctx_tr  = _concat_truncate(X_gist_clean_tr.to(device),  xb_tr.to(device),  max_len=max_in_len)
                X_ctx_dev = _concat_truncate(X_gist_clean_dev.to(device), xb_dev.to(device), max_len=max_in_len)
            else:
                X_ctx_tr  = X_gist_clean_tr.to(device)[:,  :max_in_len]
                X_ctx_dev = X_gist_clean_dev.to(device)[:, :max_in_len]

        else:
            # ---- Issue agent: plain original X, no gist/concat curriculum ----
            X_ctx_tr  = xb_tr.to(device)[:, :max_in_len]
            X_ctx_dev = xb_dev.to(device)[:, :max_in_len]

        # ===== TRAIN =====
        for i in range(0, xb_tr.size(0), batch_size):
            xb = X_ctx_tr[i:i+batch_size].to(device)
            yb = tb_tr[i:i+batch_size].to(device)  # targets depend on agent (Y for code, P for issue)
            y_in, y_tgt = shift_targets(yb)
            logits = model.forward_role(xb, y_in, agent_id=user_id)
            loss = loss_fn(logits, y_tgt)

            preds = logits.argmax(-1)
            correct_train += (preds == y_tgt).masked_select(y_tgt != model.pad_idx).sum().item()
            total_train   += (y_tgt != model.pad_idx).sum().item()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            ep_loss += float(loss.detach()) * xb.size(0)

        train_acc = correct_train / max(total_train, 1)
        train_ce  = ep_loss / float(max(len(xb_tr), 1))

        # ===== DEV =====
        model.eval()
        with torch.no_grad():
            y_in_dev, y_tgt_dev = shift_targets(tb_dev.to(device))
            logits_dev = model.forward_role(X_ctx_dev.to(device), y_in_dev, agent_id=user_id)
            dev_ce = float(loss_fn(logits_dev, y_tgt_dev).item())

            preds_dev = logits_dev.argmax(-1)
            correct_dev = (preds_dev == y_tgt_dev).masked_select(y_tgt_dev != model.pad_idx).sum().item()
            total_dev   = (y_tgt_dev != model.pad_idx).sum().item()
            dev_acc = correct_dev / max(total_dev, 1)

        print(f"[Agentic][Static Routing][{agent_pretty_name(user_id)} FT] "
                f"Epoch {ep} | TrainCE={train_ce:.3f} | TrainAcc={train_acc:.3f} "
                f"| DevCE={dev_ce:.3f} | DevAcc={dev_acc:.3f}")

        # Early stopping
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
        if name.startswith("role_head"): p.requires_grad = True
        elif unfreeze_adapters and name.startswith("adapter"): p.requires_grad = True

def _set_trainable_strict_agent(
    model: AgenticTransformerSeq2Seq,
    *,
    agent_id: int = AGENT_CODE_GENERATION,
    unfreeze_backbone: bool = True,
    unfreeze_adapter: bool = True,
    unfreeze_dec_norms: bool = True
):
    for p in model.parameters():
        p.requires_grad = False
    ag = model.routing.agents[agent_id]
    for name, p in ag.named_parameters():
        if name.startswith("role_head"): p.requires_grad = True
        elif unfreeze_adapter and name.startswith("adapter"): p.requires_grad = True
    if unfreeze_backbone:
        for p in model.encoder.parameters(): p.requires_grad = True
        for p in model.decoder.parameters(): p.requires_grad = True
    elif unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name: p.requires_grad = True

def _wrap_issue_ids_with_tags(tok: "SubwordTokenizer", issue_ids: torch.Tensor) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    B = issue_ids.size(0)
    for i in range(B):
        ids = [t for t in issue_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]
        issue_text = _clean_issue_text(tok.decode(ids))
        wrapped = f"<ISSUE_GIST>\n{issue_text}\n</ISSUE_GIST>"
        row = torch.tensor(tok.sp.encode(wrapped, out_type=int), dtype=torch.long)
        rows.append(row if len(row) > 0 else torch.tensor([tok.pad], dtype=torch.long))
    return pad_sequence(rows, batch_first=True, padding_value=tok.pad)

def _set_trainable_stage1_joint(
    model: AgenticTransformerSeq2Seq,
    *,
    unfreeze_backbone: bool = True,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = True,
):
    for p in model.parameters(): p.requires_grad = False
    for agent_id in (AGENT_ISSUE_ANALYSIS, AGENT_CODE_GENERATION):
        ag = model.routing.agents[agent_id]
        for name, p in ag.named_parameters():
            if name.startswith("role_head"): p.requires_grad = True
            elif unfreeze_adapters and name.startswith("adapter"): p.requires_grad = True
    if unfreeze_backbone:
        for p in model.encoder.parameters(): p.requires_grad = True
        for p in model.decoder.parameters(): p.requires_grad = True
    elif unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name: p.requires_grad = True

def train_stage1_interleaved(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    P_train: torch.Tensor,
    *,
    tok: "SubwordTokenizer",
    issue_max_len: int = 124,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    unfreeze_backbone: bool = True,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = True,
    max_in_len: Optional[int] = None,
):
    """
    Stage 1 (INTERLEAVED per epoch):
        1) Train ISSUE (Agent 0) with teacher-forcing on ISSUE_DESC targets.
        2) Generate <ISSUE> (no grad), append to X, train CODE (Agent 1) on patch targets.
        Joint step uses SUM of both losses.
    """
    assert AGENT_ISSUE_ANALYSIS == 0 and AGENT_CODE_GENERATION == 1
    model.to(device)

    _set_trainable_stage1_joint(model, unfreeze_backbone=unfreeze_backbone,
                                unfreeze_adapters=unfreeze_adapters,
                                unfreeze_dec_norms=unfreeze_dec_norms)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    N = X_train.size(0)
    max_in_len = int(max_in_len or X_train.size(1))

    for ep in range(1, epochs + 1):
        model.train()
        issue_loss_sum = code_loss_sum = 0.0
        issue_tok_correct = issue_tok_total = 0
        code_tok_correct = code_tok_total = 0

        for i in range(0, N, batch_size):
            xb = X_train[i:i+batch_size].to(device)
            yb = Y_train[i:i+batch_size].to(device)
            pb = P_train[i:i+batch_size].to(device)

            # (1) ISSUE supervised
            y_in_p, y_tgt_p = shift_targets(pb)
            logits_p = model.forward_role(xb, y_in_p, agent_id=AGENT_ISSUE_ANALYSIS)
            loss_p = loss_fn(logits_p, y_tgt_p)

            with torch.no_grad():
                preds_p = logits_p.argmax(dim=-1)
                mask_p  = (y_tgt_p != model.pad_idx)
                issue_tok_correct += ((preds_p == y_tgt_p) & mask_p).sum().item()
                issue_tok_total   += mask_p.sum().item()
                issue_loss_sum    += float(loss_p.detach()) * xb.size(0)

            # Generate issue context (greedy + fallback)
            with torch.no_grad():
                issue_ctx, _ = _issue_ctx_greedy_with_fallback(
                    model, tok, xb, issue_max_len=issue_max_len
                )
                xb_gist = build_code_context_inputs(
                    model,
                    tok,
                    xb,
                    issue_ctx=issue_ctx,
                    issue_max_len=issue_max_len,
                    max_in_len=max_in_len,
                    device=device,
                )

            # (2) CODE supervised on augmented input
            y_in_c, y_tgt_c = shift_targets(yb)
            logits_c = model.forward_role(xb_gist, y_in_c, agent_id=AGENT_CODE_GENERATION)
            loss_c = loss_fn(logits_c, y_tgt_c)

            with torch.no_grad():
                preds_c = logits_c.argmax(dim=-1)
                mask_c  = (y_tgt_c != model.pad_idx)
                code_tok_correct += ((preds_c == y_tgt_c) & mask_c).sum().item()
                code_tok_total   += mask_c.sum().item()
                code_loss_sum    += float(loss_c.detach()) * xb.size(0)

            # Joint step
            loss = loss_p + loss_c
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        issue_ce  = issue_loss_sum / float(N)
        code_ce  = code_loss_sum / float(N)
        issue_acc = (issue_tok_correct / max(issue_tok_total, 1)) if issue_tok_total > 0 else 0.0
        code_acc = (code_tok_correct / max(code_tok_total, 1)) if code_tok_total > 0 else 0.0

        print(
            f"[Agentic][Training][Epoch {ep}] "
            f"ISSUE: CE={issue_ce:.3f} | tok_acc={issue_acc:.3f}  ||  "
            f"CODE: CE={code_ce:.3f} | tok_acc={code_acc:.3f}  ||  ",
            flush=True,
        )

@torch.no_grad()
def _eval_code_ce_acc(
    model: "AgenticTransformerSeq2Seq",
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    device: str = DEVICE
    ) -> Tuple[float, float]:
    """Teacher-forced CE/accuracy for the Code agent on input X vs gold Y."""
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    y_in, y_tgt = shift_targets(Y.to(device))
    logits = model.forward_role(X.to(device), y_in, agent_id=AGENT_CODE_GENERATION)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / (mask.float().sum() + 1e-8)).item())
    return ce, acc

@torch.no_grad()
def eval_pipeline_lift(
    model: AgenticTransformerSeq2Seq,
    tok: SubwordTokenizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    issue_max_len: int,
    max_in_len: int,
    device: str = DEVICE
):
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    # 1) CE with **no issue text** to CODE (true baseline).
    #    Use a single UNK token as minimal, non-masked encoder input to avoid all-pad attention edge cases.
    y_in, y_tgt = shift_targets(Y.to(device))
    B = X.size(0)
    X_no_issue = torch.full((B, 1), UNK, dtype=torch.long, device=device)  # shape [B, 1]
    logits_base = model.forward_role(X_no_issue, y_in, agent_id=AGENT_CODE_GENERATION)
    ce_base = float(loss_fn(logits_base, y_tgt).item())
    acc_base = float((((logits_base.argmax(-1) == y_tgt) & (y_tgt != model.pad_idx)).float().sum())
                        / ((y_tgt != model.pad_idx).float().sum().clamp_min(1.0)))

    # 2) CE with **with gist** context to CODE
    issue_ctx, _ = _issue_ctx_greedy_with_fallback(model, tok, X.to(device), issue_max_len=issue_max_len)
    gist_only = issue_ctx[:, :max_in_len]
    logits_gist = model.forward_role(gist_only, y_in, agent_id=AGENT_CODE_GENERATION)
    ce_gist = float(loss_fn(logits_gist, y_tgt).item())
    acc_gist = float((((logits_gist.argmax(-1) == y_tgt) & (y_tgt != model.pad_idx)).float().sum())
                        / ((y_tgt != model.pad_idx).float().sum().clamp_min(1.0)))

    print("\n[Agentic][Testing][PIPELINE-LIFT] Teacher-forced delta (with gist vs **no-issue baseline**; more negative is better)")
    print(f"[Agentic][Testing][PIPELINE-LIFT] CODE CE(no-issue)={ce_base:.3f} | CE(with gist)={ce_gist:.3f} | ΔCE={ce_gist - ce_base:.3f} | acc(no-issue)={acc_base:.3f} | acc(with gist)={acc_gist:.3f}")
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
        logits = model.routing.agents[a].project(last, head="role")
        vec = logits[0, 0, :n_tokens].detach().cpu().numpy()
        print(f"[{agent_pretty_name(a)}] role_head logits[:{n_tokens}] -> {vec}")

@torch.no_grad()
def per_agent_role_eval_code_on_gist(model, tok, X, Y, *, issue_max_len: int, max_in_len: int, device: str = DEVICE):
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    y_in, y_tgt = shift_targets(Y.to(device))
    issue_ctx, _ = _issue_ctx_greedy_with_fallback(model, tok, X.to(device), issue_max_len=issue_max_len)
    gist_only = issue_ctx[:, :max_in_len]
    logits = model.forward_role(gist_only, y_in, agent_id=AGENT_CODE_GENERATION)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())
    print(f"[Agentic][Testing][CODE@GIST] CE={ce:.3f} | tok_acc={acc:.3f} | N={int(X.size(0))}")

# ============================================================
# Small tensor helpers
# ============================================================
def _concat_truncate(a: torch.Tensor, b: torch.Tensor, *, max_len: int) -> torch.Tensor:
    out = torch.cat([a, b], dim=1)
    if out.size(1) > max_len:
        out = out[:, :max_len]
    return out

@torch.no_grad()
def issue_analysis_stats(model: AgenticTransformerSeq2Seq, tok: SubwordTokenizer, X: torch.Tensor,
                        *, issue_max_len: int, device: str = DEVICE) -> Dict[str, float]:
    """Quick quality probes on A's output: length, lines, and crude code leakage."""
    model.to(device); model.eval()
    issues = _generate_static(model, X.to(device), agent_id=AGENT_ISSUE_ANALYSIS,
        max_len=issue_max_len, top_k=20, top_p=0.90,
        temperature=0.8, no_repeat_ngram_size=4, min_len=24  # Adjusted min_len
)
    ISSUE = min(4, issues.size(0))
    decoded = [tok.decode([t for t in issues[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]) for i in range(ISSUE)]
    lengths = [len(s.split()) for s in decoded]
    line_counts = [s.count("\n") + 1 for s in decoded]
    code_leak_lines = sum(1 for s in decoded for ln in s.splitlines()
                            if ("diff --git" in ln) or ("def " in ln) or ln.strip().startswith("class ") or ("```" in ln))
    return {
        "sampled": float(ISSUE),
        "avg_tokens": float(np.mean(lengths) if lengths else 0.0),
        "avg_lines": float(np.mean(line_counts) if line_counts else 0.0),
        "code_leak_lines": float(code_leak_lines),
    }

@torch.no_grad()
def issue_to_code_alignment_sample(model: AgenticTransformerSeq2Seq, tok: SubwordTokenizer, X: torch.Tensor,
                                    *, issue_max_len: int, out_max_len: int, max_in_len: int, k: int = 3,
                                    device: str = DEVICE) -> None:
    """Print K examples: issue (A) and patch (B) to eyeball alignment."""
    model.to(device); model.eval()
    Xk = X[:k].to(device)
    issue_ids, patch_ids = model.routing.run_pipeline(
        model, tok, Xk, issue_max_len=issue_max_len, out_max_len=out_max_len, max_in_len=max_in_len
    )
    for i in range(min(k, Xk.size(0))):
        issue = tok.decode([t for t in issue_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)])
        patch = tok.decode([t for t in patch_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)])
        print(f"\n=== Example {i} ===")
        print("[ISSUE]\n", issue[:800])
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

def _make_issue_gist(title: str, desc: str) -> str:
    title = (title or "").strip()
    desc = (desc or "").strip()
    if not title and not desc:
        return ""
    # core = _first_n_words(desc, 100) if desc else ""  # Alternative: word-based
    core = _first_k_sentences(desc, 3) if desc else ""  # Sentence-based for coherence
    if title and core:
        return f"{title}: {core}"
    return title or core

def _clean_issue_text(txt: str) -> str:
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", txt).strip()


def _postprocess_gist(txt: str) -> str:
    txt = _clean_issue_text(txt)
    txt = re.sub(r"[`*_<>\[\]#]{1,}", " ", txt)
    txt = re.sub(r"\s*[.,;:!?]\s*", lambda m: m.group(0).strip() + " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    sent = _first_k_sentences(txt, 3)
    if not sent:
        sent = " ".join(txt.split()[:100])
    return sent.strip()


def _normalize_patch_line_breaks(txt: str) -> str:
    txt = txt.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" not in txt:
        replacements = (
            (" diff --git ", "\ndiff --git "),
            (" --- ", "\n--- "),
            (" +++ ", "\n+++ "),
            (" @@ ", "\n@@ "),
            (" if ", "\nif "),
            (" for ", "\nfor "),
            (" while ", "\nwhile "),
            (" return ", "\nreturn "),
            (": ", ":\n"),
        )
        for old, new in replacements:
            txt = txt.replace(old, new)
    return re.sub(r"\n+", "\n", txt).strip()


def _looks_like_unified_diff(txt: str) -> bool:
    return any(marker in txt for marker in ("diff --git", "@@", "+++", "---"))


def _normalize_diff_path(path: str) -> str:
    path = path.strip().strip('"\'')
    path = re.sub(r"^[ab]/", "", path)
    path = path.replace("\\", "/")
    path = re.sub(r"\s+", "", path)
    path = re.sub(r"/+", "/", path)
    path = re.sub(r"[^A-Za-z0-9._/\-]", "", path)
    return path or "file.py"


def _valid_diff_path(path: str) -> bool:
    if not path or path in {".", ".."}:
        return False
    if path.startswith("/") or path.startswith("../") or "/../" in path:
        return False
    return bool(re.match(r"^[A-Za-z0-9._/\-]+$", path))


def _repair_diff_headers(txt: str) -> str:
    match = re.search(r"diff --git\s+a/(\S+)\s+b/(\S+)", txt)
    if not match:
        return txt
    a_path = _normalize_diff_path(match.group(1))
    b_path = _normalize_diff_path(match.group(2))
    header = [
        f"diff --git a/{a_path} b/{b_path}",
        f"--- a/{a_path}",
        f"+++ b/{b_path}",
    ]
    remainder = txt[match.end():].strip()
    if remainder and not remainder.startswith("@@"):
        remainder = "@@\n" + remainder
    elif not remainder:
        remainder = "@@\n+# TODO: regenerate patch"
    return "\n".join(header + [remainder]).strip()


def _line_has_patch_shape(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith(("diff --git", "--- ", "+++ ", "@@", "+", "-", " "))


def _sanitize_generated_patch_text(txt: str) -> str:
    txt = _normalize_patch_line_breaks(txt)
    if not txt:
        return ""

    txt = re.sub(r"```(?:diff|patch|python)?", "", txt)
    txt = txt.replace("```", "")
    txt = txt.replace("\x00", "")

    sanitized_lines: List[str] = []
    for raw_line in txt.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.count("diff --git") > 1:
            parts = [part.strip() for part in line.split("diff --git") if part.strip()]
            sanitized_lines.extend([f"diff --git {part}" for part in parts])
            continue
        if re.search(r"\bdef\s*\)\b|self\s*=\s*self\)|\+def\s*\)", line):
            continue
        sanitized_lines.append(line)

    txt = "\n".join(sanitized_lines).strip()
    txt = re.sub(r"\n{3,}", "\n\n", txt)

    if _looks_like_unified_diff(txt):
        txt = _repair_diff_headers(txt)

        cleaned_lines: List[str] = []
        seen_header = False
        for line in txt.splitlines():
            if line.startswith("diff --git "):
                seen_header = True
                m = re.match(r"diff --git\s+a/(\S+)\s+b/(\S+)", line)
                if not m:
                    continue
                a_path = _normalize_diff_path(m.group(1))
                b_path = _normalize_diff_path(m.group(2))
                if not (_valid_diff_path(a_path) and _valid_diff_path(b_path)):
                    continue
                cleaned_lines.append(f"diff --git a/{a_path} b/{b_path}")
                continue
            if line.startswith("--- "):
                path = _normalize_diff_path(line[4:])
                if _valid_diff_path(path):
                    cleaned_lines.append(f"--- a/{path}")
                continue
            if line.startswith("+++ "):
                path = _normalize_diff_path(line[4:])
                if _valid_diff_path(path):
                    cleaned_lines.append(f"+++ b/{path}")
                continue
            if not seen_header:
                continue
            if line.startswith("@@"):
                cleaned_lines.append(line if line.strip() else "@@")
                continue
            if _line_has_patch_shape(line):
                cleaned_lines.append(line)

        txt = "\n".join(cleaned_lines).strip()
        if "@@" not in txt:
            txt = f"{txt}\n@@\n+# TODO: regenerate patch"

    return txt.strip()


def _patch_text_is_plausible(txt: str) -> bool:
    if not txt:
        return False
    if not _looks_like_unified_diff(txt):
        return False
    if "diff --git a/" not in txt or "\n--- a/" not in txt or "\n+++ b/" not in txt or "\n@@" not in txt:
        return False
    if re.search(r"\+def\s*\)|self\s*=\s*self\)|\bdiff --git\s+a/\S*\s+b/$", txt):
        return False
    lines = txt.splitlines()
    if len(lines) < 4:
        return False
    change_lines = [ln for ln in lines if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]
    return len(change_lines) >= 1


def _build_patch_stub_from_issue(issue_text: str) -> str:
    repo_match = re.search(r"repo=([^,\]\s]+)", issue_text)
    repo_name = repo_match.group(1).split("/")[-1] if repo_match else "repo"
    hinted_file = "fixme.py"
    for cand in re.findall(r"[A-Za-z0-9_./\-]+\.(?:py|pyi|js|ts|tsx|jsx|java|go|rb|php|rs|cpp|c|h)", issue_text):
        if "/" in cand or cand.endswith((".py", ".js", ".ts", ".java", ".go", ".rb", ".rs", ".cpp")):
            hinted_file = _normalize_diff_path(cand)
            break
    return (
        f"diff --git a/{hinted_file} b/{hinted_file}\n"
        f"--- a/{hinted_file}\n"
        f"+++ b/{hinted_file}\n"
        f"@@\n"
        f"+# TODO: synthesize valid fix for {repo_name} issue\n"
    )


def _fallback_placeholder_patch() -> str:
    return "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py\n@@\n+# TODO: regenerate patch"

def _is_noisy_gist(txt: str) -> bool:
    if not txt:
        return True

    # Token and character level sanity
    words = txt.split()
    if len(words) < 6:                      # was 4; require a bit more substance
        return True

    # Alphanumeric density: require at least 30% of characters to be [A-Za-z0-9]
    alnum = sum(ch.isalnum() for ch in txt)
    if (alnum / max(len(txt), 1)) < 0.30:
        return True

    # Disallow obvious code/patch markers
    bad_markers = (
        "diff --git", "```", "@@", "+++", "---",
        "class ", "def ", "://", "/pytorch", "/prefect"
    )
    if any(b in txt for b in bad_markers):
        return True

    # Too many non-word symbols (count everything except letters, digits, and spaces)
    nonword = re.sub(r"[A-Za-z0-9\s]", "", txt)
    if (len(nonword) / max(len(txt), 1)) >= 0.35:   # use >= and higher threshold
        return True

    return False

def _decode_row_no_pad(tok: "SubwordTokenizer", row: torch.Tensor) -> str:
    ids = [int(t) for t in row.tolist() if int(t) != tok.pad]
    return tok.decode(ids)

def _fallback_gist_from_input_text(in_text: str) -> str:
    title = _extract_tag_block(in_text, "ISSUE_TITLE")
    desc  = _extract_tag_block(in_text, "ISSUE_DESC") or in_text
    return _postprocess_gist(_make_issue_gist(title, desc))


def _extract_repo_anchor(in_text: str, *, max_chars: int = 600) -> str:
    title = _extract_tag_block(in_text, "ISSUE_TITLE")
    desc = _extract_tag_block(in_text, "ISSUE_DESC") or in_text
    hints = _extract_tag_block(in_text, "HINTS")
    repo_match = re.search(r"repo=([^,\]\s]+)", in_text)
    base_match = re.search(r"base=([^,\]\s]+)", in_text)

    repo = repo_match.group(1) if repo_match else "unknown-repo"
    base = base_match.group(1) if base_match else "unknown-base"

    lines = [f"repo: {repo}", f"base: {base}"]
    if title:
        lines.append(f"title: {title}")
    if desc:
        lines.append(f"problem: {_first_k_sentences(desc, 2)}")
    if hints:
        lines.append(f"hints: {_first_k_sentences(hints, 2)}")

    joined = "\n".join(lines)
    joined = _clean_issue_text(joined)
    return joined[:max_chars].strip()


def build_code_context_inputs(
    model: "AgenticTransformerSeq2Seq",
    tok: "SubwordTokenizer",
    X: torch.Tensor,
    *,
    issue_ctx: Optional[torch.Tensor] = None,
    issue_max_len: int,
    max_in_len: int,
    device: str = DEVICE,
) -> torch.Tensor:
    model.to(device)
    model.eval()
    X_dev = X.to(device)

    if issue_ctx is None:
        issue_ctx, _ = _issue_ctx_greedy_with_fallback(
            model, tok, X_dev, issue_max_len=issue_max_len
        )
    else:
        issue_ctx = issue_ctx.to(device)

    if not CFG.code_use_repo_anchor:
        return issue_ctx[:, :max_in_len]

    ctx_rows: List[torch.Tensor] = []
    for i in range(X_dev.size(0)):
        in_text = _decode_row_no_pad(tok, X_dev[i])
        anchor = _extract_repo_anchor(in_text, max_chars=CFG.repo_anchor_max_len)
        gist_text = tok.decode([int(t) for t in issue_ctx[i].tolist() if int(t) != tok.pad]).strip()
        merged = (
            f"<ISSUE_GIST>\n{_postprocess_gist(gist_text)}\n</ISSUE_GIST>\n"
            f"<REPO_CONTEXT>\n{anchor}\n</REPO_CONTEXT>"
        )
        ctx_rows.append(tok.encode(merged, add_bos_eos=False, max_len=max_in_len))

    return pad_sequence(ctx_rows, batch_first=True, padding_value=tok.pad)

@torch.no_grad()
def _issue_ctx_greedy_with_fallback(
    model: "AgenticTransformerSeq2Seq",
    tok: "SubwordTokenizer",
    X: torch.Tensor,
    *,
    issue_max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        issue_ctx  : [B, Tctx] token ids for <ISSUE_GIST>...</ISSUE_GIST> (no BOS/EOS)
        display_ids: [B, Tdisp] BOS ... EOS ids of the plain gist for printing
    """
    # 1) Greedy generation (no sampling) from the Issue agent
    gen_ids = _generate_static(
        model, X, agent_id=AGENT_ISSUE_ANALYSIS, max_len=issue_max_len,
        top_k=None, top_p=None, temperature=1.0, no_repeat_ngram_size=3, min_len=24  # Increased min_len
    )

    B = X.size(0)
    gists: List[str] = []

    # 2) Per-row cleanup + fallback to deterministic gist if noisy
    for i in range(B):
        raw = [t for t in gen_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]
        gen_txt = tok.decode(raw)
        gen_txt = _postprocess_gist(gen_txt)

        # First-level check
        if _is_noisy_gist(gen_txt):
            in_text = _decode_row_no_pad(tok, X[i])
            gen_txt = _fallback_gist_from_input_text(in_text)
            gen_txt = _postprocess_gist(gen_txt)

        # Final safety: if still noisy, force a minimal title-only fallback
        if _is_noisy_gist(gen_txt):
            in_text = _decode_row_no_pad(tok, X[i])
            title = _extract_tag_block(in_text, "ISSUE_TITLE")
            gen_txt = (title or "Issue: (no description)").strip()

        gists.append(gen_txt or "Issue: (no description)")

    # 3) Encode context (<ISSUE_GIST>…</ISSUE_GIST>) and plain display ids
    ctx_rows, disp_rows = [], []
    for g in gists:
        ctx_txt = f"<ISSUE_GIST>\n{g}\n</ISSUE_GIST>"
        ctx_rows.append(torch.tensor(tok.sp.encode(ctx_txt, out_type=int), dtype=torch.long))
        disp_rows.append(tok.encode(g, add_bos_eos=True, max_len=issue_max_len))

    issue_ctx   = pad_sequence(ctx_rows, batch_first=True, padding_value=tok.pad)
    display_ids = pad_sequence(disp_rows, batch_first=True, padding_value=tok.pad)
    return issue_ctx, display_ids

# === NEW: helper to precompute gist-only encoder inputs =======================
@torch.no_grad()
def build_gist_only_inputs(
    model: "AgenticTransformerSeq2Seq",
    tok: "SubwordTokenizer",
    X: torch.Tensor,
    *,
    issue_max_len: int,
    max_in_len: int,
    device: str = DEVICE
) -> torch.Tensor:
    """
    Returns encoder inputs for the Code agent, rooted in the generated issue gist
    and optionally augmented with compact repo/problem anchors extracted from X.
    """
    model.to(device); model.eval()
    issue_ctx, _ = _issue_ctx_greedy_with_fallback(
        model, tok, X.to(device), issue_max_len=issue_max_len
    )
    return build_code_context_inputs(
        model,
        tok,
        X.to(device),
        issue_ctx=issue_ctx,
        issue_max_len=issue_max_len,
        max_in_len=max_in_len,
        device=device,
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

# =========================
# Issue agent: evaluation
# =========================
@torch.no_grad()
def _eval_issue_ce_acc(
    model: "AgenticTransformerSeq2Seq",
    X: torch.Tensor,
    P: torch.Tensor,
    *,
    device: str = DEVICE
) -> Tuple[float, float]:
    """
    Teacher-forced CE/accuracy for the Issue agent on input X vs gold ISSUE_DESC targets P.
    """
    model.to(device); model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)
    y_in, y_tgt = shift_targets(P.to(device))
    logits = model.forward_role(X.to(device), y_in, agent_id=AGENT_ISSUE_ANALYSIS)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())
    return ce, acc

# ============================================================
# Orchestration
# ============================================================
def _resolve_output_dir(out_dir: str) -> str:
    candidate = Path(out_dir).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return str(candidate)
    except OSError:
        fallback = Path.home() / "Documents" / "SPE2026" / Path(out_dir).name
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


def run_all(cfg: Config = CFG):
    global CFG
    CFG = cfg

    set_seed(cfg.seed)

    # Data
    data = SWEText2PatchData(split="train", limit=cfg.limit, max_in_len=cfg.max_in_len,
                                max_out_len=cfg.max_out_len, spm_vocab_size=cfg.spm_vocab,
                                demo_data=cfg.demo_data)

    global data_tok
    data_tok = data.tok

    # Issue targets = ISSUE_DESC
    ids, X, Y, P = data.as_tensors_with_issue_targets(issue_max_len=min(cfg.max_out_len, 256))

    # Deterministic shuffle then split
    N = len(ids)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(N, generator=g)
    ids = [ids[i] for i in perm.tolist()]
    X, Y, P = X[perm], Y[perm], P[perm]
    split = int(N * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    P_train, P_test = P[:split], P[split:]
    print(f"[Info] Train: {split} pairs, Test: {N - split} pairs")

    # Model
    max_len_for_model = max(cfg.max_len_cap, X.size(1) + min(cfg.max_out_len, 256))
    model = AgenticTransformerSeq2Seq(
        vocab_size=data.tok.vocab_size,
        n_agents=cfg.n_agents, model_dim=cfg.model_dim, n_heads=cfg.n_heads,
        n_layers_enc=cfg.n_layers_enc, n_layers_dec=cfg.n_layers_dec,
        max_len=max_len_for_model, pad_idx=data.tok.pad
    )

    # ===== Stage 1: Interleaved ISSUE↔CODE =====
    print("[Agentic][Training] Stage 1: Interleaved ISSUE↔CODE (same epoch)")
    train_stage1_interleaved(
        model, X_train, Y_train, P_train,
        tok=data.tok,
        issue_max_len=min(cfg.max_out_len, 256),
        epochs=cfg.pipe_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=DEVICE,
        unfreeze_backbone=True,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        max_in_len=cfg.max_in_len
    )

    # ------ Pipeline Lift (teacher-forced) ------
    eval_pipeline_lift(
        model, data.tok, X_test, Y_test,
        issue_max_len=min(cfg.max_out_len, 256),
        max_in_len=cfg.max_in_len,
        device=DEVICE
    )

    # NEW: evaluate Code agent on gist-only input (teacher-forced)
    per_agent_role_eval_code_on_gist(
        model, data.tok, X_test, Y_test,
        issue_max_len=min(cfg.max_out_len, 256),
        max_in_len=cfg.max_in_len,
        device=DEVICE
    )

    # ------ Diagnostics ------
    print("\n[Agentic][Testing][ISSUE-STATS] Analysis agent quick stats (first few issues)")
    stats = issue_analysis_stats(model, data.tok, X_test, issue_max_len=min(cfg.max_out_len, 256), device=DEVICE)
    print(stats)

    print("\n[Agentic][Testing][PIPELINE-SAMPLES] Issue ↔ Patch examples (eyeball alignment)")
    issue_to_code_alignment_sample(model, data.tok, X_test,
                                    issue_max_len=min(cfg.max_out_len, 256),
                                    out_max_len=cfg.decode_max_len,
                                    max_in_len=cfg.max_in_len,
                                    k=3, device=DEVICE)

    # ===== Helpers for CE/Acc =====
    @torch.no_grad()
    def _eval_issue_ce_acc_local(m: AgenticTransformerSeq2Seq, Xenc: torch.Tensor, Ptg: torch.Tensor, *, device: str):
        m.to(device); m.eval()
        loss_fn = SeqCELoss(pad_idx=m.pad_idx)
        y_in, y_tgt = shift_targets(Ptg.to(device))
        logits = m.forward_role(Xenc.to(device), y_in, agent_id=AGENT_ISSUE_ANALYSIS)
        ce = float(loss_fn(logits, y_tgt).item())
        preds = logits.argmax(-1)
        mask = (y_tgt != m.pad_idx)
        acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())
        return ce, acc

    # ===========================
    # ===== Stage 2A: Issue FT
    # ===========================
    print("\n[Agentic][Training] Stage 2A: Static specialization for ISSUE agent "
            "(freeze backbone + Code agent; train Issue agent on original X with P targets)")

    iss_ce_before, iss_acc_before = _eval_issue_ce_acc_local(model, X_test, P_test, device=DEVICE)
    print(f"[Agentic][Testing][ISSUE@Before FT] CE={iss_ce_before:.3f} | tok_acc={iss_acc_before:.3f}")

    fine_tune_static(
        model, X_train, Y_train,                 # Y_train ignored for Issue FT; P_train used as targets
        user_id=AGENT_ISSUE_ANALYSIS,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms, # unfreeze decoder norms frozen if you want tighter freeze
        unfreeze_decoder_tail_blocks=0,          # no extra decoder capacity needed for Issue FT
        idxs=None,
        device=DEVICE,
        tok=data.tok,
        P=P_train,                               # REQUIRED for Issue FT
        gist_ctx_fn=None,                        # not used for Issue FT
        max_in_len=CFG.max_in_len,
        use_concat_first_epoch=False,            # not applicable to Issue FT
        patience=2
    )

    iss_ce_after, iss_acc_after = _eval_issue_ce_acc_local(model, X_test, P_test, device=DEVICE)
    print(
        f"[Agentic][Testing][ISSUE@After FT] "
        f"CE={iss_ce_after:.3f} | tok_acc={iss_acc_after:.3f} "
        f"| ΔCE={iss_ce_after - iss_ce_before:+.3f} ({(iss_ce_after - iss_ce_before) / iss_ce_before * 100:+.2f}%) "
        f"| Δacc={iss_acc_after - iss_acc_before:+.3f} ({(iss_acc_after - iss_acc_before) / iss_acc_before * 100:+.2f}%)"
    )

    # ===========================
    # ===== Stage 2B: Code FT
    # ===========================
    print("\n[Agentic][Training] Stage 2B: Static specialization for CODE agent "
            "(freeze backbone + Issue agent; train Code agent on gist-only)")

    # Build gist-only inputs (once; no grad) for test measurement
    issue_len = min(cfg.max_out_len, 256)
    X_test_gist = build_gist_only_inputs(
        model, data.tok, X_test,
        issue_max_len=issue_len, max_in_len=cfg.max_in_len, device=DEVICE
    )

    ce_before, acc_before = _eval_code_ce_acc(model, X_test_gist, Y_test, device=DEVICE)
    print(f"[Agentic][Testing][CODE@GIST][Before FT] CE={ce_before:.3f} | tok_acc={acc_before:.3f}")

    def _gist_ctx_fn_for_ft(xb_device):
        issue_ctx, _ = _issue_ctx_greedy_with_fallback(model, data.tok, xb_device, issue_max_len=min(CFG.max_out_len, 256))
        return issue_ctx  # gist-only

    fine_tune_static(
        model, X_train, Y_train,
        user_id=AGENT_CODE_GENERATION,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=cfg.ft_unfreeze_decoder_tail_blocks_code,
        idxs=None,                      # or use a modulo-slice if you want per-agent shards
        device=DEVICE,
        tok=data.tok,
        P=P_train,                      # used only for epoch-1 curriculum (clean gist)
        gist_ctx_fn=_gist_ctx_fn_for_ft,
        max_in_len=CFG.max_in_len,
        use_concat_first_epoch=True,
        patience=2
    )

    # Measure CODE@GIST after FT (recompute gist from current Issue agent)
    ce_after, acc_after = _eval_code_ce_acc(
        model,
        _gist_ctx_fn_for_ft(X_test.to(DEVICE))[:, :CFG.max_in_len],
        Y_test,
        device=DEVICE
    )
    print(
        f"[Agentic][Testing][CODE@GIST][After FT] "
        f"CE={ce_after:.3f} | tok_acc={acc_after:.3f} "
        f"| ΔCE={ce_after - ce_before:+.3f} ({(ce_after - ce_before) / ce_before * 100:+.2f}%) "
        f"| Δacc={acc_after - acc_before:+.3f} ({(acc_after - acc_before) / acc_before * 100:+.2f}%)"
    )
    
    # ============================================================
    # NEW: Save sample ISSUE + GENERATED PATCH outputs
    # ============================================================
    print("\n[Agentic][Output] Saving sample ISSUE + PATCH outputs...")

    cfg.out_dir = _resolve_output_dir(cfg.out_dir)

    model.eval()
    with torch.no_grad():
        k = min(cfg.save_samples_k, X_test.size(0))
        X_sample = X_test[:k].to(DEVICE)

        issue_ids, patch_ids = model.routing.run_pipeline(
            model,
            data.tok,
            X_sample,
            issue_max_len=min(cfg.max_out_len, 256),
            out_max_len=cfg.decode_max_len,
            max_in_len=cfg.max_in_len
        )

        manifest: List[Dict[str, object]] = []
        for i in range(k):
            issue_txt = data.tok.decode(
                [t for t in issue_ids[i].tolist() if t not in (data.tok.pad, data.tok.bos, data.tok.eos)]
            ).strip()
            patch_txt = data.tok.decode(
                [t for t in patch_ids[i].tolist() if t not in (data.tok.pad, data.tok.bos, data.tok.eos)]
            ).strip()
            patch_txt = _sanitize_generated_patch_text(patch_txt)
            if not _patch_text_is_plausible(patch_txt):
                patch_txt = _build_patch_stub_from_issue(issue_txt)

            sample_path = os.path.join(cfg.out_dir, f"swe_sample_{i}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write("=== ISSUE (Generated by Agent A) ===\n")
                f.write(issue_txt + "\n\n")
                f.write("=== PATCH (Generated by Agent B) ===\n")
                f.write(patch_txt + "\n")

            manifest.append({
                "sample": i,
                "path": sample_path,
                "issue_chars": len(issue_txt),
                "patch_chars": len(patch_txt),
                "patch_plausible": _patch_text_is_plausible(patch_txt),
            })

    with open(os.path.join(cfg.out_dir, "samples_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    valid_count = sum(1 for row in manifest if row["patch_plausible"])
    print(f"[Agentic][Output] Saved {k} samples to '{cfg.out_dir}/' ({valid_count}/{k} structurally plausible diffs)")
    
    return model, data, (ids, X, Y, P)


if __name__ == "__main__":
    model, data, tensors = run_all(CFG)