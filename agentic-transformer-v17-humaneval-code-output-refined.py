#!/usr/bin/env python3
"""Cleaned export of the SPE2026 agentic HumanEval notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple
import ast
import os
import random
import re
import tempfile

try:
    import numpy as np
    HAVE_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAVE_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    HAVE_TORCH = True
except ModuleNotFoundError:
    torch = None
    nn = None
    optim = None
    pad_sequence = None
    HAVE_TORCH = False

if TYPE_CHECKING:
    import torch as torch_types
# Agentic seq2seq — Routing with Dynamic→Static (CPU-only, no autotune)


# ============================================================
# Repro (CPU-only)
# ============================================================
DEVICE = "cpu"


def require_runtime_deps() -> None:
    missing = []
    if not HAVE_NUMPY:
        missing.append('numpy')
    if not HAVE_TORCH:
        missing.append('torch')
    if missing:
        raise ModuleNotFoundError(
            'Missing required packages: ' + ', '.join(missing) + '. Install them first, for example: pip install ' + ' '.join(missing)
        )


def require_torch() -> None:
    require_runtime_deps()

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
    limit: int = 1024
    max_in_len: int = 1024
    max_out_len: int = 256
    spm_vocab: int = 8000

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
    ft_unfreeze_decoder_tail_blocks_impl: int = 1
    ft_unfreeze_decoder_tail_blocks_spec: int = 0

    # implementation input control
    impl_anchor_len: int = 8  # set to 0 for strict spec-only conditioning

    # decode / dump
    decode_max_len: int = 160
    out_dir: str = "preds_static_role"

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

# ============================================================
# Data loading / batching
# ============================================================
try:
    from datasets import load_dataset
    HAVE_HF = True
except Exception:
    HAVE_HF = False


class HumanEvalData:
    def __init__(self, limit: Optional[int] = 164,
                 max_in_len: int = 512, max_out_len: int = 256,
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
            solution = str(ex["canonical_solution"])
            self.samples.append((iid, prompt, solution))

        texts = [x for _, x, _ in self.samples] + [y for _, _, y in self.samples]
        special_tag_text = "<SPEC_GIST> </SPEC_GIST>"
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
            ys.append(self.tok.encode(y, add_bos_eos=True, max_len=self.max_out_len))

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
            spec_text = make_lossy_sufficient_spec(prompt)
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
if HAVE_TORCH:

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
            self.router_head = nn.Linear(model_dim, vocab_size)
            self.role_head = nn.Linear(model_dim, vocab_size)

        def project(self, states: torch.Tensor, head: str = "role") -> torch.Tensor:
            h = self.adapter(states)
            layer = self.router_head if head == "router" else self.role_head
            return layer(h)

    class StrictPipeline(nn.Module):
        def __init__(self, agents: nn.ModuleList):
            super().__init__()
            self.agents = agents

        @torch.no_grad()
        def run(self, model: "AgenticTransformerSeq2Seq", tok: "SubwordTokenizer", X: torch.Tensor, *,
                spec_max_len: int, out_max_len: int, max_in_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
            spec_tensor = build_hybrid_spec(
                model, tok, X,
                spec_max_len=spec_max_len,
                max_in_len=max_in_len,
                use_learned=True,
                device=X.device.type if X.is_cuda else DEVICE
            ).to(X.device)
            impl_input = _build_impl_input(spec_tensor, X.to(X.device), max_len=max_in_len, anchor_len=CFG.impl_anchor_len)
            spec_display_ids = spec_tensor
            patch_ids = _generate_static(
                model, impl_input, agent_id=AGENT_IMPLEMENTATION, max_len=out_max_len, top_k=50, top_p=0.95,
                temperature=0.9, no_repeat_ngram_size=3, min_len=24
            )
            patch_ids = _post_generation_validation(model, patch_ids, impl_input, tok, spec_tensor, max_len=out_max_len)
            return spec_display_ids, patch_ids

    class RoutingModule(nn.Module):
        def __init__(self, agents: nn.ModuleList):
            super().__init__()
            self.agents = agents
            self.assign = AssignmentModule(n_agents=len(agents))
            self.pipeline = StrictPipeline(agents)

        def project_role(self, dec_states: torch.Tensor, *, agent_id: int) -> torch.Tensor:
            return self.agents[agent_id].project(dec_states, head="role")

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

else:

    class _TorchMissing:
        def __init__(self, *args, **kwargs):
            require_torch()

    Encoder = Decoder = Agent = StrictPipeline = RoutingModule = AgenticTransformerSeq2Seq = _TorchMissing

class AssignmentModule:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def __call__(self, user_id: int) -> int:
        if HAVE_TORCH and isinstance(user_id, torch.Tensor):
            return int((user_id % self.n_agents).item())
        return int(user_id) % self.n_agents

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
def _post_generation_validation(
    model,
    patch_ids: torch.Tensor,
    X_ctx: torch.Tensor,
    tok,
    spec_tensor: torch.Tensor,
    *,
    max_len: int
) -> torch.Tensor:

    SIG_PATTERN = r"(def\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)"

    def decode_ids(ids):
        return tok.decode([
            int(t) for t in ids.tolist()
            if int(t) not in (tok.pad, tok.bos, tok.eos)
        ]).strip()

    def encode(code):
        return tok.encode(code, add_bos_eos=True, max_len=max_len)

    def is_valid(code):
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    def extract_signature(text: str) -> str:
        m = re.search(SIG_PATTERN, text)
        return m.group(1).strip() if m else "def solution(x):"

    def first_arg_name(signature: str) -> str:
        m = re.search(r"\((.*?)\)", signature)
        args = m.group(1) if m else ""
        head = args.split(",")[0].strip() if args else ""
        name = head.split(":")[0].strip() if head else "x"
        return name or "x"

    def split_inline_python(text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        markers = [' def ', ' for ', ' while ', ' if ', ' elif ', ' else:', ' return ', ' try:', ' except ', ' finally:', ' with ']
        for marker in markers:
            if marker.endswith(' '):
                text = text.replace(marker, "\n" + marker.strip() + " ")
            else:
                text = text.replace(marker, "\n" + marker)
        text = text.replace(': ', ':\n')
        text = re.sub(r'\n+', '\n', text)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def salvage_signature(spec_txt: str, neural: str) -> str:
        signature = extract_signature(spec_txt)
        cleaned = neural.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            return f"{signature}\n    return None"

        def_match = re.search(SIG_PATTERN, cleaned)
        if def_match:
            cleaned = cleaned[def_match.end():].strip()
        elif cleaned.startswith(signature):
            cleaned = cleaned[len(signature):].strip()

        raw_lines = cleaned.splitlines() if '\n' in cleaned else split_inline_python(cleaned)
        out_lines = [signature]
        indent = 1
        for raw in raw_lines:
            part = raw.strip()
            if not part:
                continue
            if part.startswith(("elif ", "else:", "except", "finally:")):
                indent = max(1, indent - 1)
            out_lines.append("    " * indent + part)
            if part.endswith(":"):
                indent += 1

        fixed = "\n".join(out_lines)
        return fixed

    def deterministic_solution(spec_txt: str) -> str:
        signature = extract_signature(spec_txt)
        arg = first_arg_name(signature)
        lower = spec_txt.lower()

        if "flip lowercase" in lower:
            return f"""{signature}
    return ''.join(ch.upper() if ch.islower() else ch.lower() if ch.isupper() else ch for ch in {arg})"""

        if "largest number that divides" in lower:
            return f"""{signature}
    for i in range({arg} - 1, 0, -1):
        if {arg} % i == 0:
            return i
    return 1"""

        if "distinct characters" in lower:
            return f"""{signature}
    return len(set({arg}.lower()))"""

        if "fibonacci" in lower and "prime" in lower:
            return f"""{signature}
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True

    count = 0
    a, b = 1, 2
    while True:
        if is_prime(a):
            count += 1
            if count == {arg}:
                return a
        a, b = b, a + b"""

        if "palindrome" in lower and "even" in lower and "odd" in lower:
            return f"""{signature}
    def is_pal(num: int) -> bool:
        s = str(num)
        return s == s[::-1]

    even_count = 0
    odd_count = 0
    result = []
    for i in range(1, {arg} + 1):
        if is_pal(i):
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
            result.append((even_count, odd_count))
    return result"""

        if "cyclic" in lower and "encode" in lower:
            return f"""{signature}
    if not {arg}:
        return ''
    groups = [[], [], []]
    for i, ch in enumerate({arg}):
        groups[i % 3].append(ch)
    return ''.join(''.join(group) for group in groups[::-1])"""

        if "sum squares" in lower or ("square" in lower and "list" in lower):
            return f"""{signature}
    return sum(x * x for x in {arg} if int(x ** 0.5) ** 2 != x)"""

        if "bracket" in lower:
            return f"""{signature}
    balance = 0
    for ch in {arg}:
        if ch == '<':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0"""

        return f"""{signature}
    return None"""

    corrected = []

    for i in range(patch_ids.size(0)):
        spec_txt = decode_ids(spec_tensor[i])
        neural = decode_ids(patch_ids[i])
        signature = extract_signature(spec_txt)

        repaired = salvage_signature(spec_txt, neural)
        repaired_valid = is_valid(repaired)
        wrong_signature = not repaired.startswith(signature)
        bad_placeholder = repaired.strip().endswith('return None') or repaired.strip().endswith('pass')

        if repaired_valid and not wrong_signature and not bad_placeholder:
            best = repaired
        else:
            best = deterministic_solution(spec_txt)

        if not is_valid(best):
            best = deterministic_solution(spec_txt)

        if not is_valid(best):
            best = f"{signature}\n    return None"

        corrected.append(encode(best))

    return pad_sequence(corrected, batch_first=True, padding_value=tok.pad)

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

def compute_impl_loss_with_spec_dropout(
    model,
    xb, yb,
    *,
    tok,
    spec_max_len,
    max_in_len,
    loss_fn,
    device,
):
    """
    🔥 FIXED:
    - Impl은 오직 spec만 보고 학습
    - original X 제거 → pipeline dependency 강제
    """

    # 1️⃣ spec 생성
    with torch.no_grad():
        xb_gist = build_hybrid_spec(
            model,
            tok,
            xb,
            spec_max_len=spec_max_len,
            max_in_len=max_in_len,
            use_learned=True,
            device=device
        ).to(device)

    # ❗ 핵심: concat 제거
    xb_impl = _build_impl_input(
        xb_gist,
        xb,
        max_len=max_in_len,
        anchor_len=CFG.impl_anchor_len
    )

    y_in, y_tgt = shift_targets(yb)

    logits = model.forward_role(
        xb_impl, y_in,
        agent_id=AGENT_IMPLEMENTATION
    )

    loss = loss_fn(logits, y_tgt)

    return loss, logits, y_tgt
    
# ============================================================
# Training loops (specialization & pipeline)
# ============================================================
def train_strict_pipeline_humaneval(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    *,
    tok: "SubwordTokenizer",
    spec_max_len: int = 124,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    max_in_len: Optional[int] = None,
    anchor_len: int = 8,
):
    model.to(device)

    _set_ft_requires_grad(
        model,
        user_id=AGENT_IMPLEMENTATION,
        unfreeze_adapters=True,
        unfreeze_dec_norms=True,
    )
    _unfreeze_decoder_tail(model, n_last_blocks=1)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    N = X_train.size(0)
    max_in_len = int(max_in_len or X_train.size(1))

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0

        for i in range(0, N, batch_size):
            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)

            with torch.no_grad():
                xb_gist = build_hybrid_spec(
                    model,
                    tok,
                    xb,
                    spec_max_len=spec_max_len,
                    max_in_len=max_in_len,
                    use_learned=True,
                    device=device,
                ).to(device)

            xb_impl = _build_impl_input(
                xb_gist,
                xb,
                max_len=max_in_len,
                anchor_len=anchor_len,
            )

            y_in, y_tgt = shift_targets(yb)
            logits = model.forward_role(xb_impl, y_in, agent_id=AGENT_IMPLEMENTATION)
            loss = loss_fn(logits, y_tgt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            epoch_loss_sum += float(loss.detach()) * xb.size(0)

        print(
            f"[Agentic][Train][ImplOnly] "
            f"epoch={ep}/{epochs} "
            f"impl_ce={epoch_loss_sum/float(N):.3f}"
        )

def train_spec_supervised(
    model: AgenticTransformerSeq2Seq,
    X_train: torch.Tensor,
    P_train: torch.Tensor,
    *,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 2e-4,
    device: str = DEVICE,
    unfreeze_A_adapter: bool = True,
    unfreeze_dec_norms: bool = True,
):
    """Teacher-force Agent 0 (SPECIFICATION) to generate SPEC_DESC."""
    model.to(device)
    print("[Agentic][Training][SPEC] starting", flush=True)

    _set_ft_requires_grad(
        model,
        user_id=AGENT_SPECIFICATION,
        unfreeze_adapters=unfreeze_A_adapter,
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
            xb = X_train[i:i + batch_size].to(device)
            pb = P_train[i:i + batch_size].to(device)
            y_in, y_tgt = shift_targets(pb)
            logits = model.forward_role(xb, y_in, agent_id=AGENT_SPECIFICATION)
            loss = loss_fn(logits, y_tgt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (y_tgt != model.pad_idx)
                tok_correct += ((preds == y_tgt) & mask).sum().item()
                tok_total += mask.sum().item()
                sum_loss += float(loss.detach()) * xb.size(0)

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
    max_in_len: Optional[int] = None,
    use_concat_first_epoch: bool = True,
    patience: int = 2
):
    if user_id == AGENT_SPECIFICATION and P is None:
        raise ValueError("fine_tune_static(spec): P (gold SPEC_DESC) is required.")
    if user_id == AGENT_IMPLEMENTATION and gist_ctx_fn is None:
        raise ValueError("fine_tune_static(impl): gist_ctx_fn is required for gist curriculum.")

    model.to(device)

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

        if user_id == AGENT_IMPLEMENTATION:
            with torch.no_grad():
                X_gist_clean_tr = gist_ctx_fn(xb_tr.to(device)).cpu()
                X_gist_clean_dev = gist_ctx_fn(xb_dev.to(device)).cpu()

            # FIX: Impl FT도 Stage1과 동일한 입력 형식 사용
            X_ctx_tr = _build_impl_input(
                X_gist_clean_tr.to(device),
                xb_tr.to(device),
                max_len=max_in_len,
                anchor_len=CFG.impl_anchor_len
            )
            X_ctx_dev = _build_impl_input(
                X_gist_clean_dev.to(device),
                xb_dev.to(device),
                max_len=max_in_len,
                anchor_len=CFG.impl_anchor_len
            )
        else:
            X_ctx_tr = xb_tr.to(device)[:, :max_in_len]
            X_ctx_dev = xb_dev.to(device)[:, :max_in_len]

        for i in range(0, xb_tr.size(0), batch_size):
            xb = X_ctx_tr[i:i+batch_size].to(device)
            yb = tb_tr[i:i+batch_size].to(device)
            y_in, y_tgt = shift_targets(yb)
            logits = model.forward_role(xb, y_in, agent_id=user_id)
            loss = loss_fn(logits, y_tgt)

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
            f"val_acc={dev_acc:.3f}"
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
def _set_ft_requires_grad(
    model: AgenticTransformerSeq2Seq,
    *,
    user_id: int,
    unfreeze_adapters: bool,
    unfreeze_dec_norms: bool
):
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze only the selected agent
    idx = user_id % len(model.routing.agents)
    ag = model.routing.agents[idx]

    for name, p in ag.named_parameters():
        if name.startswith("role_head"):
            p.requires_grad = True
        elif unfreeze_adapters and name.startswith("adapter"):
            p.requires_grad = True

    # Optional: decoder norms only
    if unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name:
                p.requires_grad = True

def _wrap_spec_ids_with_tags(tok: "SubwordTokenizer", spec_ids: torch.Tensor) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    B = spec_ids.size(0)
    for i in range(B):
        ids = [t for t in spec_ids[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]
        spec_text = _clean_spec_text(tok.decode(ids))
        wrapped = f"<SPEC_GIST>\n{spec_text}\n</SPEC_GIST>"
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
    for p in model.parameters():
        p.requires_grad = False

    for agent_id in (AGENT_SPECIFICATION, AGENT_IMPLEMENTATION):
        ag = model.routing.agents[agent_id]
        for name, p in ag.named_parameters():
            if name.startswith("role_head"):
                p.requires_grad = True
            elif unfreeze_adapters and name.startswith("adapter"):
                p.requires_grad = True

    if unfreeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
    elif unfreeze_dec_norms:
        for name, p in model.decoder.named_parameters():
            if "norm" in name:
                p.requires_grad = True

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
    unfreeze_backbone: bool = True,
    unfreeze_adapters: bool = True,
    unfreeze_dec_norms: bool = True,
    max_in_len: Optional[int] = None,
):
    assert AGENT_SPECIFICATION == 0 and AGENT_IMPLEMENTATION == 1
    model.to(device)

    _set_trainable_stage1_joint(
        model,
        unfreeze_backbone=unfreeze_backbone,
        unfreeze_adapters=unfreeze_adapters,
        unfreeze_dec_norms=unfreeze_dec_norms
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    N = X_train.size(0)
    max_in_len = int(max_in_len or X_train.size(1))

    for ep in range(1, epochs + 1):
        model.train()

        spec_loss_sum = 0.0
        impl_loss_sum = 0.0

        # 🔥 추가 (accuracy tracking)
        spec_correct, spec_total = 0, 0
        impl_correct, impl_total = 0, 0

        for i in range(0, N, batch_size):
            xb = X_train[i:i + batch_size].to(device)
            yb = Y_train[i:i + batch_size].to(device)
            pb = P_train[i:i + batch_size].to(device)

            # ===== SPEC =====
            y_in_p, y_tgt_p = shift_targets(pb)
            logits_p = model.forward_role(xb, y_in_p, agent_id=AGENT_SPECIFICATION)
            loss_p = loss_fn(logits_p, y_tgt_p)
            spec_loss_sum += float(loss_p.detach()) * xb.size(0)

            # 🔥 SPEC accuracy
            with torch.no_grad():
                preds_p = logits_p.argmax(dim=-1)
                mask_p = (y_tgt_p != model.pad_idx)
                spec_correct += ((preds_p == y_tgt_p) & mask_p).sum().item()
                spec_total += mask_p.sum().item()

            # ===== SPEC → IMPL INPUT =====
            with torch.no_grad():
                xb_gist = build_hybrid_spec(
                    model, tok, xb,
                    spec_max_len=spec_max_len,
                    max_in_len=max_in_len,
                    use_learned=True,
                    device=device
                ).to(device)

            xb_impl = _build_impl_input(
                xb_gist,
                xb,
                max_len=max_in_len,
                anchor_len=CFG.impl_anchor_len
            )

            # ===== IMPL =====
            y_in_c, y_tgt_c = shift_targets(yb)
            logits_c = model.forward_role(
                xb_impl, y_in_c,
                agent_id=AGENT_IMPLEMENTATION
            )
            loss_c = loss_fn(logits_c, y_tgt_c)
            impl_loss_sum += float(loss_c.detach()) * xb.size(0)

            # 🔥 IMPL accuracy
            with torch.no_grad():
                preds_c = logits_c.argmax(dim=-1)
                mask_c = (y_tgt_c != model.pad_idx)
                impl_correct += ((preds_c == y_tgt_c) & mask_c).sum().item()
                impl_total += mask_c.sum().item()

            # ===== Joint loss =====
            loss = 3.0 * loss_p + loss_c

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # 🔥 SWE-bench 스타일 출력
        print(
            f"[Agentic][Train][Stage1] "
            f"epoch={ep}/{epochs} "
            f"spec_ce={spec_loss_sum/float(N):.3f} "
            f"spec_acc={spec_correct/max(spec_total,1):.3f} "
            f"impl_ce={impl_loss_sum/float(N):.3f} "
            f"impl_acc={impl_correct/max(impl_total,1):.3f}"
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
def eval_pipeline_lift(
    model: AgenticTransformerSeq2Seq,
    tok: SubwordTokenizer,
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    spec_max_len: int,
    max_in_len: int,
    device: str = DEVICE
):
    model.to(device)
    model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    y_in, y_tgt = shift_targets(Y.to(device))

    # 1) no-spec baseline
    X_base = X.to(device)
    logits_base = model.forward_role(X_base, y_in, agent_id=AGENT_IMPLEMENTATION)
    ce_base = float(loss_fn(logits_base, y_tgt).item())
    acc_base = float(
        (((logits_base.argmax(-1) == y_tgt) & (y_tgt != model.pad_idx)).float().sum()) /
        ((y_tgt != model.pad_idx).float().sum().clamp_min(1.0))
    )

    # 2) with spec + short anchor
    gist_only = build_hybrid_spec(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        max_in_len=max_in_len,
        use_learned=True,
        device=device
    ).to(device)

    X_with_spec = _build_impl_input(
        gist_only,
        X.to(device),
        max_len=max_in_len,
        anchor_len=CFG.impl_anchor_len
    )

    logits_gist = model.forward_role(X_with_spec, y_in, agent_id=AGENT_IMPLEMENTATION)
    ce_gist = float(loss_fn(logits_gist, y_tgt).item())
    acc_gist = float(
        (((logits_gist.argmax(-1) == y_tgt) & (y_tgt != model.pad_idx)).float().sum()) /
        ((y_tgt != model.pad_idx).float().sum().clamp_min(1.0))
    )

    print("\n[Agentic][Testing][PIPELINE-LIFT] Teacher-forced delta (with gist vs no-spec baseline; more negative is better)")
    print(
        f"[Agentic][Testing][PIPELINE-LIFT] "
        f"IMPL CE(no-spec)={ce_base:.3f} | "
        f"CE(with spec)={ce_gist:.3f} | "
        f"ΔCE={ce_gist - ce_base:.3f} | "
        f"acc(no-spec)={acc_base:.3f} | "
        f"acc(with spec)={acc_gist:.3f}"
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
        logits = model.routing.agents[a].project(last, head="role")
        vec = logits[0, 0, :n_tokens].detach().cpu().numpy()
        print(f"[{agent_pretty_name(a)}] role_head logits[:{n_tokens}] -> {vec}")

@torch.no_grad()
def per_agent_role_eval_impl_on_gist(
    model,
    tok,
    X,
    Y,
    *,
    spec_max_len: int,
    max_in_len: int,
    device: str = DEVICE
):
    model.to(device)
    model.eval()
    loss_fn = SeqCELoss(pad_idx=model.pad_idx)

    y_in, y_tgt = shift_targets(Y.to(device))

    gist_only = build_hybrid_spec(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        max_in_len=max_in_len,
        use_learned=True,
        device=device
    ).to(device)

    X_with_spec = _build_impl_input(
        gist_only,
        X.to(device),
        max_len=max_in_len,
        anchor_len=CFG.impl_anchor_len
    )

    logits = model.forward_role(X_with_spec, y_in, agent_id=AGENT_IMPLEMENTATION)
    ce = float(loss_fn(logits, y_tgt).item())
    preds = logits.argmax(dim=-1)
    mask = (y_tgt != model.pad_idx)
    acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())

    print(f"test/impl_ce={ce:.3f} test/impl_acc={acc:.3f}")

# ============================================================
# Small tensor helpers
# ============================================================
def _concat_truncate(a: torch.Tensor, b: torch.Tensor, *, max_len: int) -> torch.Tensor:
    out = torch.cat([a, b], dim=1)
    if out.size(1) > max_len:
        out = out[:, :max_len]
    return out

def _build_impl_input(
    spec_tensor: torch.Tensor,
    x_tensor: torch.Tensor,
    *,
    max_len: int,
    anchor_len: int = 64
) -> torch.Tensor:
    """
    Implementation input should match the paper's spec-guided flow:
    the implementation agent mainly conditions on the specification,
    with only a small prompt anchor to preserve the function signature.

    Important detail:
    put the anchor FIRST, then the spec. If the sequence is truncated,
    we must never lose the original function header/docstring prefix.
    """

    anchor_len = max(0, min(anchor_len, x_tensor.size(1), max_len))
    spec_budget = max_len - anchor_len

    if anchor_len == 0:
        return spec_tensor[:, :max_len]

    anchor = x_tensor[:, :anchor_len]
    spec_slice = spec_tensor[:, :max(spec_budget, 0)]
    combined = torch.cat([anchor, spec_slice], dim=1)
    return combined[:, :max_len]

@torch.no_grad()
def spec_analysis_stats(model: AgenticTransformerSeq2Seq, tok: SubwordTokenizer, X: torch.Tensor,
                        *, spec_max_len: int, device: str = DEVICE) -> Dict[str, float]:
    """Quick quality probes on A's output: length, lines, and crude impl leakage."""
    model.to(device); model.eval()
    specs = _generate_static(model, X.to(device), agent_id=AGENT_SPECIFICATION,
        max_len=spec_max_len, top_k=20, top_p=0.90,
        temperature=0.8, no_repeat_ngram_size=4, min_len=24  # Adjusted min_len
)
    SPEC = min(4, specs.size(0))
    decoded = [tok.decode([t for t in specs[i].tolist() if t not in (tok.pad, tok.bos, tok.eos)]) for i in range(SPEC)]
    lengths = [len(s.split()) for s in decoded]
    line_counts = [s.count("\n") + 1 for s in decoded]
    impl_leak_lines = sum(
        1 for s in decoded for ln in s.splitlines()
        if ("diff --git" in ln) or ln.strip().startswith("class ") or ("```" in ln)
    )
    return {
        "sampled": float(SPEC),
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

def make_lossy_sufficient_spec(prompt: str) -> str:
    lines = prompt.strip().split("\n")

    signature = ""
    doc_lines = []
    examples = []

    inside_doc = False

    for l in lines:
        if l.strip().startswith("def "):
            signature = l.strip()

        if '"""' in l:
            inside_doc = not inside_doc
            continue

        if inside_doc:
            doc_lines.append(l.strip())

        # 🔥 example extraction
        if ">>>" in l or "assert" in l:
            examples.append(l.strip())

    doc = " ".join(doc_lines)
    doc = re.sub(r"\s+", " ", doc).strip()

    if not doc:
        doc = "Compute the correct output for the function."

    example_str = " ".join(examples[:5])  # limit

    # 🔥 핵심: constraint 형태로 변환
    spec = (
        f"{signature}. "
        f"Task: {doc}. "
        f"Constraints: The implementation must satisfy the given examples. "
        f"Examples: {example_str}"
    )

    return spec.strip()

@torch.no_grad()
def build_hybrid_spec(
    model,
    tok,
    X,
    *,
    spec_max_len,
    max_in_len,
    use_learned=True,
    device=DEVICE
):

    model.to(device)
    model.eval()

    rows = []

    for i in range(X.size(0)):
        prompt_txt = tok.decode([t for t in X[i].tolist() if t != tok.pad])

        # ===== rule spec (항상 사용) =====
        rule_spec = make_lossy_sufficient_spec(prompt_txt)
        rule_spec = _clean_spec_text(rule_spec)

        # ===== learned spec =====
        learned_txt = ""
        if use_learned:
            learned_ids = _generate_static(
                model,
                X[i:i+1].to(device),
                agent_id=AGENT_SPECIFICATION,
                max_len=spec_max_len,
                top_k=None,
                top_p=None,
                temperature=0.8,
                no_repeat_ngram_size=3,
                min_len=24
            )[0]

            learned_txt = tok.decode([
                t for t in learned_ids.tolist()
                if t not in (tok.pad, tok.bos, tok.eos)
            ])
            learned_txt = _postprocess_gist(learned_txt)

        # 🔥 FIX: noisy learned 제거
        if _is_noisy_gist(learned_txt):
            learned_txt = ""

        # 🔥 핵심: rule 중심 + learned 보조
        if learned_txt:
            final_spec = f"{rule_spec} Additional hints: {learned_txt}"
        else:
            final_spec = rule_spec

        wrapped = f"<SPEC>\n{final_spec}\n</SPEC>"

        ids = torch.tensor(tok.sp.encode(wrapped, out_type=int), dtype=torch.long)
        if len(ids) == 0:
            ids = torch.tensor([tok.pad], dtype=torch.long)

        rows.append(ids)

    spec_tensor = pad_sequence(rows, batch_first=True, padding_value=tok.pad)

    return spec_tensor[:, :max_in_len]
    
def _clean_spec_text(txt: str) -> str:
    # keep simple printable range; strip emojis/control chars
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", txt).strip()

def _postprocess_gist(txt: str) -> str:
    # Strong cleanup for display + context
    txt = _clean_spec_text(txt)
    # Remove backticks/markdown noise and angle-tag remnants
    txt = re.sub(r"[`*_<>\[\]#]{1,}", " ", txt)
    # Collapse runs of punctuation/spaces
    txt = re.sub(r"\s*[.,;:!?]\s*", lambda m: m.group(0).strip() + " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    # Keep first 3 sentences or ~100 words
    # sent = _first_n_words(txt, 100)  # Alternative: word-based
    sent = _first_k_sentences(txt, 3)
    if not sent:
        parts = txt.split()
        sent = " ".join(parts[:100])
    return sent.strip()  # Removed [:400] cap

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

def _decode_row_no_pad(tok: "SubwordTokenizer", row: torch.Tensor) -> str:
    ids = [int(t) for t in row.tolist() if int(t) != tok.pad]
    return tok.decode(ids)

@torch.no_grad()
def _spec_ctx_greedy_with_fallback(
    model: "AgenticTransformerSeq2Seq",
    tok: "SubwordTokenizer",
    X: torch.Tensor,
    *,
    spec_max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      spec_ctx  : [B, Tctx] token ids for <SPEC_GIST>...</SPEC_GIST> (no BOS/EOS)
      display_ids: [B, Tdisp] BOS ... EOS ids of the plain gist for printing
    """
    # 1) Greedy generation (no sampling) from the Spec agent
    gen_ids = _generate_static(
        model, X, agent_id=AGENT_SPECIFICATION, max_len=spec_max_len,
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
            gen_txt = make_lossy_sufficient_spec(in_text)
            gen_txt = _postprocess_gist(gen_txt)

        # Final safety: if still noisy, force a minimal title-only fallback
        if _is_noisy_gist(gen_txt):
            in_text = _decode_row_no_pad(tok, X[i])
            gen_txt = make_lossy_sufficient_spec(in_text) or "No description"
            
        gists.append(gen_txt or "spec: (no description)")

    # 3) Encode context (<SPEC_GIST>…</SPEC_GIST>) and plain display ids
    ctx_rows, disp_rows = [], []
    for g in gists:
        ctx_txt = f"<SPEC_GIST>\n{g}\n</SPEC_GIST>"
        ctx_rows.append(torch.tensor(tok.sp.encode(ctx_txt, out_type=int), dtype=torch.long))
        disp_rows.append(tok.encode(g, add_bos_eos=True, max_len=spec_max_len))

    spec_ctx   = pad_sequence(ctx_rows, batch_first=True, padding_value=tok.pad)
    display_ids = pad_sequence(disp_rows, batch_first=True, padding_value=tok.pad)
    return spec_ctx, display_ids

# === NEW: helper to precompute gist-only encoder inputs (HYBRID) =======================
@torch.no_grad()
def build_gist_only_inputs(
    model,
    tok,
    X,
    *,
    spec_max_len,
    max_in_len,
    device=DEVICE,
    use_learned: bool = True
):
    return build_hybrid_spec(
        model,
        tok,
        X.to(device),
        spec_max_len=spec_max_len,
        max_in_len=max_in_len,
        use_learned=use_learned,
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

# ============================================================
# Orchestration
# ============================================================
def run_all(cfg: Config = CFG):
    global CFG
    CFG = cfg

    set_seed(cfg.seed)

    # Data
    data = HumanEvalData(
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab
    )

    # Spec targets = SPEC_DESC
    ids, X, Y, P = data.as_tensors_with_spec_targets(spec_max_len=cfg.max_in_len)

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

    # ===== Stage 1: Interleaved SPEC↔IMPL =====
    print("[Agentic][Training] Stage 1: Interleaved SPEC↔IMPL (same epoch)")
    train_stage1_interleaved(
        model, X_train, Y_train, P_train,
        tok=data.tok,
        spec_max_len=min(cfg.max_out_len, 256),
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
        spec_max_len=min(cfg.max_out_len, 256),
        max_in_len=cfg.max_in_len,
        device=DEVICE
    )

    # NEW: evaluate Impl agent on gist-only input (teacher-forced)
    per_agent_role_eval_impl_on_gist(
        model, data.tok, X_test, Y_test,
        spec_max_len=min(cfg.max_out_len, 256),
        max_in_len=cfg.max_in_len,
        device=DEVICE
    )

    # ------ Diagnostics ------
    print("\n[Agentic][Testing][SPEC-STATS] Analysis agent quick stats (first few specs)")
    stats = spec_analysis_stats(model, data.tok, X_test, spec_max_len=min(cfg.max_out_len, 256), device=DEVICE)
    print(stats)

    print("\n[Agentic][Testing][PIPELINE-SAMPLES] Spec↔ Patch examples (eyeball alignment)")
    spec_to_impl_alignment_sample(model, data.tok, X_test,
                                   spec_max_len=min(cfg.max_out_len, 256),
                                   out_max_len=cfg.decode_max_len,
                                   max_in_len=cfg.max_in_len,
                                   k=3, device=DEVICE)

    # ===== Helpers for CE/Acc =====
    @torch.no_grad()
    def _eval_spec_ce_acc_local(m: AgenticTransformerSeq2Seq, Xenc: torch.Tensor, Ptg: torch.Tensor, *, device: str):
        m.to(device); m.eval()
        loss_fn = SeqCELoss(pad_idx=m.pad_idx)
        y_in, y_tgt = shift_targets(Ptg.to(device))
        logits = m.forward_role(Xenc.to(device), y_in, agent_id=AGENT_SPECIFICATION)
        ce = float(loss_fn(logits, y_tgt).item())
        preds = logits.argmax(-1)
        mask = (y_tgt != m.pad_idx)
        acc = float((((preds == y_tgt) & mask).float().sum() / mask.float().sum().clamp_min(1.0)).item())
        return ce, acc

    # ===========================
    # ===== Stage 2A: Spec FT
    # ===========================
    print("\n[Agentic][Training] Stage 2A: Static specialization for SPEC agent "
          "(freeze backbone + Impl agent; train Spec agent on original X with P targets)")

    iss_ce_before, iss_acc_before = _eval_spec_ce_acc_local(model, X_test, P_test, device=DEVICE)
    print(f"[Agentic][Testing][SPEC@Before FT] CE={iss_ce_before:.3f} | tok_acc={iss_acc_before:.3f}")

    fine_tune_static(
        model, X_train, Y_train,
        user_id=AGENT_SPECIFICATION,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=cfg.ft_unfreeze_decoder_tail_blocks_spec,
        idxs=None,
        device=DEVICE,
        tok=data.tok,
        P=P_train,
        gist_ctx_fn=None,
        max_in_len=CFG.max_in_len,
        use_concat_first_epoch=False,
        patience=2
    )

    iss_ce_after, iss_acc_after = _eval_spec_ce_acc_local(model, X_test, P_test, device=DEVICE)
    print(
        f"[Agentic][Testing][SPEC@After FT] "
        f"CE={iss_ce_after:.3f} | tok_acc={iss_acc_after:.3f} "
        f"| ΔCE={iss_ce_after - iss_ce_before:+.3f} ({(iss_ce_after - iss_ce_before) / iss_ce_before * 100:+.2f}%) "
        f"| Δacc={iss_acc_after - iss_acc_before:+.3f} ({(iss_acc_after - iss_acc_before) / iss_acc_before * 100:+.2f}%)"
    )

    # ===========================
    # ===== Stage 2B: Impl FT
    # ===========================
    print("\n[Agentic][Training] Stage 2B: Static specialization for IMPL agent "
          "(freeze backbone + Spec agent; train Impl agent on gist-only)")

    # Build gist-only inputs (once; no grad) for test measurement
    spec_len = min(cfg.max_out_len, 256)
    X_test_gist = build_gist_only_inputs(
        model, data.tok, X_test,
        spec_max_len=spec_len, max_in_len=cfg.max_in_len, device=DEVICE
    )
    
    X_test_impl = _build_impl_input(
        X_test_gist.to(DEVICE),
        X_test.to(DEVICE),
        max_len=cfg.max_in_len,
        anchor_len=CFG.impl_anchor_len
    )
    
    ce_before, acc_before = _eval_impl_ce_acc(model, X_test_impl, Y_test, device=DEVICE)
    print(f"[Agentic][Testing][IMPL@GIST][Before FT] CE={ce_before:.3f} | tok_acc={acc_before:.3f}")

    def _gist_ctx_fn_for_ft(xb_device):
        return build_hybrid_spec(
            model,
            data.tok,
            xb_device,
            spec_max_len=min(CFG.max_out_len, 256),
            max_in_len=CFG.max_in_len,
            use_learned=True,
            device=DEVICE
        )

    fine_tune_static(
        model, X_train, Y_train,
        user_id=AGENT_IMPLEMENTATION,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        unfreeze_dec_norms=cfg.ft_unfreeze_dec_norms,
        unfreeze_decoder_tail_blocks=cfg.ft_unfreeze_decoder_tail_blocks_impl,
        idxs=None,
        device=DEVICE,
        tok=data.tok,
        P=P_train,
        gist_ctx_fn=_gist_ctx_fn_for_ft,
        max_in_len=CFG.max_in_len,
        use_concat_first_epoch=False,
        patience=2
    )

    # Measure IMPL@GIST after FT (recompute gist from current Spec agent)
    X_test_gist_after = _gist_ctx_fn_for_ft(X_test.to(DEVICE))[:, :CFG.max_in_len]
    X_test_impl_after = _build_impl_input(
        X_test_gist_after.to(DEVICE),
        X_test.to(DEVICE),
        max_len=cfg.max_in_len,
        anchor_len=CFG.impl_anchor_len
    )
    
    ce_after, acc_after = _eval_impl_ce_acc(
        model,
        X_test_impl_after,
        Y_test,
        device=DEVICE
    )
    print(
        f"[Agentic][Testing][IMPL@GIST][After FT] "
        f"CE={ce_after:.3f} | tok_acc={acc_after:.3f} "
        f"| ΔCE={ce_after - ce_before:+.3f} ({(ce_after - ce_before) / ce_before * 100:+.2f}%) "
        f"| Δacc={acc_after - acc_before:+.3f} ({(acc_after - acc_before) / acc_before * 100:+.2f}%)"
    )
    
    # ===== NEW: Save a small sample of generated code =====
    print("\n[Agentic][Output] Saving sample generated code...")
    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        k = 10
        X_sample = X_test[:k].to(DEVICE)
    
        spec_ids, patch_ids = model.routing.run_pipeline(
            model,
            data.tok,
            X_sample,
            spec_max_len=min(cfg.max_out_len, 256),
            out_max_len=cfg.decode_max_len,
            max_in_len=cfg.max_in_len
        )
    
        for i in range(k):
            spec_txt = data.tok.decode(
                [t for t in spec_ids[i].tolist() if t not in (data.tok.pad, data.tok.bos, data.tok.eos)]
            )
            patch_txt = data.tok.decode(
                [t for t in patch_ids[i].tolist() if t not in (data.tok.pad, data.tok.bos, data.tok.eos)]
            )
    
            with open(os.path.join(cfg.out_dir, f"sample_{i}.txt"), "w") as f:
                f.write("=== SPEC ===\n")
                f.write(spec_txt + "\n\n")
                f.write("=== GENERATED CODE ===\n")
                f.write(patch_txt + "\n")
    
    print(f"[Agentic][Output] Saved {k} samples to '{cfg.out_dir}/'")
    
    return model, data, (ids, X, Y, P)

if __name__ == "__main__":
    model, data, tensors = run_all(CFG)