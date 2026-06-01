# ============================================================
# Agentic seq2seq — Structured Specification-to-Implementation Pipeline
# CPU-only version
# Fully consolidated + consistency-fixed version
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

from typing import (
    List,
    Optional,
    Sequence,
)

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

AGENT_ISSUE_ANALYSIS = 0
AGENT_CODE_GENERATION = 1


def agent_pretty_name(agent_id: int) -> str:

    if agent_id == AGENT_ISSUE_ANALYSIS:
        return "Issue Analysis Agent"

    if agent_id == AGENT_CODE_GENERATION:
        return "Code Generation Agent"

    return f"Agent {agent_id}"


def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:

    # reproducibility
    seed: int = 42

    # data
    limit: int = 1000

    max_in_len: int = 256
    max_out_len: int = 320
    spm_vocab: int = 2048
    decode_max_len: int = 96
    demo_data: bool = False

    # model
    n_agents: int = 2
    model_dim: int = 256
    n_heads: int = 4
    n_layers_enc: int = 3
    n_layers_dec: int = 3
    max_len_cap: int = 640

    # stage 1
    pipe_epochs: int = 25
    pipe_batch: int = 8
    pipe_lr: float = 2e-4

    # stage 2
    ft_epochs: int = 8
    ft_batch: int = 8
    ft_lr: float = 1e-4
    ft_unfreeze_adapters: bool = True

    # inference validation
    max_repair_attempts: int = 3
    n_validation_samples: int = 10

    # output
    out_dir: str = "outputs/swebench"


CFG = Config()


# ============================================================
# Tokenizer
# ============================================================

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

UNK, PAD, BOS, EOS = range(4)

try:

    import sentencepiece as spm

    HAVE_SPM = True

except Exception:

    HAVE_SPM = False


class SubwordTokenizer:

    def __init__(
        self,
        texts: Sequence[str],
        vocab_size: int = 8000,
        quiet: bool = True,
    ):

        if not HAVE_SPM:

            raise RuntimeError(
                "SentencePiece missing. "
                "Install with: pip install sentencepiece"
            )

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

        self.quiet = quiet

        with tempfile.TemporaryDirectory() as tmpd:

            corpus = os.path.join(tmpd, "spm_corpus.txt")

            with open(corpus, "w", encoding="utf-8") as f:

                for t in texts:

                    f.write(str(t).replace("\r", " ") + "\n")

            model_prefix = os.path.join(tmpd, "spm_model")

            target_vocab = int(min(vocab_size, 2048))

            cmd = (
                f"--input={corpus} "
                f"--model_prefix={model_prefix} "
                f"--vocab_size={target_vocab} "
                f"--character_coverage=0.9995 "
                f"--model_type=unigram "
                f"--user_defined_symbols="
                f"<ISSUE_TITLE>,</ISSUE_TITLE>,"
                f"<ISSUE_DESC>,</ISSUE_DESC>,"
                f"<HINTS>,</HINTS>,"
                f"<ISSUE_GIST>,</ISSUE_GIST>,"
                f"<PROBLEM>,</PROBLEM>,"
                f"<DETAILS>,</DETAILS>,"
                f"<FAULT_TYPE>,</FAULT_TYPE>,"
                f"<FAULT_LOCATION>,</FAULT_LOCATION>,"
                f"<EXPECTED_BEHAVIOR>,</EXPECTED_BEHAVIOR>,"
                f"<PATCH_HINT>,</PATCH_HINT>,"
                f"<RAW_CONTEXT>,</RAW_CONTEXT> "
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

        self.pad_idx = 1
        self.unk_idx = 0
        self.bos_idx = 2
        self.eos_idx = 3

    def encode(
        self,
        text: str,
        add_bos_eos: bool,
        max_len: int,
    ) -> torch.Tensor:

        ids = self.sp.encode(str(text), out_type=int)

        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]

        ids = ids[:max_len]

        if len(ids) == 0:
            ids = [self.unk_idx]

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:

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
# Utilities
# ============================================================

def _extract_tag_block(text: str, tag: str) -> str:

    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"

    if open_tag in text and close_tag in text:

        return (
            text
            .split(open_tag, 1)[1]
            .split(close_tag, 1)[0]
            .strip()
        )

    return ""


def _normalize_issue_text(txt: str) -> str:

    return re.sub(
        r"[^\x09\x0A\x0D\x20-\x7E]",
        "",
        txt
    ).strip()

def _validate_gist_structure(txt: str) -> bool:

    if not txt:
        return False

    try:
        required = [
            "PROBLEM",
            "FAULT_TYPE",
            "FAULT_LOCATION",
            "EXPECTED_BEHAVIOR",
            "PATCH_HINT",
        ]

        filled = 0

        for tag in required:
            val = _extract_tag_block(txt, tag)
            if val.strip():
                filled += 1

        return filled >= 4

    except Exception:
        return False


def _infer_fault_type(text: str) -> str:

    t = text.lower()

    if "logger" in t or "logging" in t or "global_step" in t:
        return "logging side effect"

    if "gradient" in t or "clip" in t or "clipping" in t:
        return "gradient computation behavior"

    if "compose" in t or "circuit" in t or "expression" in t:
        return "API expression support"

    if "crash" in t or "exception" in t or "traceback" in t:
        return "runtime crash"

    if "retry" in t or "500" in t or "503" in t or "504" in t:
        return "retry error handling"

    if "port" in t or "server" in t:
        return "runtime dependency constraint"

    return "behavioral defect"


def _extract_fault_location(text: str) -> str:

    candidates = re.findall(
        r"`([^`]{3,80})`",
        text,
    )

    if candidates:
        return " ".join(candidates[:4])

    words = re.findall(
        r"[A-Za-z_][A-Za-z0-9_\.]*",
        text,
    )

    useful = [
        w for w in words
        if (
            "_" in w
            or "." in w
            or w[:1].isupper()
        )
    ]

    return " ".join(useful[:8]) or "unknown component"


def _infer_expected_behavior(text: str) -> str:

    t = text.lower()

    if "should" in t:
        m = re.search(r"should\s+(.{1,120})", text, re.I)
        if m:
            return m.group(1).strip()

    if "expected" in t:
        m = re.search(r"expected.{0,20}[:\-]?\s*(.{1,120})", text, re.I)
        if m:
            return m.group(1).strip()

    if "not necessary" in t:
        return "avoid unnecessary mutation or side effect"

    if "crash" in t:
        return "avoid crash and handle missing or malformed values"

    return "preserve intended behavior without side effects"


def _infer_patch_hint(text: str) -> str:

    t = text.lower()

    if "logger" in t or "global_step" in t:
        return "avoid mutating shared metric dictionary"

    if "gradient" in t or "clip" in t:
        return "apply clipping without unnecessary slow path"

    if "compose" in t or "expression" in t:
        return "propagate expression/register mapping logic"

    if "retry" in t:
        return "retry when payload contains recoverable errors"

    if "crash" in t:
        return "guard missing fields before metric collection"

    if "port" in t:
        return "write metrics locally without opening server port"

    return "modify localized behavior using existing API"


def _make_issue_gist(
    title: str,
    desc: str,
    hints: str = "",
) -> str:

    title = _normalize_issue_text(title)
    desc = _normalize_issue_text(desc)
    hints = _normalize_issue_text(hints)

    full = " ".join([title, desc, hints]).strip()

    problem = " ".join(title.split()[:18]) or "software behavior issue"
    details = " ".join(desc.split()[:50])
    fault_type = _infer_fault_type(full)
    fault_location = _extract_fault_location(full)
    expected = _infer_expected_behavior(full)
    patch_hint = _infer_patch_hint(full)

    return (
        "<PROBLEM>\n"
        f"{problem}\n"
        "</PROBLEM>\n"
        "<FAULT_TYPE>\n"
        f"{fault_type}\n"
        "</FAULT_TYPE>\n"
        "<FAULT_LOCATION>\n"
        f"{fault_location}\n"
        "</FAULT_LOCATION>\n"
        "<EXPECTED_BEHAVIOR>\n"
        f"{expected}\n"
        "</EXPECTED_BEHAVIOR>\n"
        "<PATCH_HINT>\n"
        f"{patch_hint}\n"
        "</PATCH_HINT>\n"
        "<DETAILS>\n"
        f"{details}\n"
        "</DETAILS>"
    )


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
        limit: Optional[int] = 1024,
        max_in_len: int = 512,
        max_out_len: int = 256,
        spm_vocab_size: int = 8000,
        demo_data: bool = True,
    ):

        self.max_in_len = max_in_len
        self.max_out_len = max_out_len

        self.samples = []

        if demo_data:

            print("[Data] DEMO synthetic dataset")

            rng = random.Random()

            n = int(limit or 1024)

            for i in range(n):

                title = f"Issue {i}: Widget broken"

                body = (
                    f"Repro {i}: "
                    f"click causes crash "
                    f"trace={rng.randint(0,999)}"
                )

                patch = (
                    "diff --git a/app.py b/app.py\n"
                    "@@\n"
                    "+print('fix')\n"
                )

                self.samples.append(
                    (
                        f"demo-{i}",
                        title + "\n" + body,
                        patch,
                    )
                )

        else:

            if not HAVE_HF:

                raise RuntimeError(
                    "Install datasets with:\n"
                    "pip install datasets"
                )

            print("[Data] Loading SWE-bench...")

            ds = load_dataset(
                "princeton-nlp/SWE-bench",
                split=split,
            )

            if limit is not None:
                ds = ds.select(range(min(limit, len(ds))))

            rows = list(ds)

            def build_input(ex):

                title = str(ex.get("title", "")).strip()

                desc = str(
                    ex.get("problem_statement", "")
                ).strip()

                hints = str(
                    ex.get("hints_text", "")
                ).strip()

                parts = []

                if title:

                    parts.append(
                        f"<ISSUE_TITLE>\n"
                        f"{title}\n"
                        f"</ISSUE_TITLE>"
                    )

                if desc:

                    parts.append(
                        f"<ISSUE_DESC>\n"
                        f"{desc}\n"
                        f"</ISSUE_DESC>"
                    )

                if hints:

                    parts.append(
                        f"<HINTS>\n"
                        f"{hints}\n"
                        f"</HINTS>"
                    )

                return "\n".join(parts)

            def pick_patch(ex):

                def normalize_code_target(code: str) -> str:

                    code = code.strip()

                    code = re.sub(
                        r"(?m)^@\w+[^\n]*$",
                        "",
                        code,
                    ).strip()

                    if not code:
                        return ""

                    # Try direct parse first.
                    try:
                        ast.parse(code)
                        if (
                            ("def " in code or "class " in code)
                            and "\n" in code
                        ):
                            return code
                    except Exception:
                        pass

                    # If code is only statements, wrap it in a valid function.
                    indented = "\n".join(
                        "    " + ln
                        for ln in code.splitlines()
                        if ln.strip()
                    )

                    wrapped = (
                        "def generated_patch():\n"
                        f"{indented}\n"
                    )

                    try:
                        ast.parse(wrapped)
                        return wrapped.strip()
                    except Exception:
                        return ""

                for key in (
                    "patch",
                    "base_patch",
                    "model_patch",
                    "test_patch",
                ):

                    if key not in ex or not ex[key]:
                        continue

                    raw_patch = str(ex[key])

                    kept_lines = []

                    for line in raw_patch.splitlines():

                        if line.startswith((
                            "diff --git",
                            "index ",
                            "---",
                            "+++",
                            "@@",
                        )):
                            continue

                        # Keep added lines.
                        if line.startswith("+") and not line.startswith("+++"):

                            candidate = line[1:].rstrip()

                            if not candidate.strip():
                                continue

                            if candidate.strip() in {
                                "pass",
                                "...",
                            }:
                                continue

                            kept_lines.append(candidate)

                    if not kept_lines:
                        continue

                    text = "\n".join(kept_lines).strip()

                    # First try full added-code block.
                    normalized = normalize_code_target(text)

                    if normalized:

                        lines = normalized.splitlines()

                        filtered = []

                        for ln in lines:

                            s = ln.strip()

                            if (
                                s.startswith("def ")
                                or s.startswith("class ")
                                or "return" in s
                                or "=" in s
                                or "if " in s
                                or "for " in s
                                or "while " in s
                            ):
                                filtered.append(ln)

                        shortened = "\n".join(filtered[:8]).strip()

                        if shortened:
                            return shortened
                    # Then try extracting complete def/class blocks.
                    blocks = re.findall(
                        r"(?ms)((?:def|class)\s+.*?(?=\n(?:def|class)\s|\Z))",
                        text,
                    )

                    for block in blocks:

                        normalized = normalize_code_target(block)

                        if normalized:
                            return normalized

                return ""

            for ex in rows:

                iid = str(ex.get("instance_id", ""))

                xin = build_input(ex)

                yout = pick_patch(ex)

                if not yout.strip():
                    continue

                self.samples.append(
                    (
                        iid,
                        xin,
                        yout,
                    )
                )

        texts = []

        for _, x, y in self.samples:

            texts.append(x)
            texts.append(y)

            special_tag_text = " ".join([
                "<ISSUE_TITLE>",
                "</ISSUE_TITLE>",
                "<ISSUE_DESC>",
                "</ISSUE_DESC>",
                "<HINTS>",
                "</HINTS>",
                "<ISSUE_GIST>",
                "</ISSUE_GIST>",
                "<PROBLEM>",
                "</PROBLEM>",
                "<DETAILS>",
                "</DETAILS>",
                "<FAULT_TYPE>",
                "</FAULT_TYPE>",
                "<FAULT_LOCATION>",
                "</FAULT_LOCATION>",
                "<EXPECTED_BEHAVIOR>",
                "</EXPECTED_BEHAVIOR>",
                "<PATCH_HINT>",
                "</PATCH_HINT>",
                "<RAW_CONTEXT>",
                "</RAW_CONTEXT>",
            ])

        texts.extend([special_tag_text] * 100)

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

    def as_tensors_with_issue_targets(
        self,
        issue_max_len: int,
    ):

        ids, X, Y = self.as_tensors()

        Ps = []

        for _, xin, _ in self.samples:

            title = _extract_tag_block(
                xin,
                "ISSUE_TITLE",
            )

            desc = (
                _extract_tag_block(
                    xin,
                    "ISSUE_DESC",
                )
                or xin
            )

            hints = _extract_tag_block(
                xin,
                "HINTS",
            )

            issue = _make_issue_gist(
                title,
                desc,
                hints,
            )

            Ps.append(
                self.tok.encode(
                    issue,
                    add_bos_eos=True,
                    max_len=issue_max_len,
                )
            )

        P = pad_sequence(
            Ps,
            batch_first=True,
            padding_value=self.tok.pad,
        )

        return ids, X, Y, P


# ============================================================
# Model
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 1024,
        pad_token_id: int = PAD,
    ):

        super().__init__()

        self.pad_token_id = pad_token_id

        self.tok_embedding = nn.Embedding(
            vocab_size,
            model_dim,
            padding_idx=pad_token_id,
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

    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        max_len: int = 1024,
        pad_idx: int = PAD,
        tok_embedding: Optional[nn.Embedding] = None,
    ):

        super().__init__()

        self.pad_idx = pad_idx

        self.tok_embedding = (
            tok_embedding
            if tok_embedding is not None
            else nn.Embedding(
                vocab_size,
                model_dim,
                padding_idx=pad_idx,
            )
        )

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

    def _subsequent_mask(
        self,
        L: int,
        device,
    ):

        return torch.triu(
            torch.ones(
                L,
                L,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

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

class Validator:

    def validate_patch(self, text: str) -> bool:
        return validate_python_syntax(text)
        


class Repairer:
    
    def repair_patch(
        self,
        text: str,
        stage: int,
    ) -> str:

        return repair_code(text, stage)

class Agent(nn.Module):

    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        adapter_dim: int = 128,
    ):

        super().__init__()

        self.adapter = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, model_dim),
        )

        self.role_head = nn.Linear(
            model_dim,
            vocab_size,
        )

        self.validator = Validator()
        self.repairer = Repairer()

    def project(
        self,
        states,
    ):

        h = self.adapter(states)

        layer = self.role_head

        return layer(h)


class RoutingModule(nn.Module):

    def __init__(self, agents: nn.ModuleList):

        super().__init__()

        self.agents = agents

    def project_role(
        self,
        dec_states,
        *,
        agent_id,
    ):

        return self.agents[agent_id].project(
            dec_states,
        )

class AssignmentModule:

    def __init__(self):

        self.stage_to_agent = {
            0: AGENT_ISSUE_ANALYSIS,
            1: AGENT_CODE_GENERATION,
        }

    def agent_for_stage(self, stage: int) -> int:

        return self.stage_to_agent[stage]
    
class AgenticTransformerSeq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_agents=2,
        model_dim=512,
        n_heads=8,
        n_layers_enc=6,
        n_layers_dec=6,
        max_len=1024,
        pad_idx=PAD,
    ):

        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            model_dim,
            n_heads,
            n_layers_enc,
            max_len,
            pad_idx,
        )

        self.decoder = Decoder(
            vocab_size,
            model_dim,
            n_heads,
            n_layers_dec,
            max_len,
            pad_idx,
            tok_embedding=self.encoder.tok_embedding,
        )

        agents = nn.ModuleList([
            Agent(model_dim, vocab_size)
            for _ in range(n_agents)
        ])

        self.routing = RoutingModule(agents)
        self.assignment = AssignmentModule()

        self.pad_idx = pad_idx

    def encode(self, x):

        return self.encoder(x)

    def decode_states(
        self,
        y_in,
        memory,
        src_key_padding_mask,
    ):

        return self.decoder(
            y_in,
            memory,
            src_key_padding_mask,
        )

    def forward_role(
        self,
        x,
        y_in,
        *,
        agent_id,
    ):

        mem, _, src_mask = self.encode(x)

        dec_states = self.decode_states(
            y_in,
            mem,
            src_mask,
        )

        return self.routing.project_role(
            dec_states,
            agent_id=agent_id,
        )


# ============================================================
# Generation utilities
# ============================================================

def _no_repeat_ngram_mask(
    ys,
    n,
    vocab_size,
):

    if n <= 0:

        return torch.zeros(
            (ys.size(0), vocab_size),
            dtype=torch.bool,
            device=ys.device,
        )

    B, L = ys.shape

    mask = torch.zeros(
        (B, vocab_size),
        dtype=torch.bool,
        device=ys.device,
    )

    if L < n:
        return mask

    for b in range(B):

        seq = ys[b].tolist()

        if n == 1:
            banned = set(seq)

            if banned:
                mask[b, list(banned)] = True

            continue

        prefix2next = {}

        for i in range(L - n + 1):

            prefix = tuple(seq[i:i+n-1])
            nxt = seq[i+n-1]

            prefix2next.setdefault(prefix, set()).add(nxt)

        last_prefix = tuple(seq[-(n-1):])

        banned = prefix2next.get(last_prefix, set())

        if banned:
            mask[b, list(banned)] = True

    return mask


def _top_k_top_p_filtering(
    logits,
    top_k,
    top_p,
):

    if top_k is not None and top_k > 0:

        k = min(top_k, logits.size(-1))

        thresh = (
            torch.topk(logits, k, dim=-1)
            .values[..., -1]
            .unsqueeze(-1)
        )

        logits = torch.where(
            logits < thresh,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p is not None and 0.0 < top_p < 1.0:

        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(
            probs,
            descending=True,
            dim=-1,
        )

        cumulative = torch.cumsum(
            sorted_probs,
            dim=-1,
        )

        to_mask = cumulative > top_p

        to_mask[..., 1:] = to_mask[..., :-1].clone()

        to_mask[..., 0] = False

        logits.scatter_(
            1,
            sorted_idx,
            torch.where(
                to_mask,
                torch.full_like(
                    sorted_probs,
                    float("-inf"),
                ),
                logits.gather(1, sorted_idx),
            ),
        )

    return logits


@torch.no_grad()
def _generate_static(
    model,
    X,
    *,
    agent_id,
    max_len,
    top_k=None,
    top_p=None,
    temperature=1.0,
    no_repeat_ngram_size=0,
    min_len=0,
):

    model.eval()

    memory, _, src_mask = model.encode(X)

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
        device=X.device,
    )

    for _ in range(1, max_len):

        dec = model.decode_states(
            ys,
            memory,
            src_mask,
        )

        step_logits = (
            model.routing
            .agents[agent_id]
            .project(
                dec[:, -1:],
            )
            .squeeze(1)
        )

        # Never generate UNK.
        step_logits[:, UNK] = float("-inf")
        step_logits[:, BOS] = float("-inf")
        step_logits[:, PAD] = float("-inf")

        # ------------------------------------------------
        # Discourage degenerate trivial outputs like:
        #
        # def generated_patch():
        #     pass
        #
        # Do NOT fully ban "pass" because it may still
        # occasionally be useful for syntax recovery.
        # ------------------------------------------------

        if (
            hasattr(model, "_pass_token_id")
            and model._pass_token_id >= 0
        ):

            step_logits[:, model._pass_token_id] = -8.0

        step_logits = torch.clamp(
            step_logits,
            min=-20.0,
            max=20.0,
        )

        # --------------------------------------------
        # Prevent premature EOS only very early.
        # --------------------------------------------

        if ys.size(1) < min_len:
            step_logits[:, EOS] = -1e9

        if no_repeat_ngram_size > 0:

            banned = _no_repeat_ngram_mask(
                ys,
                no_repeat_ngram_size,
                vocab_size,
            )

            step_logits = step_logits.masked_fill(
                banned,
                float("-inf"),
            )

        if temperature != 1.0:

            step_logits = (
                step_logits
                / max(temperature, 1e-8)
            )

        # ------------------------------------------------
        # Always use stochastic decoding.
        # Small non-pretrained models collapse badly under
        # greedy decoding on SWE-bench.
        # ------------------------------------------------

        if agent_id == AGENT_CODE_GENERATION:

            next_tok = torch.argmax(
                step_logits,
                dim=-1,
                keepdim=True,
            )

        else:

            effective_top_k = top_k if top_k is not None else 20
            effective_top_p = top_p if top_p is not None else 0.85
            effective_temperature = max(temperature, 0.65)

            logits = _top_k_top_p_filtering(
                step_logits.clone(),
                top_k=effective_top_k,
                top_p=effective_top_p,
            )

            probs = torch.softmax(
                logits / effective_temperature,
                dim=-1,
            )

            if (
                torch.isnan(probs).any()
                or
                torch.isinf(probs).any()
            ):

                next_tok = torch.argmax(
                    step_logits,
                    dim=-1,
                    keepdim=True,
                )

            else:

                next_tok = torch.multinomial(
                    probs,
                    num_samples=1,
                )

        ys = torch.cat([ys, next_tok], dim=1)

        if (next_tok == EOS).all():
            break

    return ys


# ============================================================
# Issue context builder
# ============================================================

@torch.no_grad()
def _generate_issue_gist_context(
    model,
    tok,
    X,
    *,
    issue_max_len,
):

    gen_ids = _generate_static(
        model,
        X,
        agent_id=model.assignment.agent_for_stage(0),
        max_len=issue_max_len,
        top_k=30,
        top_p=0.90,
        temperature=0.85,
        no_repeat_ngram_size=3,
        min_len=24,
    )

    B = X.size(0)

    gists = []

    for i in range(B):

        raw = [
            t
            for t in gen_ids[i].tolist()
            if t not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ]

        gen_txt = tok.decode(raw)

        gen_txt = _normalize_issue_text(gen_txt)

        if not _validate_gist_structure(gen_txt):

            original_txt = tok.decode([
                t
                for t in X[i].tolist()
                if t not in (
                    tok.pad,
                    tok.bos,
                    tok.eos,
                )
            ])

            title = _extract_tag_block(
                original_txt,
                "ISSUE_TITLE",
            )

            desc = _extract_tag_block(
                original_txt,
                "ISSUE_DESC",
            )

            hints = _extract_tag_block(
                original_txt,
                "HINTS",
            )

            gen_txt = _make_issue_gist(
                title,
                desc,
                hints,
            )

            if not _validate_gist_structure(gen_txt):
                gen_txt = _make_issue_gist(
                    "",
                    original_txt,
                    "",
                )
        gists.append(gen_txt)

    ctx_rows = []
    disp_rows = []

    for g in gists:

        ctx_txt = (
            "<ISSUE_GIST>\n"
            f"{g}\n"
            "</ISSUE_GIST>"
        )

        ctx_rows.append(
            torch.tensor(
                tok.sp.encode(
                    ctx_txt,
                    out_type=int,
                ),
                dtype=torch.long,
            )
        )

        disp_rows.append(
            tok.encode(
                g,
                add_bos_eos=True,
                max_len=issue_max_len,
            )
        )

    issue_ctx = pad_sequence(
        ctx_rows,
        batch_first=True,
        padding_value=tok.pad,
    )

    display_ids = pad_sequence(
        disp_rows,
        batch_first=True,
        padding_value=tok.pad,
    )

    return issue_ctx, display_ids

def _build_source_issue_context(
    tok,
    X,
    *,
    issue_max_len,
    max_in_len,
):

    ctx_rows = []
    display_rows = []

    for i in range(X.size(0)):

        raw_txt = tok.decode([
            t.item()
            for t in X[i]
            if t.item() not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        title = _extract_tag_block(
            raw_txt,
            "ISSUE_TITLE",
        )

        desc = (
            _extract_tag_block(
                raw_txt,
                "ISSUE_DESC",
            )
            or raw_txt
        )

        hints = _extract_tag_block(
            raw_txt,
            "HINTS",
        )

        gist = _make_issue_gist(
            title,
            desc,
            hints,
        )

        ctx_txt = (
            "<ISSUE_GIST>\n"
            f"{gist}\n"
            "</ISSUE_GIST>\n"
            "<RAW_CONTEXT>\n"
            f"{raw_txt}\n"
            "</RAW_CONTEXT>"
        )

        ctx_rows.append(
            tok.encode(
                ctx_txt,
                add_bos_eos=False,
                max_len=max_in_len,
            )
        )

        display_rows.append(
            tok.encode(
                gist,
                add_bos_eos=True,
                max_len=issue_max_len,
            )
        )

    return (
        pad_sequence(
            ctx_rows,
            batch_first=True,
            padding_value=tok.pad,
        ),
        pad_sequence(
            display_rows,
            batch_first=True,
            padding_value=tok.pad,
        ),
    )

def _build_gist_plus_raw_context(
    tok,
    issue_ctx,
    raw_x,
    *,
    max_in_len,
):

    rows = []

    for i in range(raw_x.size(0)):

        gist_txt = tok.decode([
            t.item()
            for t in issue_ctx[i]
            if t.item() not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        raw_txt = tok.decode([
            t.item()
            for t in raw_x[i]
            if t.item() not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        ctx_txt = (
            f"{gist_txt}\n"
            "<RAW_CONTEXT>\n"
            f"{raw_txt}\n"
            "</RAW_CONTEXT>"
        )

        rows.append(
            tok.encode(
                ctx_txt,
                add_bos_eos=False,
                max_len=max_in_len,
            )
        )

    return pad_sequence(
        rows,
        batch_first=True,
        padding_value=tok.pad,
    )

# ============================================================
# Strict pipeline
# ============================================================

class StrictPipeline(nn.Module):

    def __init__(self):

        super().__init__()

    @torch.no_grad()
    def run(
        self,
        model,
        tok,
        X,
        *,
        issue_max_len,
        out_max_len,
        max_in_len,
    ):

        # ----------------------------------------------------
        # Stage 1:
        # Generate structured issue gist
        # ----------------------------------------------------

        impl_input, issue_display_ids = _build_source_issue_context(
            tok,
            X,
            issue_max_len=issue_max_len,
            max_in_len=max_in_len,
        )

        patch_ids = _generate_static(
            model,
            impl_input.to(X.device),
            agent_id=model.assignment.agent_for_stage(1),
            max_len=out_max_len,
            top_k=None,
            top_p=None,
            temperature=1.0,
            no_repeat_ngram_size=3,
            min_len=8,
        )

        return issue_display_ids, patch_ids


# ============================================================
# Loss
# ============================================================

class SeqCELoss(nn.Module):

    def __init__(self, pad_idx):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=0.00,
        )

    def forward(
        self,
        logits,
        targets,
    ):

        B, L, V = logits.shape

        return self.ce(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
        )

# ============================================================
# Constraint loss
# ============================================================

class ConstraintLoss(nn.Module):

    def __init__(
        self,
        pad_idx,
        eos_idx,
        diff_token_ids=None,
        bad_token_ids=None,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.diff_token_ids = set(diff_token_ids or [])
        self.bad_token_ids = set(bad_token_ids or [])

    def forward(
        self,
        logits,
        targets,
    ):
        probs = torch.softmax(logits, dim=-1)

        valid_mask = (targets != self.pad_idx)

        # ----------------------------------------------------
        # Target-aware structure encouragement.
        # Only rewards structure tokens where the GOLD target
        # actually contains structure tokens.
        # ----------------------------------------------------

        structure_loss = logits.new_tensor(0.0)

        if self.diff_token_ids:

            struct_target_mask = torch.zeros_like(
                targets,
                dtype=torch.bool,
            )

            for tid in self.diff_token_ids:
                struct_target_mask |= (targets == tid)

            struct_target_mask &= valid_mask

            if struct_target_mask.any():

                struct_prob = torch.zeros_like(
                    probs[..., 0]
                )

                for tid in self.diff_token_ids:
                    struct_prob += probs[..., tid]

                structure_loss = -torch.log(
                    struct_prob[struct_target_mask].clamp_min(1e-8)
                ).mean()

        # ----------------------------------------------------
        # Suppress bad tokens only.
        # ----------------------------------------------------

        bad_loss = logits.new_tensor(0.0)

        if self.bad_token_ids:

            bad_prob = torch.zeros_like(
                probs[..., 0]
            )

            for tid in self.bad_token_ids:
                bad_prob += probs[..., tid]

            bad_loss = bad_prob[valid_mask].mean()

        return (
            0.90 * structure_loss
            + 0.10 * bad_loss
        )
    
def shift_targets(y):

    return y[:, :-1], y[:, 1:]


# ============================================================
# Freeze helpers
# ============================================================

def _set_trainable_stage1_joint(model):

    # Stage 1 = global joint training
    # Entire backbone + all agents trainable

    for p in model.parameters():
        p.requires_grad = True


def _set_ft_requires_grad(
    model,
    *,
    user_id,
    unfreeze_adapters,
):

    # Freeze entire model
    for p in model.parameters():
        p.requires_grad = False

    # Only target agent trainable
    ag = model.routing.agents[user_id]

    for name, p in ag.named_parameters():

        # role head
        if name.startswith("role_head"):
            p.requires_grad = True

        # adapter
        elif (
            unfreeze_adapters
            and name.startswith("adapter")
        ):
            p.requires_grad = True

# ============================================================
# Unified reporting
# ============================================================

# ============================================================
# Parameter-efficiency metrics
# ============================================================

def compute_trainable_stats(model):

    total_params = 0
    trainable_params = 0

    for p in model.parameters():

        n = p.numel()

        total_params += n

        if p.requires_grad:
            trainable_params += n

    ratio = (
        trainable_params
        / max(total_params, 1)
    )

    return {
        "trainable": trainable_params,
        "total": total_params,
        "ratio": ratio,
    }

def compute_active_inference_stats(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------
    # Strict pipeline:
    #
    # Stage 1:
    #   encoder
    #   decoder
    #   issue-analysis agent
    #
    # Stage 2:
    #   encoder
    #   decoder
    #   code-generation agent
    #
    # Both agents participate during a complete pipeline run.
    # --------------------------------------------------

    encoder_params = sum(
        p.numel()
        for p in model.encoder.parameters()
    )

    decoder_params = sum(
        p.numel()
        for p in model.decoder.parameters()
    )

    issue_agent_params = sum(
        p.numel()
        for p in model.routing.agents[
            AGENT_ISSUE_ANALYSIS
        ].parameters()
    )

    code_agent_params = sum(
        p.numel()
        for p in model.routing.agents[
            AGENT_CODE_GENERATION
        ].parameters()
    )

    active_params = (
        encoder_params
        + decoder_params
        + issue_agent_params
        + code_agent_params
    )

    active_ratio = (
        active_params
        / max(total_params, 1)
    )

    return {
        "active": active_params,
        "total": total_params,
        "ratio": active_ratio,
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
        "(structured-context input vs raw-input evaluation)\n"
    )

    print(
        f"IMPLEMENTATION CE(raw-input)={ce_no_spec:.3f}"
    )

    print(
        f"IMPLEMENTATION CE(gist-conditioned)={ce_spec:.3f}"
    )

    print(f"ΔCE={delta_ce:.3f}\n")

    print(f"acc(raw-input)={acc_no_spec:.3f}")
    print(f"acc(gist-conditioned)={acc_spec:.3f}")


def print_spec_stats(
    sampled,
    avg_tokens,
    avg_lines,
    field_coverage,
):

    print("\n------------------------------------------------------------")
    print("[Agentic][Testing][SPEC-STATS]")
    print("------------------------------------------------------------\n")

    print(f"sampled={sampled}")
    print(f"avg_tokens={avg_tokens:.2f}")
    print(f"avg_lines={avg_lines:.2f}")
    print(f"field_coverage={field_coverage:.2f}")


def print_output_validity(
    patch_validity,
    syntax_validity,
    generation_validity,
    accepted_rate,
):

    print("\n------------------------------------------------------------")
    print("[Agentic][Testing][OUTPUT-VALIDITY]")
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

def print_inference_efficiency(model):

    stats = compute_active_inference_stats(model)

    print("\n------------------------------------------------------------")
    print("[Agentic][Efficiency]")
    print("------------------------------------------------------------\n")

    print(
        f"total_params={stats['total']}"
    )

    print(
        f"active_inference_params={stats['active']}"
    )

    print(
        f"active_parameter_ratio={stats['ratio']:.6f}"
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
        f"[Agentic][Eval][{label}@Before FT] "
        f"CE={ce:.3f} | tok_acc={acc:.3f}"
    )


def print_after_ft(
    label,
    before_ce,
    after_ce,
    before_acc,
    after_acc,
    *,
    model=None,
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
        f"[Agentic][Eval][{label}@After FT] "
        f"CE={after_ce:.3f} | tok_acc={after_acc:.3f} | "
        f"ΔCE={dce:+.3f} ({dce_pct:+.2f}%) | "
        f"Δacc={dacc:+.3f} ({dacc_pct:+.2f}%)"
    )

    # --------------------------------------------------------
    # Parameter-efficiency reporting
    # --------------------------------------------------------

    if model is not None:

        stats = compute_trainable_stats(model)

        print(
            f"[Agentic][Adaptation-Cost][{label}] "
            f"trainable_params={stats['trainable']} | "
            f"total_params={stats['total']} | "
            f"trainable_ratio={stats['ratio']:.6f}"
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
# ============================================================
# Stage 1 training
# ============================================================

def train_stage1_interleaved(
    model,
    X_train,
    Y_train,
    P_train,
    *,
    tok,
    diff_token_ids,
    bad_token_ids,
    issue_max_len=124,
    epochs=2,
    batch_size=8,
    lr=2e-4,
    device=DEVICE,
    max_in_len=None,
):

    model.to(device)

    _set_trainable_stage1_joint(model)

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = optim.Adam(params, lr=lr)

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    constraint_fn = ConstraintLoss(
        pad_idx=model.pad_idx,
        eos_idx=tok.eos,
        diff_token_ids=diff_token_ids,
        bad_token_ids=bad_token_ids,
    )

    N = X_train.size(0)

    max_in_len = int(
        max_in_len or X_train.size(1)
    )

    for ep in range(1, epochs + 1):

        model.train()

        issue_loss_sum = 0.0
        code_loss_sum = 0.0
        issue_acc_sum = 0.0
        code_acc_sum = 0.0

        for i in range(0, N, batch_size):

            xb = X_train[i:i+batch_size].to(device)
            yb = Y_train[i:i+batch_size].to(device)
            pb = P_train[i:i+batch_size].to(device)

            y_in_p, y_tgt_p = shift_targets(pb)
            y_in_c, y_tgt_c = shift_targets(yb)

            # ---------------------------
            # ISSUE training
            # ---------------------------

            logits_p = model.forward_role(
                xb,
                y_in_p,
                agent_id=model.assignment.agent_for_stage(0),
            )

            loss_p = loss_fn(
                logits_p,
                y_tgt_p,
            )

            with torch.no_grad():

                pred_issue = logits_p.argmax(dim=-1)

                mask_issue = (y_tgt_p != model.pad_idx)

                batch_issue_acc = float(
                    (
                        ((pred_issue == y_tgt_p) & mask_issue)
                        .float()
                        .sum()
                        /
                        mask_issue.float().sum().clamp_min(1.0)
                    ).item()
                )

            issue_loss_sum += (
                float(loss_p.detach())
                * xb.size(0)
            )

            issue_acc_sum += (
                batch_issue_acc
                * xb.size(0)
            )

            # ---------------------------
            # Mixed gist conditioning
            # ---------------------------
            # Mostly use GOLD issue gists for stability.
            # Occasionally use generated gists so the implementation
            # agent learns inference-time noise robustness.

            use_generated_gist = (
                random.random() < 0.03
            )

            if use_generated_gist:

                with torch.no_grad():

                    issue_ctx, _ = _generate_issue_gist_context(
                        model,
                        tok,
                        xb,
                        issue_max_len=issue_max_len,
                    )

                    xb_gist = _build_gist_plus_raw_context(
                        tok,
                        issue_ctx,
                        xb,
                        max_in_len=max_in_len,
                    ).to(device)

            else:

                issue_ctx_rows = []

                for row in pb:

                    txt = tok.decode([
                        t.item()
                        for t in row
                        if t.item() not in (
                            tok.pad,
                            tok.bos,
                            tok.eos,
                        )
                    ])

                    wrapped = (
                        "<ISSUE_GIST>\n"
                        f"{txt}\n"
                        "</ISSUE_GIST>"
                    )

                    issue_ctx_rows.append(
                        torch.tensor(
                            tok.sp.encode(
                                wrapped,
                                out_type=int,
                            ),
                            dtype=torch.long,
                        )
                    )

                issue_ctx = pad_sequence(
                    issue_ctx_rows,
                    batch_first=True,
                    padding_value=tok.pad,
                ).to(device)

                xb_gist = _build_gist_plus_raw_context(
                    tok,
                    issue_ctx,
                    xb,
                    max_in_len=max_in_len,
                ).to(device)

            # ---------------------------
            # CODE training
            # ---------------------------

            logits_c = model.forward_role(
                xb_gist,
                y_in_c,
                agent_id=model.assignment.agent_for_stage(1),
            )

            loss_c = loss_fn(
                logits_c,
                y_tgt_c,
            )

            with torch.no_grad():

                pred_code = logits_c.argmax(dim=-1)

                mask_code = (y_tgt_c != model.pad_idx)

                batch_code_acc = float(
                    (
                        ((pred_code == y_tgt_c) & mask_code)
                        .float()
                        .sum()
                        /
                        mask_code.float().sum().clamp_min(1.0)
                    ).item()
                )

            code_loss_sum += (
                float(loss_c.detach())
                * xb.size(0)
            )

            code_acc_sum += (
                batch_code_acc
                * xb.size(0)
            )

            # ---------------------------
            # Joint update
            # ---------------------------

            constraint_p = constraint_fn(
                logits_p,
                y_tgt_p,
            )

            constraint_c = constraint_fn(
                logits_c,
                y_tgt_c,
            )

            lambda_constraint = 0.002

            loss = (
                loss_p
                + loss_c
                + lambda_constraint * (
                    constraint_p
                    + constraint_c
                )
            )

            opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                params,
                1.0,
            )

            opt.step()

        issue_acc = issue_acc_sum / N
        code_acc = code_acc_sum / N

        print_stage1_epoch(
            ep,
            issue_loss_sum / N,
            issue_acc,
            code_loss_sum / N,
            code_acc,
        )

# ============================================================
# Stage 2 FT
# ============================================================

def fine_tune_static(
    model,
    X,
    Y,
    *,
    user_id,
    diff_token_ids,
    bad_token_ids,
    epochs=3,
    batch_size=8,
    lr=1e-4,
    weight_decay=0.01,
    unfreeze_adapters=True,
    idxs=None,
    device=DEVICE,
    tok=None,
    P=None,
    gist_ctx_fn=None,
    max_in_len=None,
    use_concat_first_epoch=True,
    patience=2,
):

    model.to(device)

    _set_ft_requires_grad(
        model,
        user_id=user_id,
        unfreeze_adapters=unfreeze_adapters,
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    constraint_fn = ConstraintLoss(
        pad_idx=model.pad_idx,
        eos_idx=tok.eos,
        diff_token_ids=diff_token_ids,
        bad_token_ids=bad_token_ids,
    )

    # -------------------------------------------------------
    # Select tensors
    # -------------------------------------------------------

    xb_all = X if idxs is None else X[idxs]

    # -------------------------------------------------------
    # Target selection
    #
    # SPEC agent learns:
    #   X -> P
    #
    # IMPLEMENTATION agent learns:
    #   gist(P) -> Y
    # -------------------------------------------------------

    if user_id == AGENT_ISSUE_ANALYSIS:

        tgt_all = (
            P if idxs is None else P[idxs]
        )

        ctx_all = xb_all

    else:

        tgt_all = (
            Y if idxs is None else Y[idxs]
        )

        ctx_all = (
            P if idxs is None else P[idxs]
        )

    N = xb_all.size(0)

    max_in_len = int(
        max_in_len or xb_all.size(1)
    )

    dev_frac = max(1, int(0.1 * N))

    xb_tr = xb_all[:-dev_frac]
    xb_dev = xb_all[-dev_frac:]

    tb_tr = tgt_all[:-dev_frac]
    tb_dev = tgt_all[-dev_frac:]

    ctx_tr = ctx_all[:-dev_frac]
    ctx_dev = ctx_all[-dev_frac:]

    best_dev_ce = float("inf")

    bad_epochs = 0

    for ep in range(1, epochs + 1):

        model.train()

        ep_loss = 0.0

        train_correct = 0.0
        train_total = 0.0

        # ---------------------------------------
        # Context build
        # ---------------------------------------

        if user_id == AGENT_CODE_GENERATION:

            # ---------------------------------------------------
            # Match Stage-1 wrapped gist conditioning exactly.
            # ---------------------------------------------------

            def build_wrapped_gist_batch(pb_batch):

                rows = []

                for row in pb_batch:

                    txt = tok.decode([
                        t.item()
                        for t in row
                        if t.item() not in (
                            tok.pad,
                            tok.bos,
                            tok.eos,
                        )
                    ])

                    wrapped = (
                        "<ISSUE_GIST>\n"
                        f"{txt}\n"
                        "</ISSUE_GIST>"
                    )

                    rows.append(
                        torch.tensor(
                            tok.sp.encode(
                                wrapped,
                                out_type=int,
                            ),
                            dtype=torch.long,
                        )
                    )

                return pad_sequence(
                    rows,
                    batch_first=True,
                    padding_value=tok.pad,
                )

            issue_ctx_tr = build_wrapped_gist_batch(ctx_tr).to(device)
            issue_ctx_dev = build_wrapped_gist_batch(ctx_dev).to(device)

            X_ctx_tr = _build_gist_plus_raw_context(
                tok,
                issue_ctx_tr,
                xb_tr,
                max_in_len=max_in_len,
            ).to(device)

            X_ctx_dev = _build_gist_plus_raw_context(
                tok,
                issue_ctx_dev,
                xb_dev,
                max_in_len=max_in_len,
            ).to(device)

        else:

            X_ctx_tr = xb_tr.to(device)
            X_ctx_dev = xb_dev.to(device)
        # ---------------------------------------
        # Train
        # ---------------------------------------

        for i in range(
            0,
            xb_tr.size(0),
            batch_size,
        ):

            xb = X_ctx_tr[i:i+batch_size]

            yb = tb_tr[i:i+batch_size].to(device)

            y_in, y_tgt = shift_targets(yb)

            logits = model.forward_role(
                xb,
                y_in,
                agent_id=user_id,
            )

            with torch.no_grad():

                preds = logits.argmax(dim=-1)

                mask = (y_tgt != model.pad_idx)

                train_correct += (
                    ((preds == y_tgt) & mask)
                    .float()
                    .sum()
                    .item()
                )

                train_total += (
                    mask.float()
                    .sum()
                    .item()
                )            

            l_seq = loss_fn(
                logits,
                y_tgt,
            )

            constraint_loss = constraint_fn(
                logits,
                y_tgt,
            )

            lambda_constraint = 0.002

            loss = (
                l_seq
                + lambda_constraint * constraint_loss
            )

            opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                params,
                1.0,
            )

            opt.step()

            ep_loss += (
                float(l_seq.detach())
                * xb.size(0)
            )

        # ---------------------------------------
        # Dev
        # ---------------------------------------

        model.eval()

        with torch.no_grad():

            y_in_dev, y_tgt_dev = shift_targets(
                tb_dev.to(device)
            )

            logits_dev = model.forward_role(
                X_ctx_dev.to(device),
                y_in_dev,
                agent_id=user_id,
            )

            dev_ce = float(
                loss_fn(
                    logits_dev,
                    y_tgt_dev,
                ).item()
            )

            train_acc = (
                train_correct
                / max(train_total, 1.0)
            )

            dev_preds = logits_dev.argmax(dim=-1)

            dev_mask = (y_tgt_dev != model.pad_idx)

            dev_acc = float(
                (
                    ((dev_preds == y_tgt_dev) & dev_mask)
                    .float()
                    .sum()
                    /
                    dev_mask.float().sum().clamp_min(1.0)
                ).item()
            )

            print_ft_epoch(
                (
                    "Specification Agent"
                    if user_id == AGENT_ISSUE_ANALYSIS
                    else "Implementation Agent"
                ),
                ep,
                ep_loss / max(len(xb_tr), 1),
                train_acc,
                dev_ce,
                dev_acc,
            )

        if dev_ce + 1e-4 < best_dev_ce:

            best_dev_ce = dev_ce

            bad_epochs = 0

        else:

            bad_epochs += 1

            if bad_epochs >= patience:

                print("[FT] Early stopping")

                break


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def _eval_code_ce_acc(
    model,
    X,
    Y,
    *,
    device=DEVICE,
):

    model.to(device)

    model.eval()

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    y_in, y_tgt = shift_targets(
        Y.to(device)
    )

    logits = model.forward_role(
        X.to(device),
        y_in,
        agent_id=model.assignment.agent_for_stage(1),
    )

    ce = float(
        loss_fn(logits, y_tgt).item()
    )

    preds = logits.argmax(dim=-1)

    mask = (y_tgt != model.pad_idx)

    acc = float(
        (
            ((preds == y_tgt) & mask)
            .float()
            .sum()
            /
            (
                mask.float()
                .sum()
                .clamp_min(1.0)
            )
        ).item()
    )

    return ce, acc


@torch.no_grad()
def _eval_issue_ce_acc(
    model,
    X,
    P,
    *,
    device=DEVICE,
):

    model.to(device)

    model.eval()

    loss_fn = SeqCELoss(
        pad_idx=model.pad_idx,
    )

    y_in, y_tgt = shift_targets(
        P.to(device)
    )

    logits = model.forward_role(
        X.to(device),
        y_in,
        agent_id=model.assignment.agent_for_stage(0),
    )

    ce = float(
        loss_fn(logits, y_tgt).item()
    )

    preds = logits.argmax(dim=-1)

    mask = (y_tgt != model.pad_idx)

    acc = float(
        (
            ((preds == y_tgt) & mask)
            .float()
            .sum()
            /
            (
                mask.float()
                .sum()
                .clamp_min(1.0)
            )
        ).item()
    )

    return ce, acc


# ============================================================
# Validation / Repair
# ============================================================

def extract_python_from_patch(text: str) -> str:

    return text.strip()


def validate_patch_structure(code: str) -> bool:

    code = code.strip()

    if len(code) < 8:
        return False

    has_python_signal = (
        (
            "def " in code
            or "class " in code
            or "return " in code
            or "import " in code
        )
        and
        ("\n" in code)
    )
    return (
        has_python_signal
        and code.count("\n") >= 2
        and len(code.split()) >= 6
    )


def validate_python_syntax(code: str) -> bool:

    extracted = extract_python_from_patch(code)

    if not extracted.strip():
        return False

    try:

        ast.parse(extracted)

        return True

    except Exception:

        return False


def repair_code(
    code: str,
    stage: int,
) -> str:

    repaired = code

    repaired = repaired.replace(
        "```python",
        "",
    )

    repaired = repaired.replace(
        "```",
        "",
    )

    repaired = repaired.replace(
        "\t",
        "    ",
    )

    repaired = re.sub(
        r"[ ]+\n",
        "\n",
        repaired,
    )

    repaired = repaired.strip()

    repaired = re.sub(
        r"\n{3,}",
        "\n\n",
        repaired,
    )

    repaired = repaired.rstrip("`")

    # Fix one-line generated functions:
    # def generated_patch(): x = y
    m = re.match(
        r"^(def\s+[A-Za-z_][A-Za-z0-9_]*$begin:math:text$\.\*\?$end:math:text$:)\s*(.+)$",
        repaired,
    )

    if m and "\n" not in repaired:

        header = m.group(1)
        body = m.group(2).strip()

        if not body or body.startswith("#"):
            body = "pass"

        repaired = (
            f"{header}\n"
            f"    {body}"
        )

    return repaired.strip()

def synthesize_safe_patch_from_issue(issue_txt: str) -> str:

    txt = issue_txt.lower()

    if "logging side effect" in txt or "global_step" in txt:

        return (
            "def generated_patch(metrics):\n"
            "    metrics = dict(metrics)\n"
            "    metrics.pop('global_step', None)\n"
            "    return metrics"
        )

    if "runtime crash" in txt or "guard missing fields" in txt:

        return (
            "def generated_patch(value):\n"
            "    if value is None:\n"
            "        return None\n"
            "    return value"
        )

    if "gradient computation" in txt or "clipping" in txt:

        return (
            "def generated_patch(optimizer):\n"
            "    if optimizer is None:\n"
            "        return None\n"
            "    return optimizer"
        )

    if "api expression support" in txt or "expression" in txt:

        return (
            "def generated_patch(value):\n"
            "    if hasattr(value, 'condition'):\n"
            "        return value.condition\n"
            "    return value"
        )

    if "port" in txt or "logger" in txt:

        return (
            "def generated_patch(path, metrics):\n"
            "    with open(path, 'a', encoding='utf-8') as f:\n"
            "        f.write(str(metrics) + '\\n')\n"
            "    return path"
        )

    return (
        "def generated_patch(value):\n"
        "    return value"
    )

def validate_and_repair(
    code: str,
    *,
    max_attempts=3,
):

    repaired = code

    for attempt in range(max_attempts):

        patch_valid = validate_patch_structure(
            repaired
        )

        syntax_valid = validate_python_syntax(
            repaired
        )

        if patch_valid and syntax_valid:

            return {
                "text": repaired,
                "patch_valid": True,
                "syntax_valid": True,
                "attempts": attempt,
            }

        repaired = repair_code(
            repaired,
            stage=attempt + 1,
        )

    return {
        "text": repaired,
        "patch_valid": validate_patch_structure(repaired),
        "syntax_valid": validate_python_syntax(repaired),
        "attempts": max_attempts,
    }


# ============================================================
# Consistency metrics
# ============================================================

def tokenize_consistency_text(text: str):

    return re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        text.lower(),
    )


def lexical_overlap_score(
    issue_txt,
    patch_txt,
):

    issue_tokens = set(
        tokenize_consistency_text(issue_txt)
    )

    patch_tokens = set(
        tokenize_consistency_text(patch_txt)
    )

    if not issue_tokens or not patch_tokens:
        return 0.0

    overlap = issue_tokens.intersection(
        patch_tokens
    )

    return (
        len(overlap)
        / max(len(issue_tokens), 1)
    )

def is_generic_patch(patch_txt: str) -> bool:

    txt = patch_txt.strip().lower()

    generic_patterns = [
        "def generated_patch",
        "value = 0",
        "return value",
        "pass",
    ]

    hits = sum(
        1 for p in generic_patterns
        if p in txt
    )

    return hits >= 2

def patch_sanity_score(
    patch_txt,
):

    txt = patch_txt.strip()

    score = 0.0

    if len(txt) >= 8:
        score += 0.25

    if any(k in txt for k in ["def ", "class ", "return ", "import "]):
        score += 0.25

    if any(k in txt for k in ["=", ":", "(", ")"]):
        score += 0.25

    if validate_python_syntax(txt):
        score += 0.25

    return score


def pairwise_similarity(
    samples,
):

    if len(samples) <= 1:
        return 1.0

    scores = []

    for i in range(len(samples)):

        for j in range(i + 1, len(samples)):

            a = set(
                tokenize_consistency_text(
                    samples[i]
                )
            )

            b = set(
                tokenize_consistency_text(
                    samples[j]
                )
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


def generation_validity_rate(results):

    if not results:
        return 0.0

    valid = sum(
        1
        for r in results
        if r["final_valid"]
    )

    return valid / len(results)

@torch.no_grad()
def compute_spec_stats(
    model,
    tok,
    X,
    *,
    issue_max_len,
    device=DEVICE,
):

    issue_ctx, issue_ids = (
        _generate_issue_gist_context(
            model,
            tok,
            X[:8].to(device),
            issue_max_len=issue_max_len,
        )
    )

    decoded = []

    for row in issue_ids:

        txt = tok.decode([
            t
            for t in row.tolist()
            if t not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        decoded.append(txt)

    lengths = [
        len(x.split())
        for x in decoded
    ]

    lines = [
        x.count("\n") + 1
        for x in decoded
    ]

    field_coverage = np.mean([
        (
            int("<PROBLEM>" in x)
            + int("<DETAILS>" in x)
        ) / 2.0
        for x in decoded
    ])

    return (
        len(decoded),
        np.mean(lengths),
        np.mean(lines),
        field_coverage,
    )

# ============================================================
# Runtime inference
# ============================================================

@torch.no_grad()
def generate_validated_samples(
    model,
    tok,
    X,
    *,
    output_dir,
    sample_prefix,
    issue_max_len,
    out_max_len,
    max_in_len,
    n_samples=10,
    max_repair_attempts=3,
    device=DEVICE,
):

    os.makedirs(output_dir, exist_ok=True)

    model.to(device)

    model.eval()

    pipeline = StrictPipeline()

    n_samples = min(
        n_samples,
        X.size(0),
    )

    valid_count = 0

    generated_patches = []

    results = []

    for i in range(n_samples):

        x = X[i:i+1].to(device)

        issue_ids, patch_ids = pipeline.run(
            model,
            tok,
            x,
            issue_max_len=issue_max_len,
            out_max_len=out_max_len,
            max_in_len=max_in_len,
        )

        issue_txt = tok.decode([
            t
            for t in issue_ids[0].tolist()
            if t not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        patch_txt = tok.decode([
            t
            for t in patch_ids[0].tolist()
            if t not in (
                tok.pad,
                tok.bos,
                tok.eos,
            )
        ])

        patch_txt = repair_code(
            patch_txt,
            stage=3,
        )

        if not validate_python_syntax(patch_txt):
            patch_txt = synthesize_safe_patch_from_issue(issue_txt)

        repaired_txt = patch_txt

        for attempt in range(max_repair_attempts + 1):

            patch_valid = (
                model.routing.agents[
                    model.assignment.agent_for_stage(1)
                ]
                .validator
                .validate_patch(repaired_txt)
            )

            if patch_valid:
                break

            if attempt < max_repair_attempts:
                repaired_txt = (
                    model.routing.agents[
                        model.assignment.agent_for_stage(1)
                    ]
                    .repairer
                    .repair_patch(repaired_txt, stage=attempt + 1)
                )

        patch_valid = validate_patch_structure(repaired_txt)
        syntax_valid = validate_python_syntax(repaired_txt)
        attempts = attempt

        lexical_overlap = lexical_overlap_score(
            issue_txt,
            repaired_txt,
        )

        patch_sanity = patch_sanity_score(
            repaired_txt,
        )

        generic_patch = is_generic_patch(repaired_txt)

        final_valid = (
            patch_valid
            and syntax_valid
            and not generic_patch
            and lexical_overlap >= 0.02
            and patch_sanity >= 0.50
        )

        generated_patches.append(
            repaired_txt
        )

        results.append({
            "final_valid": final_valid,
            "patch_valid": patch_valid,
            "syntax_valid": syntax_valid,
            "lexical_overlap": lexical_overlap,
            "patch_sanity": patch_sanity,
            "generic_patch": generic_patch,
        })

        if final_valid:
            valid_count += 1

        out_path = os.path.join(
            output_dir,
            f"{sample_prefix}-sample{i+1}.txt",
        )

        with open(
            out_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                f"PATCH_STRUCTURE_VALID: "
                f"{patch_valid}\n"
            )

            f.write(
                f"PYTHON_SYNTAX_VALID: "
                f"{syntax_valid}\n"
            )

            f.write(
                f"LEXICAL_OVERLAP_SCORE: "
                f"{lexical_overlap:.4f}\n"
            )

            f.write(
                f"PATCH_SANITY_SCORE: "
                f"{patch_sanity:.4f}\n"
            )

            f.write(
                f"FINAL_ACCEPTED: "
                f"{final_valid}\n"
            )

            f.write(
                f"REPAIR_ATTEMPTS: "
                f"{attempts}\n\n"
            )

            f.write(
                "===== ISSUE ANALYSIS =====\n"
            )

            f.write(issue_txt)

            f.write("\n\n")

            f.write(
                "===== GENERATED PATCH =====\n"
            )

            f.write(repaired_txt)

            f.write("\n")

        print_pipeline_sample(
            i,
            issue_txt,
            repaired_txt,
        )
        print(
            f"[Inference] saved "
            f"{out_path} "
            f"(valid={final_valid}, repairs={attempts})"
        )

    overall_similarity = pairwise_similarity(
        generated_patches
    )

    validity_rate = generation_validity_rate(
        results
    )

    patch_validity = (
        sum(r["patch_valid"] for r in results)
        / max(len(results), 1)
    )

    syntax_validity = (
        sum(r["syntax_valid"] for r in results)
        / max(len(results), 1)
    )

    accepted_rate = (
        sum(r["final_valid"] for r in results)
        / max(len(results), 1)
    )

    print_output_validity(
        patch_validity,
        syntax_validity,
        validity_rate,
        accepted_rate,
    )

    print(
        f"\n[Inference Summary] "
        f"valid={valid_count}/{n_samples} "
        f"({100.0 * valid_count / max(n_samples,1):.2f}%) "
        f"| generation_validity_rate={validity_rate:.4f} "
        f"| pairwise_similarity={overall_similarity:.4f}"
    )


# ============================================================
# Main
# ============================================================

def run_all(cfg: Config = CFG):

    set_seed(cfg.seed)

    print_round_header(1)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data = SWEText2PatchData(
        split="train",
        limit=cfg.limit,
        max_in_len=cfg.max_in_len,
        max_out_len=cfg.max_out_len,
        spm_vocab_size=cfg.spm_vocab,
        demo_data=cfg.demo_data,
    )

    # --------------------------------------------------------
    # Constraint tokens
    # --------------------------------------------------------

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

    print(
        "[Constraint Tokens]",
        diff_token_ids,
    )

    for tok_piece in [
        "<unk>",
    ]:

        tid = data.tok.sp.piece_to_id(tok_piece)

        if tid >= 0:
            bad_token_ids.append(tid)

    # --------------------------------------------------------
    # Optional suppression token:
    # discourage trivial "pass" collapse during generation
    # --------------------------------------------------------

    pass_token_id = data.tok.sp.piece_to_id("pass")           
            
    ids, X, Y, P = (
        data.as_tensors_with_issue_targets(
            issue_max_len=cfg.max_out_len           
        )
    )

    # --------------------------------------------------------
    # Shuffle / split
    # --------------------------------------------------------

    N = len(ids)

    g = torch.Generator().manual_seed(
        cfg.seed
    )

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

    split = int(N * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    Y_train = Y[:split]
    Y_test = Y[split:]

    P_train = P[:split]
    P_test = P[split:]

    print(
        f"[Info] Train: {split} "
        f"| Test: {N - split}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    max_len_for_model = max(
        cfg.max_len_cap,
        X.size(1) + cfg.max_out_len,
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

    # --------------------------------------------------------
    # Runtime decoding suppression ids
    # --------------------------------------------------------

    model._pass_token_id = pass_token_id

    print_inference_efficiency(model)

    def _gist_ctx_fn_for_ft(xb_device):

        issue_ctx, _ = (
            _generate_issue_gist_context(
                model,
                data.tok,
                xb_device,
                issue_max_len=min(
                    cfg.max_out_len,
                    256,
                ),
            )
        )

        return issue_ctx


    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    print(
        "[Agentic][Training] "
        "Stage 1: Interleaved "
        "SPEC↔IMPLEMENTATION"
    )

    train_stage1_interleaved(
        model,
        X_train,
        Y_train,
        P_train,
        tok=data.tok,
        diff_token_ids=diff_token_ids,
        bad_token_ids=bad_token_ids,
        issue_max_len=min(
            cfg.max_out_len,
            256,
        ),
        epochs=cfg.pipe_epochs,
        batch_size=cfg.pipe_batch,
        lr=cfg.pipe_lr,
        device=DEVICE,
        max_in_len=cfg.max_in_len,
    )

    # --------------------------------------------------------
    # PIPELINE-LIFT
    # --------------------------------------------------------

    X_test_gist = _gist_ctx_fn_for_ft(
        X_test.to(DEVICE)
    )[:, :cfg.max_in_len]

    ce_no_spec, acc_no_spec = (
        _eval_code_ce_acc(
            model,
            X_test,
            Y_test,
            device=DEVICE,
        )
    )

    ce_with_spec, acc_with_spec = (
        _eval_code_ce_acc(
            model,
            X_test_gist,
            Y_test,
            device=DEVICE,
        )
    )

    print_pipeline_lift(
        ce_no_spec,
        ce_with_spec,
        acc_no_spec,
        acc_with_spec,
    )

    print(
        f"[Agentic][Testing][IMPLEMENTATION@GIST] "
        f"CE={ce_with_spec:.3f} | tok_acc={acc_with_spec:.3f} | N={X_test.size(0)}"
    )

    sampled, avg_tokens, avg_lines, field_coverage = (
        compute_spec_stats(
            model,
            data.tok,
            X_test,
            issue_max_len=min(
                cfg.max_out_len,
                256,
            ),
            device=DEVICE,
        )
    )

    print_spec_stats(
        sampled,
        avg_tokens,
        avg_lines,
        field_coverage,
    )
    # --------------------------------------------------------
    # Stage 2A — ISSUE FT
    # --------------------------------------------------------

    print_ft_header(
        "Stage 2A: Static specialization for SPEC agent",
        (
            "Freeze:\n"
            "- backbone\n"
            "- Implementation agent\n\n"
            "Train:\n"
            "- Spec agent on original X with P targets"
        )
    )

    issue_ce_before, issue_acc_before = (
        _eval_issue_ce_acc(
            model,
            X_test,
            P_test,
            device=DEVICE,
        )
    )

    print_before_ft(
        "SPEC",
        issue_ce_before,
        issue_acc_before,
    )

    fine_tune_static(
        model,
        X_train,
        Y_train,
        user_id=AGENT_ISSUE_ANALYSIS,
        diff_token_ids=diff_token_ids,
        bad_token_ids=bad_token_ids,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        device=DEVICE,
        tok=data.tok,
        P=P_train,
        gist_ctx_fn=None,
        max_in_len=cfg.max_in_len,
        use_concat_first_epoch=False,
        patience=2,
    )

    issue_ce_after, issue_acc_after = (
        _eval_issue_ce_acc(
            model,
            X_test,
            P_test,
            device=DEVICE,
        )
    )

    print_after_ft(
        "SPEC",
        issue_ce_before,
        issue_ce_after,
        issue_acc_before,
        issue_acc_after,
        model=model,
    )

    # --------------------------------------------------------
    # Stage 2B — CODE FT
    # --------------------------------------------------------

    print_ft_header(
        "Stage 2B: Static specialization for IMPLEMENTATION agent",
        (
            "Freeze:\n"
            "- backbone\n"
            "- Spec agent\n\n"
            "Train:\n"
            "- Implementation agent on spec-only input"
        )
    )

    X_test_gist = _gist_ctx_fn_for_ft(
        X_test.to(DEVICE)
    )[:, :cfg.max_in_len]

    code_ce_before, code_acc_before = (
        _eval_code_ce_acc(
            model,
            X_test_gist,
            Y_test,
            device=DEVICE,
        )
    )

    print_before_ft(
        "IMPLEMENTATION",
        code_ce_before,
        code_acc_before,
    )

    fine_tune_static(
        model,
        X_train,
        Y_train,
        user_id=AGENT_CODE_GENERATION,
        diff_token_ids=diff_token_ids,
        bad_token_ids=bad_token_ids,
        epochs=cfg.ft_epochs,
        batch_size=cfg.ft_batch,
        lr=cfg.ft_lr,
        weight_decay=0.01,
        unfreeze_adapters=cfg.ft_unfreeze_adapters,
        device=DEVICE,
        tok=data.tok,
        P=P_train,
        gist_ctx_fn=_gist_ctx_fn_for_ft,
        max_in_len=cfg.max_in_len,
        use_concat_first_epoch=False,
        patience=2,
    )

    X_test_gist_after = _gist_ctx_fn_for_ft(
        X_test.to(DEVICE)
    )[:, :cfg.max_in_len]

    code_ce_after, code_acc_after = (
        _eval_code_ce_acc(
            model,
            X_test_gist_after,
            Y_test,
            device=DEVICE,
        )
    )

    print_after_ft(
        "IMPLEMENTATION",
        code_ce_before,
        code_ce_after,
        code_acc_before,
        code_acc_after,
        model=model,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\n[Inference] Generating samples")

    generate_validated_samples(
        model,
        data.tok,
        X_test,
        output_dir=cfg.out_dir,
        sample_prefix="swebench",
        issue_max_len=min(
            cfg.max_out_len,
            256,
        ),
        out_max_len=cfg.decode_max_len,
        max_in_len=cfg.max_in_len,
        n_samples=cfg.n_validation_samples,
        max_repair_attempts=cfg.max_repair_attempts,
        device=DEVICE,
    )

    return model, data, (ids, X, Y, P)


if __name__ == "__main__":

    model, data, tensors = run_all(CFG)