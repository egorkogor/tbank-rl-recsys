"""GPT-2 policy over track tokens and the batch builder shared by SFT, RM, RL and evaluation.

Input of a position = embedding(track) + embedding(outcome) + embedding(organic flag) + embedding(gap
bucket), fed to GPT2Model through inputs_embeds. The output head is tied to the track embedding.
Sequences are LEFT-padded, so the last real event is always the last column and generated tracks
are appended on the right (positions come from the attention mask).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2Model

from .config import GEN_GAP, N_SPECIAL, PAD, UNKNOWN

FIELDS = ("tok", "outcome", "org", "gap")


RECENT = 32  # the second copy feature counts the last RECENT events


class Policy(nn.Module):
    """copy=True adds a repeat channel (as in RepeatNet): logit(v) += w0 * log1p(count of v in the context)
    + w1 * log1p(count of v in the last RECENT events), counted causally up to the predicting position."""

    def __init__(self, n_tok: int, d: int = 256, n_layer: int = 4, n_head: int = 4, n_pos: int = 320,
                 dropout: float = 0.1, copy: bool = False):
        super().__init__()
        self.cfg = dict(n_tok=n_tok, d=d, n_layer=n_layer, n_head=n_head, n_pos=n_pos, dropout=dropout, copy=copy)
        self.copy = copy
        if copy:
            self.copy_w = nn.Parameter(torch.zeros(2))
        self.tok = nn.Embedding(n_tok, d, padding_idx=PAD)
        self.outcome = nn.Embedding(4, d)
        self.org = nn.Embedding(2, d)
        self.gap = nn.Embedding(8, d)
        self.gpt = GPT2Model(GPT2Config(vocab_size=1, n_positions=n_pos, n_embd=d, n_layer=n_layer, n_head=n_head,
                                        resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout))
        self.bias = nn.Parameter(torch.zeros(n_tok))
        nn.init.normal_(self.tok.weight, std=0.02)
        # PAD and UNK are never recommended.
        self.register_buffer("forbid", torch.zeros(n_tok, dtype=torch.bool), persistent=False)
        self.forbid[:N_SPECIAL] = True

    def hidden(self, b: dict) -> torch.Tensor:
        x = self.tok(b["tok"]) + self.outcome(b["outcome"]) + self.org(b["org"]) + self.gap(b["gap"])
        mask = b["mask"]
        pos = (mask.long().cumsum(1) - 1).clamp(min=0)
        return self.gpt(inputs_embeds=x, attention_mask=mask, position_ids=pos).last_hidden_state

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        z = (h @ self.tok.weight.T + self.bias).float()
        return z.masked_fill(self.forbid, float("-inf"))

    def copy_bonus(self, tok: torch.Tensor, valid: torch.Tensor, recent: torch.Tensor) -> torch.Tensor:
        """tok / valid / recent [N, L] -> bonus [N, V] from causal context counts."""
        C = torch.zeros(tok.shape[0], self.cfg["n_tok"], device=tok.device)
        C.scatter_add_(1, tok, valid.float())
        R = torch.zeros_like(C).scatter_add_(1, tok, recent.float())
        return self.copy_w[0] * torch.log1p(C) + self.copy_w[1] * torch.log1p(R)

    def context_masks(self, mask: torch.Tensor, j: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For predicting positions j [N] over rows of `mask` [N, L]: events <= j, and the last RECENT of them."""
        idx = torch.arange(mask.shape[1], device=mask.device)[None]
        valid = mask & (idx <= j[:, None])
        return valid, valid & (idx > j[:, None] - RECENT)

    def logits_pos(self, b: dict, h: torch.Tensor, pos) -> torch.Tensor:
        """Logits predicting the event after each column in `pos` -> [B, P, V] (repeat channel included)."""
        z = self.logits(h[:, pos])
        if not self.copy:
            return z
        B = h.shape[0]
        out = []
        for k, p in enumerate(pos):
            j = torch.full((B,), int(p), device=h.device)
            valid, recent = self.context_masks(b["mask"], j)
            out.append(z[:, k] + self.copy_bonus(b["tok"], valid, recent))
        return torch.stack(out, 1)

    def last_logits(self, b: dict) -> torch.Tensor:
        h = self.hidden(b)
        return self.logits_pos(b, h, [h.shape[1] - 1])[:, 0]


def _ce_chunk(h, w, bias, y):
    # bf16 logits (log-softmax accumulates in fp32); targets are never PAD/UNK, so no mask is needed here.
    return F.cross_entropy(h @ w.T + bias, y, reduction="sum").float()


def _ce_chunk_copy(model, h, w, bias, y, tok, valid, recent):
    z = (h @ w.T + bias).float() + model.copy_bonus(tok, valid, recent)
    return F.cross_entropy(z, y, reduction="sum")


def chunked_ce(model: Policy, h: torch.Tensor, y: torch.Tensor, chunk: int = 2048, ctx: tuple | None = None
               ) -> torch.Tensor:
    """Mean next-track cross-entropy over the full vocabulary, logits recomputed in backward per chunk.

    ctx = (tok [B, L], mask [B, L], row [N], col [N]) of the kept positions: needed by the repeat channel.
    """
    total = h.new_zeros((), dtype=torch.float32)
    if model.copy:
        chunk = min(chunk, 512)
    for i in range(0, h.shape[0], chunk):
        sl = slice(i, i + chunk)
        if model.copy:
            tok, mask, row, col = ctx
            valid, recent = model.context_masks(mask[row[sl]], col[sl])
            total = total + checkpoint(_ce_chunk_copy, model, h[sl], model.tok.weight.to(h.dtype),
                                       model.bias.to(h.dtype), y[sl], tok[row[sl]], valid, recent,
                                       use_reentrant=False)
        else:
            total = total + checkpoint(_ce_chunk, h[sl], model.tok.weight.to(h.dtype),
                                       model.bias.to(h.dtype), y[sl], use_reentrant=False)
    return total / max(1, h.shape[0])


def make_batch(seq, ranges: list[tuple[int, int]], length: int, device, drop_outcome: float = 0.0,
               rng: np.random.Generator | None = None) -> dict:
    """Left-padded batch of global listen-index ranges [s, e), truncated to the last `length` events."""
    B = len(ranges)
    out = {f: np.zeros((B, length), np.int64) for f in FIELDS}
    out["gi"] = np.full((B, length), -1, np.int64)
    mask = np.zeros((B, length), bool)
    for r, (s, e) in enumerate(ranges):
        s = max(s, e - length)
        n = e - s
        if n <= 0:
            continue
        idx = np.arange(s, e)
        out["tok"][r, -n:] = seq.tok[idx]
        out["outcome"][r, -n:] = seq.outcome[idx]
        out["org"][r, -n:] = seq.org[idx]
        out["gap"][r, -n:] = seq.gap[idx]
        out["gi"][r, -n:] = idx
        mask[r, -n:] = True
    if drop_outcome > 0:
        rng = rng or np.random.default_rng()
        out["outcome"][(rng.random((B, length)) < drop_outcome) & mask] = UNKNOWN
    b = {k: torch.from_numpy(v).to(device) for k, v in out.items()}
    b["mask"] = torch.from_numpy(mask).to(device)
    return b


def append(b: dict, tok: torch.Tensor, outcome: torch.Tensor | None = None) -> dict:
    """Append one generated track per row: outcome UNKNOWN (or given), algorithmic, next-in-session gap."""
    B = tok.shape[0]
    col = lambda v: torch.full((B, 1), v, dtype=torch.long, device=tok.device)  # noqa: E731
    new = dict(b)
    new["tok"] = torch.cat([b["tok"], tok[:, None]], 1)
    new["outcome"] = torch.cat([b["outcome"], col(UNKNOWN) if outcome is None else outcome[:, None]], 1)
    new["org"] = torch.cat([b["org"], col(0)], 1)
    new["gap"] = torch.cat([b["gap"], col(GEN_GAP)], 1)
    new["gi"] = torch.cat([b["gi"], col(-1)], 1)
    new["mask"] = torch.cat([b["mask"], torch.ones((B, 1), dtype=torch.bool, device=tok.device)], 1)
    return new


def limit_gpu(device) -> None:
    """Cap this process's share of VRAM. On Windows an over-full GPU silently spills into system RAM
    (WDDM shared memory) and runs ~6x slower; with the cap the process fails loudly instead.
    Desktop apps hold ~2.6 GB of the 8 GB here. Override with RLREC_GPU_FRACTION."""
    import os
    if torch.device(device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("RLREC_GPU_FRACTION", "0.6")))


def save(model: nn.Module, path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load_policy(path, device, copy: bool | None = None) -> Policy:
    """copy=True on a checkpoint without the repeat channel adds it with zero weights."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    if copy is not None:
        cfg["copy"] = copy
    m = Policy(**cfg).to(device)
    m.load_state_dict(ck["state"], strict=not (cfg.get("copy") and "copy_w" not in ck["state"]))
    return m
