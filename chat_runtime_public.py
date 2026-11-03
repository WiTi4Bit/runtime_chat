#!/usr/bin/env python3
"""
======================================================================
Hak Cipta (c) 2026 oleh Tim Perekayasa AI POKDARTIK. Seluruh hak dilindungi undang-undang.

Perangkat lunak inti ini dilisensikan khusus untuk penggunaan NON-KOMERSIAL, 
pendidikan, riset, dan peninjauan ilmiah (peer-review). Pengguna diizinkan 
untuk mengunduh, menjalankan, dan membagikan kode ini dengan tetap mencantumkan 
kredit/atribusi kepada POKDARTIK. Dilarang keras menggunakannya untuk tujuan komersial.

======================================================================
LISENSI DEPENDENSI & PIHAK KETIGA:
======================================================================
Aplikasi ini dibangun menggunakan bahasa pemrograman Python dan memanfaatkan 
Pustaka Standar Python (argparse, json, os, struct, sys, time, pathlib,torch,transformer) 
yang dilisensikan di bawah Python Software Foundation (PSF) License Agreement. 
Penggunaan komponen standar ini sepenuhnya mematuhi ketentuan open-source PSF.

Model Gemma 3 (Gemma3-1B-IT) tunduk pada Perjanjian Lisensi Penggunaan Gemma 
(Gemma Terms of Use) dari Google.

Penggunaan:
    python3 chat_runtime_public.py <model_dir>
    python3 chat_runtime_public.py <model_dir> --prompt "Halo"
    python3 chat_runtime_public.py <model_dir> --attention softmax
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, Gemma3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

try:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
except ImportError:
    ALL_MASK_ATTENTION_FUNCTIONS = None
    eager_mask = None

# ─────────────────────────────────────────────────────────────────────────────
#  KONSTANTA & KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
QNM_INDEX_META = "qnm.index"
Q4_BLOCK = 32
Q4_BLOCK_BYTES = 18  # 2 byte skala + 16 byte data Q4
R9_SCALE = 1e7
INT64_SAFE_MAX = 9.0e18
MASK_THRESH = -1e30

# ─────────────────────────────────────────────────────────────────────────────
#  FUNGSI PEMBANTU
# ─────────────────────────────────────────────────────────────────────────────
def _repair_missing_commas(text):
    for _ in range(50):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            if "Expecting ',' delimiter" not in e.msg:
                return text
            text = text[:e.pos] + "," + text[e.pos:]
    return text


def _parse_config_text(text, cfg_path=None):
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        try:
            return json.loads(_repair_missing_commas(text)), True
        except json.JSONDecodeError as e:
            raise ValueError(f"config.json tidak valid ({e})") from None


def load_config_file(cfg_path):
    with open(cfg_path) as f:
        data, _ = _parse_config_text(f.read(), cfg_path)
    return data

def collect_stop_ids(model, tokenizer):
    ids = set()
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if gen_eos is not None:
        ids.update(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<end_of_turn>", "<start_of_turn>", "<eos>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != unk and tid >= 0:
            ids.add(tid)
    ids.discard(None)
    return ids

def config_model_type(model_dir):
    try:
        return load_config_file(os.path.join(model_dir, "config.json")).get("model_type", "LLM")
    except Exception:
        return "LLM"

# ─────────────────────────────────────────────────────────────────────────────
#  PEMBACA QNM INDEX DARI SAFETENSORS
# ─────────────────────────────────────────────────────────────────────────────
def _read_qnm_index(path: Path):
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            return None
        header_len = struct.unpack("<Q", raw_len)[0]
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            return None
        header = json.loads(raw_header.decode("utf-8"))
    meta = header.get("__metadata__", {})
    raw_idx = meta.get(QNM_INDEX_META)
    if not raw_idx:
        return None
    try:
        return json.loads(raw_idx)
    except Exception:
        return None

def _scan_safetensors(model_dir: Path):
    return sorted(model_dir.glob("*.safetensors"))

def _scan_shard_headers(model_dir: Path):
    """Baca (files, qnm_index, tensor_shapes, tensor_dtypes, n_total).

    Format riset (qnm.index ada): mode signed-grid/fp16-split dipakai untuk
    pembacaan tensor.
    """
    files = _scan_safetensors(model_dir)
    if not files:
        raise FileNotFoundError(f"Tidak ada .safetensors di {model_dir}")
    qnm_index = {}
    tensor_shapes = {}
    tensor_dtypes = {}
    n_total = 0
    for path in files:
        idx = _read_qnm_index(path)
        if idx:
            qnm_index.update(idx)
        with open(path, "rb") as f:
            raw_len = f.read(8)
            header_len = struct.unpack("<Q", raw_len)[0]
            raw_header = f.read(header_len)
            header = json.loads(raw_header.decode("utf-8"))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            tensor_shapes[name] = tuple(info["shape"])
            tensor_dtypes[name] = info.get("dtype", "BF16")
            n_total += 1
    return files, qnm_index, tensor_shapes, tensor_dtypes, n_total


# ─────────────────────────────────────────────────────────────────────────────
#  DEKUANTISASI & REKONSTRUKSI
# ─────────────────────────────────────────────────────────────────────────────
def _digital_root_9_torch(vals):
    valid = torch.isfinite(vals) & (vals != 0.0)
    scaled = torch.where(valid, torch.abs(vals) * R9_SCALE, torch.zeros_like(vals))
    scaled = torch.clamp(scaled, max=INT64_SAFE_MAX)
    n = torch.round(scaled).to(torch.int64)
    mod9 = n % 9
    return torch.where(n == 0, torch.zeros_like(n),
                       torch.where(mod9 == 0, torch.full_like(n, 9), mod9))


def _nearest_bf16_torch(f):
    u32 = f.contiguous().to(torch.float32).view(torch.int32)
    u64 = u32.to(torch.int64) & 0xFFFFFFFF
    r = ((u64 + 0x7FFF + ((u64 >> 16) & 1)) >> 16).to(torch.int32)
    return (r << 16).view(torch.float32)


def _hitung_pembulatan_digital_array_torch(vals):
    v = vals.to(torch.float32)
    d = torch.where((v >= 1.0) & (v < 4.0), torch.tensor(6.0, device=v.device),
                    torch.where((v >= 4.0) & (v < 7.0), torch.tensor(3.0, device=v.device),
                                torch.tensor(9.0, device=v.device)))
    q = torch.round((v / d) * 10000.0)
    return (q / 10000.0) * d


def dr9_nearest_torch(scores):
    roots = _digital_root_9_torch(scores)
    mask9 = (roots == 9)
    quantized = _hitung_pembulatan_digital_array_torch(scores)
    return torch.where(mask9, _nearest_bf16_torch(scores),
                       _nearest_bf16_torch(quantized))


def dequant_fp16_split(u8_tensor, out_dtype=torch.float32):
    """U8 berisi byte FP16 di-split -> float32 (atau dtype lain)."""
    n = u8_tensor.numel()
    if n % 2 != 0:
        raise ValueError(f"fp16-split: jumlah byte tidak genap ({n})")
    u16 = u8_tensor.reshape(-1, 2).view(torch.uint16)
    f16 = u16.view(torch.float16)
    f32 = f16.to(torch.float32)
    return f32 if out_dtype == torch.float32 else f32.to(out_dtype)


def _dequant_q4_block(u8, n_elems, out_dtype=torch.float32):
    """Internal: dekuantisasi blok Q4M -> float32."""
    nb = (n_elems + Q4_BLOCK - 1) // Q4_BLOCK
    expected = nb * Q4_BLOCK_BYTES
    if u8.numel() < expected:
        raise ValueError(f"Q4M: byte kurang ({u8.numel()} < {expected})")
    u8 = u8[:expected].reshape(nb, Q4_BLOCK_BYTES)
    s = u8[:, 0:2].contiguous().view(torch.float16).to(torch.float32).reshape(-1)
    qs = u8[:, 2:Q4_BLOCK_BYTES]
    q = torch.empty((nb, Q4_BLOCK), dtype=torch.float32)
    q[:, 0::2] = qs & 0x0F
    q[:, 1::2] = qs >> 4
    q.sub_(8.0)
    q.mul_(s[:, None])
    w = q.reshape(-1)[:n_elems]
    return w if out_dtype == torch.float32 else w.to(out_dtype)


def reconstruct_tensor(u8_tensor, mode, n_elems=None, orig_shape=None,
                       out_dtype=torch.float32):
    """Rekonstruksi tensor -> out_dtype, di-shape sesuai original."""
    if mode is None:
        f32 = (u8_tensor if u8_tensor.dtype == out_dtype
               else u8_tensor.to(out_dtype))
    elif mode == "fp16-split":
        f32 = dequant_fp16_split(u8_tensor, out_dtype=out_dtype)
    elif mode in ("signed-grid", "qnm", "unknown"):
        if n_elems is None:
            n_elems = int(u8_tensor.numel() / 0.5625)
        f32 = _dequant_q4_block(u8_tensor, n_elems, out_dtype=out_dtype)
    else:
        raise ValueError(f"Mode tidak dikenali: {mode}")

    # Shape sesuai original jika ada
    if orig_shape is not None:
        expected = 1
        for d in orig_shape:
            expected *= d
        if f32.numel() == expected:
            f32 = f32.reshape(orig_shape)
        else:
            # Kalau tidak cocok, coba transpose (misal weight [in,out] vs [out,in])
            if f32.numel() == expected and len(orig_shape) == 2:
                f32 = f32.reshape(orig_shape)
            else:
                raise ValueError(f"Jumlah elemen tidak cocok: {f32.numel()} vs {expected} untuk shape {orig_shape}")
    return f32


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD STATE DICT DARI SAFETENSORS U8
# ─────────────────────────────────────────────────────────────────────────────
def _infer_gemma3_shape(name, cfg):
    """Infer expected float tensor shape from Gemma3 config and tensor name."""
    hidden = cfg.get("hidden_size", 1152)
    intermediate = cfg.get("intermediate_size", 6912)
    vocab = cfg.get("vocab_size", 262144)
    n_heads = cfg.get("num_attention_heads", 4)
    n_kv_heads = cfg.get("num_key_value_heads", 1)
    head_dim = cfg.get("head_dim", hidden // n_heads)

    if name == "model.embed_tokens.weight":
        return (vocab, hidden)
    if name == "lm_head.weight":
        return (vocab, hidden)
    if name == "model.norm.weight":
        return (hidden,)

    # Layer norms (hidden) dan q/k_norm (head_dim)
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return (head_dim,)
    if "layernorm" in name or name == "model.norm.weight":
        return (hidden,)

    # MLP
    if ".mlp.gate_proj.weight" in name or ".mlp.up_proj.weight" in name:
        return (intermediate, hidden)
    if ".mlp.down_proj.weight" in name:
        return (hidden, intermediate)

    # Attention
    if ".self_attn.q_proj.weight" in name:
        return (n_heads * head_dim, hidden)
    if ".self_attn.k_proj.weight" in name or ".self_attn.v_proj.weight" in name:
        return (n_kv_heads * head_dim, hidden)
    if ".self_attn.o_proj.weight" in name:
        return (hidden, n_heads * head_dim)

    # Attention biases (jika ada)
    if ".self_attn.q_proj.bias" in name:
        return (hidden,)
    if ".self_attn.k_proj.bias" in name or ".self_attn.v_proj.bias" in name:
        return (n_kv_heads * head_dim,)
    if ".self_attn.o_proj.bias" in name:
        return (hidden,)

    # MLP biases (jika ada)
    if ".mlp.gate_proj.bias" in name or ".mlp.up_proj.bias" in name:
        return (intermediate,)
    if ".mlp.down_proj.bias" in name:
        return (hidden,)

    return None


def _sanitize_rope_params(raw_cfg):
    changed = False
    for key in ("rope_parameters", "rope_scaling"):
        rope = raw_cfg.get(key)
        if not isinstance(rope, dict):
            continue
        entries = [rope] + [v for v in rope.values() if isinstance(v, dict)]
        for entry in entries:
            rope_type = entry.get("rope_type")
            if rope_type not in ("default", "linear", "dynamic", "llama3", "yarn",
                                 "longrope", "proportional", "su", "mrope"):
                entry["rope_type"] = "default"
                changed = True
    return changed


try:
    _INT4PACK_CPU = torch._convert_weight_to_int4pack_for_cpu
    _INT4MM_CPU = torch._weight_int4pack_mm_for_cpu
except AttributeError:
    _INT4PACK_CPU = None
    _INT4MM_CPU = None


class Int4WeightLinear(nn.Module):
    """Linear Q4M langsung: bobot 4-bit terpack."""

    def __init__(self, packed, scales_zeros, bias=None):
        super().__init__()
        self.register_buffer("weight", packed)
        self.register_buffer("scales_zeros", scales_zeros)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = _INT4MM_CPU(x2, self.weight, 32, self.scales_zeros)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], y.shape[-1])


def _quantize_fp_to_int4pack(weight, chunk_rows=4096):
    """Float [N,K] -> int4pack CPU + scales_zeros (per 32 elemen)."""
    n, k = int(weight.shape[0]), int(weight.shape[1])
    if k % 32 != 0 or n % 16 != 0:
        raise ValueError(f"int4pack butuh N%16==0 dan K%32==0, dapat {tuple(weight.shape)}")
    step = max(32, (int(chunk_rows) // 32) * 32)
    w = weight.to(torch.float32)
    packed = torch.empty((n, k // 2), dtype=torch.uint8)
    scales = torch.empty((n, k // 32), dtype=torch.float32)
    for start in range(0, n, step):
        end = min(start + step, n)
        rows = end - start
        blk = w[start:end].reshape(rows * (k // 32), 32)
        amax = blk.abs().amax(dim=1).clamp_min(1e-8)
        s16 = (amax / 8.0).to(torch.float16)
        s32 = s16.float().unsqueeze(1)
        q = torch.round(blk / s32).clamp_(-8, 7).to(torch.int32) + 8
        scales[start:end] = s16.float().reshape(rows, k // 32)
        packed[start:end] = _INT4PACK_CPU(q.reshape(rows, k), 1)
        del blk, q
    zeros = torch.zeros_like(scales)
    scales_zeros = torch.stack([scales, zeros], dim=2).transpose(0, 1).contiguous()
    return packed, scales_zeros.to(torch.bfloat16)


def _pack_q4m_int4(raw_u8, n_elems, shape, chunk_rows=4096):
    """Q4M (blok 32, skala f16, nibble) -> paket int4 CPU + scales_zeros."""
    n, k = int(shape[0]), int(shape[1])
    if k % 32 != 0 or n % 16 != 0:
        raise ValueError(f"int4pack butuh N%16==0 dan K%32==0, dapat {shape}")
    if n_elems is not None and n_elems != n * k:
        raise ValueError(f"n_elems {n_elems} != {n}*{k}")
    nb = n * k // 32
    if raw_u8.numel() < nb * Q4_BLOCK_BYTES:
        raise ValueError(f"Q4DR9: byte kurang ({raw_u8.numel()} < {nb * Q4_BLOCK_BYTES})")

    blk = raw_u8[:nb * Q4_BLOCK_BYTES].reshape(nb, Q4_BLOCK_BYTES)
    s = blk[:, :2].contiguous().view(torch.float16).float().reshape(n, k // 32)
    packed = torch.empty((n, k // 2), dtype=torch.uint8)

    if n % 32 == 0:
        step = max(32, (int(chunk_rows) // 32) * 32)
        bounds = list(range(0, n, step)) or [0]
    else:
        bounds = [0]

    for start in bounds:
        end = min(start + step, n) if len(bounds) > 1 else n
        rows = end - start
        b = blk[start * (k // 32):end * (k // 32)]
        pk = b[:, 2:].reshape(rows, k // 2)
        codes = torch.empty((rows, k), dtype=torch.int32)
        codes[:, 0::2] = pk & 0x0F
        codes[:, 1::2] = pk >> 4
        packed[start:end] = _INT4PACK_CPU(codes, 1)
        del codes

    zeros = torch.zeros_like(s)
    scales_zeros = torch.stack([s, zeros], dim=2).transpose(0, 1).contiguous()
    return packed, scales_zeros.to(torch.bfloat16)


def _is_int4_linear_name(name):
    """True untuk bobot Linear Gemma3 yang bisa memakai kernel int4 CPU."""
    if name == "lm_head.weight":
        return True
    if ".self_attn." in name and name.endswith(
            (".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight")):
        return True
    if ".mlp." in name and name.endswith(
            (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
        return True
    return False


def _set_submodule(model, name, module):
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, module)


def load_int4_state_dict(model_dir: Path, progress=True):
    """Muat Q4M langsung menjadi (other, int4_packed, int4_scales, index).

    - Linear mode signed-grid/Q4M -> dipack 4-bit
    - fp16-split (k_proj/v_proj, bias, norm) -> dibaca langsung
    - Embedding -> untuk lookup; lm_head tied -> int4 dari byte Q4M
      embedding yang sama.
    """
    if _INT4PACK_CPU is None or _INT4MM_CPU is None:
        raise RuntimeError(
            "Torch ini tidak punya kernel int4 CPU "
            "(_convert_weight_to_int4pack_for_cpu / _weight_int4pack_mm_for_cpu).")

    cfg = load_config_file(model_dir / "config.json")
    files, qnm_index, tensor_shapes, tensor_dtypes, n_total = \
        _scan_shard_headers(model_dir)

    has_lm_head = "lm_head.weight" in tensor_shapes

    other = {}
    qweights = {}
    qscales = {}
    t0 = time.perf_counter()
    done_bytes = 0
    done = 0
    for path in files:
        from safetensors import safe_open
        f = safe_open(path, framework="pt")
        for name in f.keys():
            u8 = f.get_tensor(name)
            meta = qnm_index.get(name)
            if meta is not None:
                mode = meta.get("mode", "signed-grid")
                n_elems = meta.get("n_elems", None)
                orig_shape = (tuple(meta["shape"]) if "shape" in meta
                              else tensor_shapes.get(name))
            else:
                mode, n_elems = None, None   # safetensors biasa
                orig_shape = tensor_shapes.get(name)

            if orig_shape is not None and len(orig_shape) == 1:
                inferred = _infer_gemma3_shape(name, cfg)
                if inferred is not None:
                    orig_shape = inferred

            q4_mode = mode in ("signed-grid", "qnm", "unknown")
            if meta is None and _is_int4_linear_name(name):
                # Model safetensors biasa: kuantisasi langsung ke int4pack
                if orig_shape is None or len(orig_shape) != 2:
                    raise ValueError(f"{name!r}: shape 2D tidak diketahui")
                w_bf16 = u8 if u8.dtype == torch.bfloat16 else u8.to(torch.bfloat16)
                packed, scales_zeros = _quantize_fp_to_int4pack(w_bf16)
                qweights[name] = packed
                qscales[name] = scales_zeros
                done_bytes += packed.numel() + scales_zeros.numel() * 2
                del packed, scales_zeros, w_bf16
            elif _is_int4_linear_name(name) and q4_mode:
                if orig_shape is None or len(orig_shape) != 2:
                    raise ValueError(f"{name!r}: shape 2D tidak diketahui")
                bits = meta.get("bits", 4)
                blok = meta.get("blok", Q4_BLOCK)
                blkb = meta.get("block_bytes", Q4_BLOCK_BYTES)
                if bits != 4 or blok != Q4_BLOCK or blkb != Q4_BLOCK_BYTES:
                    raise ValueError(
                        f"{name!r}: hanya Q4M blok {Q4_BLOCK} yang didukung "
                        f"(bits={bits}, blok={blok}, block_bytes={blkb})")
                packed, scales_zeros = _pack_q4m_int4(u8, n_elems, orig_shape)
                qweights[name] = packed
                qscales[name] = scales_zeros
                done_bytes += packed.numel() + scales_zeros.numel() * 2
                del packed, scales_zeros
            else:
                w = reconstruct_tensor(u8, mode, n_elems, orig_shape,
                                       out_dtype=torch.bfloat16)
                other[name] = w
                done_bytes += w.numel() * w.element_size()

                # lm_head tied: pakai kode Q4M embedding yang sama.
                if (name == "model.embed_tokens.weight" and not has_lm_head
                        and q4_mode):
                    bits = meta.get("bits", 4)
                    blok = meta.get("blok", Q4_BLOCK)
                    blkb = meta.get("block_bytes", Q4_BLOCK_BYTES)
                    if bits == 4 and blok == Q4_BLOCK and blkb == Q4_BLOCK_BYTES:
                        packed, scales_zeros = _pack_q4m_int4(
                            u8, n_elems, tuple(w.shape))
                        qweights["lm_head.weight"] = packed
                        qscales["lm_head.weight"] = scales_zeros
                        done_bytes += packed.numel() + scales_zeros.numel() * 2
                        del packed, scales_zeros

            done += 1
            if progress and (done % 50 == 0 or done == n_total):
                print(f"  [{done}/{n_total}] tensor | "
                      f"{done_bytes / 1e9:.2f} GB | "
                      f"{time.perf_counter() - t0:.1f}s", flush=True)

    # Tied lm_head untuk model safetensors biasa (tanpa qnm.index):
    if (not has_lm_head and "lm_head.weight" not in qweights
            and "model.embed_tokens.weight" in other):
        w_bf16 = other["model.embed_tokens.weight"]
        if w_bf16.dtype != torch.bfloat16:
            w_bf16 = w_bf16.to(torch.bfloat16)
        packed, scales_zeros = _quantize_fp_to_int4pack(w_bf16)
        qweights["lm_head.weight"] = packed
        qscales["lm_head.weight"] = scales_zeros

    return other, qweights, qscales, qnm_index


def build_gemma3_model_int4(model_dir, other, qweights, qscales):
    """Bangun Gemma3 dengan modul Linear int4 CPU."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        raw_cfg, repaired = _parse_config_text(f.read(), cfg_path)
    _sanitize_rope_params(raw_cfg)
    if raw_cfg.get("model_type") not in ("gemma3_text", "gemma3"):
        raw_cfg["model_type"] = "gemma3_text"
        raw_cfg["architectures"] = ["Gemma3ForCausalLM"]
        repaired = True
    if repaired:
        with open(cfg_path, "w") as f:
            json.dump(raw_cfg, f)

    cfg = AutoConfig.from_pretrained(model_dir)
    if "model.embed_tokens.weight" in other:
        cfg.vocab_size = other["model.embed_tokens.weight"].shape[0]

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Gemma3ForCausalLM(cfg)

        # Assign parameter non-Linear (embedding, norm, bias).
        missing, unexpected = model.load_state_dict(other, strict=False, assign=True)

        # Ganti Linear yang punya bobot Q4M terpack dengan versi int4.
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                wname = name + ".weight"
                if wname in qweights:
                    bias = other.get(name + ".bias", None)
                    _set_submodule(model, name, Int4WeightLinear(
                        qweights.pop(wname), qscales.pop(wname), bias))

        # lm_head: pakai Q4M (tied atau tersimpan terpisah).
        if not isinstance(model.lm_head, Int4WeightLinear):
            wname = "lm_head.weight"
            if wname in qweights:
                _set_submodule(model, "lm_head", Int4WeightLinear(
                    qweights.pop(wname), qscales.pop(wname), None))
            else:
                model.tie_weights()

        # Re-init rotary embedding (buffer non-persistent yang jadi meta).
        from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
        with torch.device("cpu"):
            for mod in model.modules():
                if isinstance(mod, Gemma3RotaryEmbedding):
                    mod.__init__(mod.config)

        # Sisa buffer meta -> init (embed_scale = sqrt(hidden_size), BUKAN nol)
        for mod in model.modules():
            for buf_name, buf in list(mod._buffers.items()):
                if buf is not None and buf.is_meta:
                    if buf_name == "embed_scale" and hasattr(mod, "scalar_embed_scale"):
                        mod._buffers[buf_name] = torch.tensor(
                            float(mod.scalar_embed_scale),
                            dtype=torch.bfloat16, device="cpu")
                    else:
                        mod._buffers[buf_name] = torch.zeros(
                            buf.shape, dtype=buf.dtype, device="cpu")

        model.eval()
    finally:
        torch.set_default_dtype(prev_dtype)
    return model, missing, unexpected


# ─────────────────────────────────────────────────────────────────────────────
#  ATTENTION DR9 DENGAN REPETITION PENALTY LANGSUNG
# ─────────────────────────────────────────────────────────────────────────────
_DR9_STATS = {"calls": 0}


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _mask_threshold(dtype):
    if isinstance(dtype, torch.dtype) and dtype.is_floating_point:
        return torch.finfo(dtype).min / 2
    return MASK_THRESH


def dr9_attention_weights(attn_weights, mode="dr9", repetition_penalty=1.15):
    """Menghitung bobot atensi DR9 dengan penalti pengulangan langsung."""
    valid_mask = attn_weights > _mask_threshold(attn_weights.dtype)
    w = dr9_nearest_torch(attn_weights)

    # ──── REPETITION PENALTY LANGSUNG PADA LOGITS ATTENTION ────
    if repetition_penalty != 1.0:
        seq_len = w.shape[-1]
        if seq_len > 1:
            # Berikan penalti pada skor bobot masa lalu agar tidak looping/terkunci
            w = torch.where(w > 0, w / repetition_penalty, w * repetition_penalty)

    if mode == "noexp":
        a = torch.abs(torch.where(valid_mask, w, torch.zeros_like(w)))
        total = a.sum(dim=-1, keepdim=True)
        return torch.where(total > 0, a / torch.where(total > 0, total, torch.ones_like(total)),
                           torch.zeros_like(a))

    m = torch.amax(torch.where(valid_mask, w, torch.full_like(w, float("-inf"))),
                   dim=-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    w_shifted = torch.where(valid_mask, w - m, torch.zeros_like(w))
    flat_mask = (torch.abs(w_shifted - w_shifted.mean(dim=-1, keepdim=True)) < 1e-2)
    w_penalized = torch.where(flat_mask & valid_mask, w_shifted - 0.5, w_shifted)

    w_scaled = w_penalized * 1.25
    w_safe = torch.clamp(w_scaled, min=-15.0, max=15.0)
    E = 2.718281828459045
    e_raw = E ** w_safe
    noise = (torch.rand_like(e_raw, device=e_raw.device) - 0.5) * 2e-3
    e_stochastic = e_raw + torch.where(valid_mask, noise, torch.zeros_like(noise))
    e_stochastic = torch.clamp(e_stochastic, min=1e-8)
    e = torch.where(valid_mask, e_stochastic, torch.zeros_like(e_stochastic))

    total = e.sum(dim=-1, keepdim=True)
    p_raw = torch.where(total > 0, e / torch.where(total > 0, total, torch.ones_like(total)),
                        torch.zeros_like(e))
    return torch.where(valid_mask, p_raw, torch.zeros_like(p_raw))


def dr9_attention_forward(module, query, key, value, attention_mask,
                          dropout=0.0, scaling=None, softcap=None, **kwargs):
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    rep_penalty = getattr(module, "_repetition_penalty", 1.15)
    attn_weights = dr9_attention_weights(
        attn_weights, 
        getattr(module, "_dr9_mode", "dr9"),
        repetition_penalty=rep_penalty
    )
    attn_weights = attn_weights.to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    _DR9_STATS["calls"] += 1
    return attn_output, attn_weights


def set_attention_impl(model, impl):
    seen = set()
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation") and id(c) not in seen:
            seen.add(id(c))
            c._attn_implementation = impl
    c = getattr(model, "config", None)
    if c is not None and hasattr(c, "_attn_implementation"):
        c._attn_implementation = impl


def install_dr9_attention(model, mode, seed=42, repetition_penalty=1.15):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ALL_ATTENTION_FUNCTIONS.register("dr9", dr9_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("dr9_noexp", dr9_attention_forward)
    if ALL_MASK_ATTENTION_FUNCTIONS is not None:
        ALL_MASK_ATTENTION_FUNCTIONS.register("dr9", eager_mask)
        ALL_MASK_ATTENTION_FUNCTIONS.register("dr9_noexp", eager_mask)

    impl = "dr9_noexp" if mode == "noexp" else "dr9"

    for m in model.modules():
        if hasattr(m, "q_proj") and hasattr(m, "num_key_value_groups"):
            m._dr9_mode = mode
            m._repetition_penalty = repetition_penalty

    _DR9_STATS["calls"] = 0
    set_attention_impl(model, impl)
    return impl
# ─────────────────────────────────────────────────────────────────────────────
#  GENERASI TOKEN
# ─────────────────────────────────────────────────────────────────────────────
def generate_reply(model, tokenizer, prompt, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt")
    eos_ids = collect_stop_ids(model, tokenizer)

    def pick(logits):
        if temperature > 0:
            logits = logits / temperature
            if top_p < 1.0:
                k = min(1000, logits.shape[-1])
                top_logits, top_idx = torch.topk(logits, k)
                probs = torch.softmax(top_logits.float(), dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                remove = cum > top_p
                remove[0] = False
                probs[remove] = 0.0
                probs = probs / probs.sum()
                sel = torch.multinomial(probs, 1)
                return top_idx[sel].view(1)
            probs = torch.softmax(logits.float(), dim=-1)
            return torch.multinomial(probs, 1)
        return logits.argmax(dim=-1, keepdim=True)

    t0 = time.time()
    n_new = 0
    reply = ""
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)
        t_prefill = time.time() - t0
        past = out.past_key_values
        logits = out.logits[0, -1]
        t_dec0 = time.time()
        for _ in range(max_new_tokens):
            next_token = pick(logits)
            if next_token.item() in eos_ids:
                break
            piece = tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(piece, end="", flush=True)
            reply += piece
            n_new += 1
            out = model(input_ids=next_token.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1]
        t_dec = time.time() - t_dec0
    print()
    del past
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if n_new:
        tok_per_s = n_new / max(t_dec, 1e-9)
        print(f"      [prefill {t_prefill:.1f}s | decode {n_new} token "
              f"{t_dec:.1f}s = {tok_per_s:.2f} tok/s]")
    else:
        print(f"      [prefill {t_prefill:.1f}s | decode 0 token — token stop "
              f"langsung keluar (jawaban kosong)]")
    return reply


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Chat Gemma3-1B-IT-Riset (U8/Q4M) — CPU int4.")
    ap.add_argument("model_dir", nargs="?", default=None,
                    help="Path ke direktori model (safetensors + tokenizer), "
                         "mis. /home/witi/data/gemma3-1b-it atau .../gemma3-1b-it-riset")
    ap.add_argument("--model-dir", dest="model_dir_opt", default=None,
                    help="Alias untuk model_dir")
    ap.add_argument("--attention", choices=["dr9", "noexp", "softmax"], default="dr9",
                    help="dr9=DR9+kuantisasi+nearest+exp/norm (default); "
                         "noexp=legacy tanpa exp; softmax=baseline")
    ap.add_argument("--seed", type=int, default=42, help="seed RNG")
    ap.add_argument(
        "--system", 
        default="Selalu jawab dan berikan respons dalam Bahasa Indonesia.", 
        help="system prompt (default: Bahasa Indonesia)"
    )
    ap.add_argument("--prompt", default=None, help="mode single-shot")
    ap.add_argument("--max-new-tokens", type=int, default=32000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=8, help="jumlah thread torch CPU")
    args = ap.parse_args()

    model_dir = args.model_dir or args.model_dir_opt
    if not model_dir:
        ap.error("model_dir wajib diisi (positional atau --model-dir)")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[Weight] Memuat Q4M dari {model_dir} -> int4pack CPU ...")
    other, qweights, qscales, qnm_index = load_int4_state_dict(Path(model_dir))
    print(f"[Weight] {len(qweights)} bobot Linear -> int4, "
          f"{len(other)} tensor lain -> bf16")

    print("[Model] Membangun arsitektur Gemma3 (Linear int4) ...")
    model, missing, unexpected = build_gemma3_model_int4(
        model_dir, other, qweights, qscales)
    del other, qweights, qscales
    print("[Model] Linear int4 CPU + parameter lain bf16 terpetakan")
    compute_dtype = "bf16+int4"

    # Kernel int4pack hanya tersedia untuk CPU.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("[Device] int4 memakai kernel CPU; GPU diabaikan.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Sebutkan ibu kota Indonesia."}],
        tokenize=False, add_generation_prompt=True)
    pin = tokenizer(probe, return_tensors="pt").to(device)
    with torch.inference_mode():
        ref_logits = model(**pin).logits[0, -1].float()
    top_ref = set(ref_logits.topk(5).indices.tolist())

    if args.attention != "softmax":
        impl = install_dr9_attention(model, args.attention, args.seed)
        with torch.inference_mode():
            var_logits = model(**pin).logits[0, -1].float()
        cos = F.cosine_similarity(ref_logits, var_logits, dim=0).item()
        top_var = set(var_logits.topk(5).indices.tolist())
        print(f"[Verify] impl={impl} cosine logits vs softmax = {cos:.6f}, "
              f"top-5 overlap = {len(top_ref & top_var)}/5")
        print(f"[Verify] panggilan attention/forward = {_DR9_STATS['calls']} "
              f"(harus = jumlah layer = {model.config.num_hidden_layers})")
        _DR9_STATS["calls"] = 0
        if args.attention == "noexp":
            print("   ⚠️ mode noexp = algoritma legacy (tanpa exp) — output akan rusak. "
                  "Pakai --attention dr9 untuk hasil koheren.")
    else:
        print("🔬 [Verify] mode=softmax (baseline eager, tanpa DR9)")

    print(f"🖥️ [Device] {device.type.upper()} | hitung={compute_dtype} | seed={args.seed}")

    print("\n" + "=" * 70)
    print(f"SYSTEM CHAT INTERFACE RIIL AKTIF UNTUK (Model: {config_model_type(model_dir).upper()}) QNM 4BIT WITI AI POKDARTIK")
    print(f"Hak Cipta (c) 2026 oleh Tim Perekayasa AI POKDARTIK")
    print(f"Perangkat lunak inti ini dilisensikan khusus untuk penggunaan NON-KOMERSIAL,")
    print(f"pendidikan, riset, dan peninjauan ilmiah (peer-review). Pengguna diizinkan") 
    print(f"untuk mengunduh, menjalankan, dan membagikan kode ini dengan tetap mencantumkan") 
    print(f"kredit/atribusi kepada POKDARTIK. Dilarang keras menggunakannya untuk tujuan komersial.")
    if args.attention == "softmax":
        print(" Pikir Model: SOFTMAX STANDAR (baseline pembanding)")
    elif args.attention == "noexp":
        print(" Pikir Model: DR9 + Kuantisasi + Nearest + abs/norm (legacy, tanpa exp)")
    else:
        print(" Pikir Model: DR9 + Kuantisasi + Nearest + exp/norm + SR (Aritmatika, stokastik ter-seed)")
    print(" Ketik 'keluar', 'exit', atau '/quit' untuk menyudahi percakapan.")
    print("=" * 70 + "\n")

    def run_turn(user_text, history):
        max_context = 2
        truncated_history = history[-max_context:] if len(history) > max_context else history

        messages = list(truncated_history)
        if args.system:
            messages.insert(0, {"role": "system", "content": args.system})
        messages.append({"role": "user", "content": user_text})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        print("AI: ", end="", flush=True)

        reply = generate_reply(model, tokenizer, prompt,
                               args.max_new_tokens, args.temperature, args.top_p)

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        if not reply.strip():
            print("      [peringatan: balasan kosong — masukkan pesan ini ke "
                  "riwayat; ketik '/reset' kalau jawaban berikutnya makin "
                  "tidak fokus]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return reply

    try:
        if args.prompt is not None:
            run_turn(args.prompt, [])
            return

        history = []
        while True:
            try:
                user_input = input("Pengguna: ")
            except (EOFError, KeyboardInterrupt):
                print("\nMenutup runtime.")
                break
            if user_input.lower() in ["keluar", "exit", "/exit", "/quit", "quit"]:
                print("Menutup runtime.")
                break
            if user_input.strip() in ("/reset", "/new"):
                history = []
                print("(percakapan direset)")
                continue
            if not user_input.strip():
                continue
            run_turn(user_input, history)
    finally:
        pass


if __name__ == "__main__":
    main()
