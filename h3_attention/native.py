"""
Native ComfyUI path for H3 Attention Control.

Intervenes MiniMax H3 attention at inference time by chain-patching the native
block-replace hook used by comfy/ldm/minimax/model.py:

    transformer_options["patches_replace"]["dit"][("double_block", i)]

No weights are modified. Fail-safe by design:

* If the model has no native ``.blocks`` (not a native H3 diffusion model) the
  model is returned unchanged and control stays inactive.
* If the controlled attention path raises (OOM, backend issue) we fall back to
  the block's native ``attn`` call, so generation still completes.
* If every control strength is 0 / ``config.is_active`` is False, the patch
  simply delegates to the previous patch (e.g. BlockSparseAttention) or to the
  native block, i.e. plain H3 (with any acceleration intact).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import AttentionControlConfig
from .metrics import AttentionMetrics

logger = logging.getLogger(__name__)

#: Sequences larger than this are never controlled (native/sparse path takes over).
MAX_CONTROLLED_SEQ = 65536

#: Query-chunk size for the controlled dense attention (reduced on OOM).
_INITIAL_CHUNK = 1024
_MIN_CHUNK = 128

#: Rows sampled from the video segment for debug/adaptive metrics.
_METRIC_SAMPLES = 32


# ---------------------------------------------------------------------------
# Step / schedule plumbing
# ---------------------------------------------------------------------------


def maybe_update_step(controller, transformer_options: Dict[str, Any]) -> None:
    """Advance the controller one denoising step using the sampler's sigmas.

    ``transformer_options["sigmas"]`` holds the current sigma (set per step by
    comfy/samplers.py) and ``["sample_sigmas"]`` the full schedule. The step
    index is recovered by nearest match; the resulting step drives the phase
    schedule (``AttentionSchedule``) and the identity-decay factor.
    """
    cur = transformer_options.get("sigmas")
    sched = transformer_options.get("sample_sigmas")
    if cur is None or sched is None:
        return
    try:
        c = float(cur.reshape(-1)[0].item()) if torch.is_tensor(cur) else float(cur)
        s = sched.float().reshape(-1)
        if s.numel() == 0:
            return
        step = int(torch.argmin((s - c).abs()).item())
        total = max(1, s.numel() - 1)
    except Exception:  # pragma: no cover - defensive
        logger.debug("h3-attention: could not derive step from sigmas", exc_info=True)
        return
    controller.update_step(step, total, c)


# ---------------------------------------------------------------------------
# Bias planner
# ---------------------------------------------------------------------------


def _seg_list(layout) -> List[Tuple[int, int, str]]:
    return [(int(a), int(b), str(k)) for (a, b, k) in layout.segments]


class NativeBiasPlanner:
    """Builds the additive attention bias ``rows(a, b) -> [b - a, S]``.

    Segment-constant mechanisms (reference, prompt, identity-global,
    subject-separation, ref-audio) live in a ``[n_seg, n_seg]`` pair table and
    are gathered per chunk. Pair-dependent mechanisms (identity-local decay,
    temporal windows, audio<->video sync) are evaluated per chunk as [chunk, S]
    masks so memory stays bounded (chunk ~1024 x S ~36k => ~150 MB fp32).

    The semantic formulas mirror ``H3AttentionController.compute_attention_bias``
    (the Diffusers path) so both integrations behave consistently.
    """

    def __init__(self, layout, config: AttentionControlConfig, state) -> None:
        segs = _seg_list(layout)
        self.n_seg = len(segs)
        S = int(layout.seq_len)
        self.seq_len = S

        seg_id = torch.empty(S, dtype=torch.long)
        for i, (a, b, _k) in enumerate(segs):
            seg_id[a:b] = i
        self.seg_id = seg_id

        pos = layout.position_ids
        if not torch.is_tensor(pos):
            pos = torch.as_tensor(pos)
        self.t = pos[:, 0].detach().to("cpu", torch.float32)

        kinds = [k for (_, _, k) in segs]
        video = [i for i, k in enumerate(kinds) if k == "video"]
        audio = [i for i, k in enumerate(kinds) if k == "audio"]
        self.video_seg: Optional[int] = video[-1] if video else None
        self.audio_seg: Optional[int] = audio[-1] if audio else None
        self.text_segs: List[int] = [i for i, k in enumerate(kinds) if k == "text"]
        # Keyframe visual/audio segments (fl2va) behave like references.
        self.ref_visual: List[int] = [i for i, k in enumerate(kinds) if k in ("ref_img", "cond")]
        self.ref_audio: List[int] = [
            i for i, k in enumerate(kinds) if k in ("ref_audio", "cond_audio")
        ]

        self.anchor_seg: Optional[int] = None
        if self.ref_visual:
            idx = int(getattr(config, "identity_reference_idx", 0) or 0)
            idx = max(0, min(idx, len(self.ref_visual) - 1))
            self.anchor_seg = self.ref_visual[idx]

        sig = getattr(layout, "signature", None)
        # Stable identity of this layout for the cache key (id() alone can be
        # reused by the allocator if the layout object is rebuilt per forward).
        self.layout_key = (
            tuple(sig) if sig is not None else None,
            S,
            self.n_seg,
            segs[0],
            segs[-1],
        )
        self.latent_t = max(1, int(sig[1])) if sig else 1
        self.audio_t = max(1, int(sig[4])) if sig else 1

        if self.video_seg is not None:
            vlen = segs[self.video_seg][1] - segs[self.video_seg][0]
            vpf = max(1, vlen // self.latent_t)
            self.vframe = torch.arange(vlen, dtype=torch.long) // vpf
            self.video_span = (segs[self.video_seg][0], segs[self.video_seg][1])
        else:
            self.vframe = None
            self.video_span = None

        if self.audio_seg is not None:
            alen = segs[self.audio_seg][1] - segs[self.audio_seg][0]
            apf = max(1, alen // self.audio_t)
            self.aframe = torch.arange(alen, dtype=torch.long) // apf
            self.audio_span = (segs[self.audio_seg][0], segs[self.audio_seg][1])
        else:
            self.aframe = None
            self.audio_span = None

        # Scalar snapshots (config may be rewritten by the phase schedule each
        # step; the planner cache key includes the strengths so this is safe).
        self.ref_strength = float(config.reference_strength)
        self.ref_mode = str(config.reference_boost_mode)
        self.identity_lock = float(config.identity_lock)
        self.identity_decay = float(config.identity_temporal_decay)
        self.identity_scope = str(config.identity_spatial_scope)
        self.prompt_adherence = float(config.prompt_adherence)
        self.temporal_lock = float(config.temporal_lock)
        self.temporal_window = int(config.temporal_window)
        self.temporal_mode = str(config.temporal_mode)
        self.subject_separation = float(config.subject_separation)
        self.audio_strength = float(config.audio_strength)
        self.audio_video_sync = bool(config.audio_video_sync)
        self.bias_scale = float(config.attention_bias_scale)
        self.clamp_bias = bool(config.clamp_bias)
        self.clamp_value = float(config.bias_clamp_value)

        step_ratio = state.step / max(1, state.total_steps)
        self.identity_decayed = self.identity_lock * (self.identity_decay ** (step_ratio * 10.0))

        self.table = self._build_table()

        self.text_cols = self._span_cols(self.text_segs)
        self.ref_cols = self._span_cols(self.ref_visual + self.ref_audio)
        self._anchor_span: Optional[Tuple[int, int]] = None
        if self.anchor_seg is not None:
            idx = torch.nonzero(self.seg_id == self.anchor_seg, as_tuple=False)
            if idx.numel() > 0:
                self._anchor_span = (int(idx[0].item()), int(idx[-1].item()) + 1)
        self._device: Optional[torch.device] = None

    # -- construction helpers ------------------------------------------------

    def _span_cols(self, segs: Sequence[int]) -> torch.Tensor:
        if not segs:
            return torch.empty(0, dtype=torch.long)
        parts = []
        for i in segs:
            # spans are contiguous in seg order for these kinds in practice;
            # gather explicitly instead of assuming contiguity.
            mask = self.seg_id == i
            parts.append(torch.nonzero(mask, as_tuple=False).squeeze(-1))
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def _build_table(self) -> torch.Tensor:
        T = torch.zeros(self.n_seg, self.n_seg, dtype=torch.float32)
        if self.video_seg is not None and self.audio_seg is None:
            pass  # layout always has both; defensive only

        # 1) Reference boost (additive mode only, mirroring the Diffusers path).
        if self.ref_strength > 0 and self.ref_mode == "additive":
            if self.video_seg is not None:
                for r in self.ref_visual:
                    T[self.video_seg, r] += self.ref_strength
                    T[r, self.video_seg] += self.ref_strength * 0.5
            if self.audio_seg is not None:
                for r in self.ref_audio:
                    T[self.audio_seg, r] += self.ref_strength * 0.8

        # 3) Prompt adherence: targets -> text (audio attenuated 0.8x).
        if self.prompt_adherence > 0:
            for ts in self.text_segs:
                if self.video_seg is not None:
                    T[self.video_seg, ts] += self.prompt_adherence
                if self.audio_seg is not None:
                    T[self.audio_seg, ts] += self.prompt_adherence * 0.8

        # 2) Identity lock, global scope (local/patch handled per chunk).
        if (
            self.identity_decayed > 0
            and self.identity_scope == "global"
            and self.anchor_seg is not None
            and self.video_seg is not None
        ):
            T[self.video_seg, self.anchor_seg] += self.identity_decayed

        # 5) Subject separation: negative bias across different visual refs.
        if self.subject_separation > 0:
            for i, a in enumerate(self.ref_visual):
                for b in self.ref_visual:
                    if a != b:
                        T[a, b] -= self.subject_separation

        # 6) Audio: audio -> ref_audio (0.7x), plus sync handled per chunk.
        if self.audio_strength > 0 and self.audio_seg is not None:
            for r in self.ref_audio:
                T[self.audio_seg, r] += self.audio_strength * 0.7

        return T

    def cache_key(self, controller) -> Tuple:
        c = controller.config
        return (
            self.layout_key,
            int(controller.state.step),
            round(float(c.reference_strength), 5),
            str(c.reference_boost_mode),
            round(float(c.identity_lock), 5),
            round(float(c.identity_temporal_decay), 5),
            str(c.identity_spatial_scope),
            int(getattr(c, "identity_reference_idx", 0) or 0),
            round(float(c.prompt_adherence), 5),
            round(float(c.temporal_lock), 5),
            int(c.temporal_window),
            str(c.temporal_mode),
            round(float(c.subject_separation), 5),
            round(float(c.audio_strength), 5),
            bool(c.audio_video_sync),
            round(float(c.attention_bias_scale), 5),
            bool(c.clamp_bias),
            round(float(c.bias_clamp_value), 5),
        )

    # -- per-chunk rows ------------------------------------------------------

    def rows(self, a: int, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        seg = self.seg_id.to(device)
        out = self.table.to(device).index_select(0, seg[a:b]).index_select(1, seg)

        # 2b) Identity lock, local/patch scope: 0.9^|dt| decay towards anchor.
        if (
            self.identity_decayed > 0
            and self.identity_scope in ("local", "patch")
            and self.anchor_seg is not None
            and self.video_span is not None
        ):
            va, vb = self.video_span
            ca, cb = self.anchor_span if self.anchor_span else (0, 0)
            qa, qb = max(a, va), min(b, vb)
            if qa < qb and cb > ca:
                t = self.t.to(device)
                tq = t[qa:qb]
                ta = t[ca:cb].mean()
                w = (0.9 ** (tq - ta).abs()).unsqueeze(1)
                out[qa - a : qb - a, ca:cb] += self.identity_decayed * w

        # 4) Temporal lock, video <-> video in frame units.
        if self.temporal_lock > 0 and self.video_span is not None and self.vframe is not None:
            va, vb = self.video_span
            qa, qb = max(a, va), min(b, vb)
            if qa < qb:
                vf = self.vframe.to(device)
                d = (vf[qa - va : qb - va].unsqueeze(1) - vf.unsqueeze(0)).abs()
                mask = (d > 0) & (d <= self.temporal_window)
                if self.temporal_mode == "strict":
                    vals = torch.full_like(d, self.temporal_lock, dtype=torch.float32)
                else:  # smooth and adaptive behave as exponential decay
                    vals = self.temporal_lock * (0.8 ** d.to(torch.float32))
                out[qa - a : qb - a, va:vb] += torch.where(mask, vals, 0.0)

            # 4b) Audio <-> audio temporal, half strength, mode independent.
        if self.temporal_lock > 0 and self.audio_span is not None and self.aframe is not None:
            aa, ab = self.audio_span
            qa, qb = max(a, aa), min(b, ab)
            if qa < qb:
                af = self.aframe.to(device)
                d = (af[qa - aa : qb - aa].unsqueeze(1) - af.unsqueeze(0)).abs()
                mask = (d > 0) & (d <= self.temporal_window)
                out[qa - a : qb - a, aa:ab] += torch.where(mask, self.temporal_lock * 0.5, 0.0)

        # 6b) Audio <-> video sync by normalized time (audio_t != latent_t).
        if (
            self.audio_strength > 0
            and self.audio_video_sync
            and self.audio_span is not None
            and self.video_span is not None
            and self.vframe is not None
            and self.aframe is not None
        ):
            aa, ab = self.audio_span
            va, vb = self.video_span
            qa, qb = max(a, aa), min(b, ab)
            if qa < qb:
                af = self.aframe.to(device)
                target = torch.round(
                    af[qa - aa : qb - aa].float() * ((self.latent_t - 1) / max(1, self.audio_t - 1))
                ).long()
                d = (self.vframe.to(device).unsqueeze(0) - target.unsqueeze(1)).abs()
                out[qa - a : qb - a, va:vb] += torch.where(d <= 1, self.audio_strength, 0.0)

            qa, qb = max(a, va), min(b, vb)
            if qa < qb:
                vf = self.vframe.to(device)
                target = torch.round(
                    vf[qa - va : qb - va].float() * ((self.audio_t - 1) / max(1, self.latent_t - 1))
                ).long()
                d = (self.aframe.to(device).unsqueeze(0) - target.unsqueeze(1)).abs()
                out[qa - a : qb - a, aa:ab] += torch.where(d <= 1, self.audio_strength, 0.0)

        out = out * self.bias_scale
        if self.clamp_bias:
            out = out.clamp(-self.clamp_value, self.clamp_value)
        return out.to(dtype)

    @property
    def anchor_span(self) -> Optional[Tuple[int, int]]:
        return self._anchor_span


def get_planner(controller, layout) -> NativeBiasPlanner:
    """Cached planner; rebuilt when the layout, step or strengths change."""
    planner = getattr(controller, "_native_planner", None)
    if planner is not None:
        try:
            if planner.cache_key(controller) == getattr(controller, "_native_planner_key", None):
                return planner
        except Exception:  # pragma: no cover - defensive
            pass
    planner = NativeBiasPlanner(layout, controller.config, controller.state)
    controller._native_planner = planner
    controller._native_planner_key = planner.cache_key(controller)
    return planner


# ---------------------------------------------------------------------------
# Metrics (debug / adaptive only)
# ---------------------------------------------------------------------------


def _maybe_record_metrics(controller, attn, planner, q, k, transformer_options) -> None:
    cfg = controller.config
    if not (cfg.debug_attention or cfg.adaptive_mode):
        return
    if int(transformer_options.get("block_index", 0)) != 0:
        return
    step = controller.state.step
    if getattr(controller, "_native_metrics_step", None) == step:
        return
    if planner.video_span is None:
        return
    controller._native_metrics_step = step
    try:
        va, vb = planner.video_span
        n = min(_METRIC_SAMPLES, vb - va)
        if n <= 0:
            return
        rows = va + torch.linspace(0, vb - va - 1, n, device=q.device).long()
        scale = attn.head_dim**-0.5
        logits = torch.einsum("nhd,shd->nhs", q[rows].float(), k.float()) * scale
        probs = logits.softmax(-1)
        if planner.ref_cols.numel() > 0:
            ref_mass = probs.index_select(-1, planner.ref_cols.to(q.device)).sum(-1).mean().item()
        else:
            ref_mass = 0.0
        if planner.text_cols.numel() > 0:
            text_mass = probs.index_select(-1, planner.text_cols.to(q.device)).sum(-1).mean().item()
        else:
            text_mass = 0.0
        entropy = float(-(probs * probs.clamp_min(1e-9).log()).sum(-1).mean().item())
        max_entropy = float(torch.log(torch.tensor(float(probs.shape[-1]))))
        bias = planner.rows(va, min(va + 512, vb), q.device, torch.float32)
        controller.record_metrics(
            AttentionMetrics(
                reference_affinity=ref_mass,
                prompt_affinity=text_mass,
                temporal_affinity=0.0,
                audio_video_affinity=0.0,
                applied_bias_mean=float(bias.abs().mean().item()),
                applied_bias_max=float(bias.abs().max().item()),
                attention_entropy=entropy,
                attention_concentration=max(0.0, 1.0 - entropy / max(1e-6, max_entropy)),
                layer_idx=0,
                step=step,
                timestep=controller.state.current_timestep,
            )
        )
    except Exception:  # pragma: no cover - metrics must never break generation
        logger.debug("h3-attention: metrics sampling failed", exc_info=True)


# ---------------------------------------------------------------------------
# Controlled attention
# ---------------------------------------------------------------------------


def controlled_forward(attn, controller, layout, x, rope_freqs, transformer_options):
    """Replica of ``comfy.ldm.minimax.model.Attention.forward`` with a
    chunked dense attention whose logits receive the planner's additive bias."""
    s = x.shape[0]
    heads, head_dim = attn.heads, attn.head_dim
    q, k, v = attn.qkv_proj(x).split(heads * head_dim, dim=-1)
    v = v.view(s, heads, head_dim)

    if rope_freqs is not None:
        import comfy.model_management
        import comfy.quant_ops

        q = q.view(1, s, heads, head_dim)
        k = k.view(1, s, heads, head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot
            )
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(s, heads, head_dim))
        k = attn.k_norm(k.view(s, heads, head_dim))

    planner = get_planner(controller, layout)
    _maybe_record_metrics(controller, attn, planner, q, k, transformer_options)

    # [s, h, d] -> [1, h, s, d] for SDPA
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)

    chunk = _INITIAL_CHUNK
    out = None
    last_err: Optional[BaseException] = None
    while chunk >= _MIN_CHUNK:
        try:
            candidate = torch.empty((s, heads * head_dim), dtype=x.dtype, device=x.device)
            for a0 in range(0, s, chunk):
                b0 = min(a0 + chunk, s)
                bias = planner.rows(a0, b0, x.device, x.dtype)  # [c, S]
                oc = F.scaled_dot_product_attention(
                    qh[:, :, a0:b0], kh, vh, attn_mask=bias.unsqueeze(0).unsqueeze(0)
                )
                candidate[a0:b0] = oc.squeeze(0).transpose(0, 1).reshape(b0 - a0, heads * head_dim)
            out = candidate
            break
        except Exception as exc:  # noqa: BLE001 - shrink chunk then fall back
            last_err = exc
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            chunk //= 2

    if out is None:
        raise RuntimeError(
            f"controlled attention failed at all chunk sizes down to {_MIN_CHUNK}: {last_err}"
        )
    return attn.out_proj(out)


def make_controlled_attention(block, controller, layout):
    """Builds the ``attention(h, rope_freqs, transformer_options)`` callable
    injected via ``args["attention"]``. Fail-safe: any error falls back to the
    block's native attention so generation always completes."""
    attn = block.attn
    logged = {"done": False}

    def attention(h, rope_freqs=None, transformer_options={}):
        try:
            return controlled_forward(attn, controller, layout, h, rope_freqs, transformer_options)
        except Exception:
            if not logged["done"]:
                logged["done"] = True
                logger.warning(
                    "h3-attention: controlled attention failed; using native attention for this run",
                    exc_info=True,
                )
            return attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)

    return attention


# ---------------------------------------------------------------------------
# ModelPatcher installation (chain-patch)
# ---------------------------------------------------------------------------


def _make_block_patch(block, block_index: int, controller, prev, max_seq: int):
    """Wraps an existing ``patches_replace['dit']`` patch (or the bare block).

    Priority: when control is active for this call, OUR attention wins over
    whatever the previous patch (e.g. BlockSparseAttention) injected; when
    control is inactive the previous patch's attention is preserved, so
    acceleration keeps working at full speed.
    """

    def block_patch(args, extra):
        topts = args.get("transformer_options") or {}
        layout = args.get("layout") or topts.get("minimax_h3_layout")
        control = False
        if controller.config.is_active and layout is not None:
            if int(topts.get("block_index", block_index)) == 0:
                maybe_update_step(controller, topts)
            seq = int(getattr(layout, "seq_len", args["img"].shape[0]))
            if seq <= max_seq:
                control = True
            elif not getattr(controller, "_native_gate_logged", False):
                controller._native_gate_logged = True
                logger.warning(
                    "h3-attention: sequence %d exceeds max_controlled_seq=%d; delegating to the "
                    "native/sparse attention path (control inactive)",
                    seq,
                    max_seq,
                )

        def wrapped(inner_args):
            if control:
                inner_args = dict(inner_args)
                inner_args["attention"] = make_controlled_attention(block, controller, layout)
            return extra["original_block"](inner_args)

        if prev is not None:
            return prev(args, {**extra, "original_block": wrapped})
        return wrapped(args)

    block_patch._h3_owner = True  # type: ignore[attr-defined]
    block_patch._h3_prev = prev  # type: ignore[attr-defined]
    return block_patch


def install_comfyui(
    model_patcher, controller, max_seq: int = MAX_CONTROLLED_SEQ
) -> Tuple[Any, bool]:
    """Install the native block-patch chain on a ComfyUI ``ModelPatcher``.

    Returns ``(patched_model, installed)``. ``installed=False`` means the model
    had no native ``.blocks`` and was returned unchanged (fail-safe).
    """
    try:
        import comfy.patcher_extension

        prepare_key = comfy.patcher_extension.CallbacksMP.ON_PREPARE_STATE
    except ImportError:  # outside ComfyUI (tests); same string value anyway
        prepare_key = "on_prepare_state"

    m = model_patcher.clone()
    try:
        diffusion = m.get_model_object("diffusion_model")
    except Exception:
        diffusion = None

    if diffusion is None or not hasattr(diffusion, "blocks"):
        logger.warning(
            "h3-attention: native MiniMax H3 model (`.blocks`) not found on %s; passing the model "
            "through unchanged (control inactive, fail-safe)",
            type(model_patcher).__name__,
        )
        return m, False

    blocks = list(diffusion.blocks)

    def install_into(transformer_options: Dict[str, Any]) -> None:
        pr = transformer_options.setdefault("patches_replace", {})
        dit = pr.setdefault("dit", {})
        for i, block in enumerate(blocks):
            key = ("double_block", i)
            cur = dit.get(key)
            if getattr(cur, "_h3_owner", False):
                # Our own previous wrapper: unwrap to its base to avoid nesting.
                prev = getattr(cur, "_h3_prev", None)
            else:
                prev = cur
            dit[key] = _make_block_patch(block, i, controller, prev, max_seq)

    # Copy-on-write into this clone's model_options (upstream nodes untouched).
    mo = dict(m.model_options)
    to = dict(mo.get("transformer_options") or {})
    install_into(to)
    mo["transformer_options"] = to
    m.model_options = mo

    # Re-verify before every sampling run (mirrors BlockSparseAttention, which
    # re-installs its override from ON_PREPARE_STATE).
    def _prepare(model_patcher_, timestep, model_options):
        try:
            install_into(model_options.setdefault("transformer_options", {}))
        except Exception:  # pragma: no cover - defensive
            logger.debug("h3-attention: ON_PREPARE_STATE reinstall failed", exc_info=True)

    m.add_callback_with_key(prepare_key, "h3_attention", _prepare)

    controller._native_installed_on = m
    logger.info(
        "h3-attention: installed native block patches on %d blocks (max_seq=%d, mechanisms=%s)",
        len(blocks),
        max_seq,
        controller.config.active_mechanisms,
    )
    return m, True
