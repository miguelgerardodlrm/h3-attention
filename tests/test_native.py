"""
End-to-end tests for the native ComfyUI block-patch path.

These run WITHOUT ComfyUI installed: they reproduce the exact contract of
``comfy/ldm/minimax/model.py`` (block args, ``extra['original_block']``,
``attention`` injection) with small fake modules, and verify:

* chain-patching wraps an existing patch (e.g. BlockSparseAttention) without
  nesting on reinstall;
* active control takes priority over the previous patch's attention;
* disabled control delegates to the previous patch (acceleration intact);
* fail-safe behaviour (no ``.blocks`` -> unchanged model, attention errors ->
  native fallback);
* bias semantics of the 7 mechanisms;
* step/schedule update from sampler sigmas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_attention import (
    AttentionControlConfig,
    H3AttentionController,
    SCHEDULE_PRESETS,
    install_comfyui,
)
from h3_attention.native import NativeBiasPlanner, MAX_CONTROLLED_SEQ


# ---------------------------------------------------------------------------
# Fakes mirroring the ComfyUI / MiniMax H3 contracts
# ---------------------------------------------------------------------------


class FakeLayout:
    """text(0:5) refA(5:15) refB(15:25) ref_audio(25:30) audio(30:40) video(40:100)"""

    def __init__(self, seq_len: int = 100):
        self.segments = [
            (0, 5, "text"),
            (5, 15, "ref_img"),
            (15, 25, "ref_img"),
            (25, 30, "ref_audio"),
            (30, 40, "audio"),
            (40, seq_len, "video"),
        ]
        self.seq_len = seq_len
        # (text_len, latent_t, latent_h, latent_w, audio_t)
        self.signature = (5, 6, 2, 2, 5)
        pos = torch.zeros(seq_len, 3, dtype=torch.float64)
        # video: 60 rows / 6 frames = 10 rows per frame; t advances per frame
        vpf = (seq_len - 40) // 6
        for i, (a, b, kind) in enumerate(self.segments):
            if kind == "video":
                for r in range(a, b):
                    pos[r, 0] = ((r - a) // vpf) * 1.6
            elif kind == "audio":
                for r in range(a, b):
                    pos[r, 0] = (r - a) // 2
        self.position_ids = pos


class FakeAttn(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4, head_dim: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden, heads * head_dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.out_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.native_calls = 0

    def forward(self, x, rope_freqs=None, transformer_options={}):
        self.native_calls += 1
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
        v = v.view(s, self.heads, self.head_dim)
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = k.transpose(0, 1).unsqueeze(0)
        vh = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        return self.out_proj(out.squeeze(0).transpose(0, 1).reshape(s, -1))


class FakeBlock:
    def __init__(self):
        self.attn = FakeAttn()

    def __call__(
        self, img, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None
    ):
        attention = self.attn if attention is None else attention
        return attention(img, rope_freqs=rope_freqs, transformer_options=transformer_options)


class FakeDiffusion(nn.Module):
    def __init__(self, n_blocks: int = 2, with_blocks: bool = True):
        super().__init__()
        if with_blocks:
            self.blocks = [FakeBlock() for _ in range(n_blocks)]


class FakeModelPatcher:
    def __init__(self, diffusion, model_options=None):
        self._diffusion = diffusion
        self.model_options = (
            model_options if model_options is not None else {"transformer_options": {}}
        )
        self.callbacks = {}

    def clone(self):
        c = FakeModelPatcher(self._diffusion, dict(self.model_options))
        c.callbacks = {k: dict(v) for k, v in self.callbacks.items()}
        return c

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self._diffusion
        raise KeyError(name)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {})[key] = callback


def make_controller(**overrides) -> H3AttentionController:
    config = AttentionControlConfig(**overrides)
    return H3AttentionController(config=config, schedule=None)


def dit_patches(patcher):
    return patcher.model_options["transformer_options"]["patches_replace"]["dit"]


def run_block(patcher, controller, block_index: int = 0, seq_len: int = 100, bad_layout=False):
    """Mimics comfy/ldm/minimax/model.py `_forward` for one block."""
    layout = FakeLayout(seq_len=seq_len)
    if bad_layout:
        _sl = seq_len

        class Broken:
            seq_len = _sl

            @property
            def segments(self):
                raise RuntimeError("boom")

            position_ids = layout.position_ids
            signature = layout.signature

        layout = Broken()
    blocks = patcher._diffusion.blocks
    dit = patcher.model_options["transformer_options"]["patches_replace"]["dit"]
    key = ("double_block", block_index)
    assert key in dit, "expected a patch for this block"
    block = blocks[block_index]
    topts = {
        "block_index": block_index,
        "sigmas": torch.tensor(0.75),
        "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
    }

    def block_wrap(args):
        return {
            "img": block(
                args["img"],
                t_emb=None,
                mod_segments=None,
                rope_freqs=args["rope_freqs"],
                transformer_options=args["transformer_options"],
                attention=args.get("attention"),
            )
        }

    args = {
        "img": torch.randn(seq_len, 32),
        "t_emb": None,
        "mod_segments": None,
        "rope_freqs": None,
        "layout": layout,
        "transformer_options": topts,
    }
    out = dit[key](args, {"original_block": block_wrap})
    return out, topts, block


def install_with_prev(controller, prev=None, n_blocks=2, max_seq=MAX_CONTROLLED_SEQ):
    diff = FakeDiffusion(n_blocks=n_blocks)
    patcher = FakeModelPatcher(diff)
    if prev is not None:
        tp = patcher.model_options["transformer_options"]
        tp["patches_replace"] = {"dit": {("double_block", i): prev for i in range(n_blocks)}}
    patched, ok = install_comfyui(patcher, controller, max_seq=max_seq)
    assert ok
    return patched, diff


# ---------------------------------------------------------------------------
# Installation / chaining
# ---------------------------------------------------------------------------


def test_install_wraps_existing_patch():
    prev_calls = {"n": 0}

    def prev_patch(args, extra):
        prev_calls["n"] += 1
        return extra["original_block"](args)

    controller = make_controller(reference_strength=0.4)
    patched, _ = install_with_prev(controller, prev=prev_patch)
    p0 = dit_patches(patched)[("double_block", 0)]
    assert getattr(p0, "_h3_owner", False) is True
    assert getattr(p0, "_h3_prev", None) is prev_patch
    run_block(patched, controller)
    assert prev_calls["n"] == 1  # previous patch still executed (chained)


def test_reinstall_does_not_nest():
    controller = make_controller(reference_strength=0.4)

    def prev_patch(args, extra):
        return extra["original_block"](args)

    patched, _ = install_with_prev(controller, prev=prev_patch)
    # Second install on the already-patched model (e.g. re-executed graph)
    patched2, ok = install_comfyui(patched, controller)
    assert ok
    p0 = dit_patches(patched2)[("double_block", 0)]
    assert getattr(p0, "_h3_owner", False) is True
    assert getattr(p0, "_h3_prev", None) is prev_patch  # unwrapped, not nested


def test_prepare_state_callback_keeps_single_layer():
    controller = make_controller(reference_strength=0.4)
    patched, _ = install_with_prev(controller)
    cb = patched.callbacks["on_prepare_state"]["h3_attention"]
    cb(patched, torch.tensor(0.5), patched.model_options)
    p0 = dit_patches(patched)[("double_block", 0)]
    assert getattr(p0, "_h3_owner", False) is True
    assert not getattr(getattr(p0, "_h3_prev", None), "_h3_owner", False)


def test_install_without_blocks_is_fail_safe():
    diff = FakeDiffusion(with_blocks=False)
    patcher = FakeModelPatcher(diff)
    controller = make_controller(reference_strength=0.4)
    out, ok = install_comfyui(patcher, controller)
    assert ok is False
    assert out is not patcher  # still a clone, but with no patches installed
    assert "patches_replace" not in out.model_options.get("transformer_options", {})


# ---------------------------------------------------------------------------
# Control priority / delegation
# ---------------------------------------------------------------------------


def test_active_control_overrides_prev_attention():
    seen = {}

    def prev_patch(args, extra):
        def prev_attention(h, rope_freqs=None, transformer_options={}):
            seen["prev_used"] = True
            return h

        return extra["original_block"]({**args, "attention": prev_attention})

    controller = make_controller(reference_strength=0.4)
    patched, _ = install_with_prev(controller, prev=prev_patch)
    out, topts, block = run_block(patched, controller)
    # Controlled path ran: it never calls ``attn.forward`` (the native path would).
    assert block.attn.native_calls == 0
    # The previous patch's attention was overridden by the controlled one.
    assert seen.get("prev_used") is None
    assert out["img"].shape == (100, 32)
    # Step plumbing fired from sigmas.
    assert controller.state.total_steps == 2


def test_disabled_control_delegates_to_prev_attention():
    seen = {}

    def prev_patch(args, extra):
        def prev_attention(h, rope_freqs=None, transformer_options={}):
            seen["prev_used"] = True
            return h + 1.0

        return extra["original_block"]({**args, "attention": prev_attention})

    controller = make_controller(reference_strength=0.0)  # everything off
    assert controller.config.is_active is False
    patched, _ = install_with_prev(controller, prev=prev_patch)
    out, _, block = run_block(patched, controller)
    assert seen.get("prev_used") is True  # acceleration patch ran its attention
    assert out["img"].shape == (100, 32)


def test_gate_delegates_when_seq_too_long():
    seen = {}

    def prev_patch(args, extra):
        def prev_attention(h, rope_freqs=None, transformer_options={}):
            seen["prev_used"] = True
            return h

        return extra["original_block"]({**args, "attention": prev_attention})

    controller = make_controller(reference_strength=0.4)
    patched, _ = install_with_prev(controller, prev=prev_patch, max_seq=50)  # seq 100 > 50
    run_block(patched, controller, seq_len=100)
    assert seen.get("prev_used") is True
    assert getattr(controller, "_native_gate_logged", False) is True


def test_attention_error_falls_back_to_native():
    controller = make_controller(reference_strength=0.4)
    patched, diff = install_with_prev(controller)
    native_calls_before = diff.blocks[0].attn.native_calls
    # Broken layout -> planner construction raises inside controlled path
    out, _, block = run_block(patched, controller, bad_layout=True)
    assert out["img"].shape == (100, 32)
    assert block.attn.native_calls > native_calls_before  # native fallback used


# ---------------------------------------------------------------------------
# Step / schedule plumbing
# ---------------------------------------------------------------------------


def test_step_update_from_sigmas_and_schedule():
    schedule = SCHEDULE_PRESETS["balanced"]
    config = AttentionControlConfig(reference_strength=0.9, identity_lock=0.9)
    controller = H3AttentionController(config=config, schedule=schedule)
    patched, _ = install_with_prev(controller)
    _, topts, _ = run_block(patched, controller)  # block_index=0 triggers update
    # sample_sigmas [1.0, 0.5, 0.0], current 0.75 -> nearest index 0 or 1; total=2
    assert controller.state.total_steps == 2
    assert controller.state.current_timestep == 0.75
    # balanced schedule EARLY phase replaces strengths (ref 0.3, identity 0.3)
    assert abs(controller.config.reference_strength - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# Bias semantics (7 mechanisms)
# ---------------------------------------------------------------------------


def _planner(**overrides):
    cfg = AttentionControlConfig(**overrides)
    controller = H3AttentionController(config=cfg, schedule=None)
    layout = FakeLayout()
    planner = NativeBiasPlanner(layout, cfg, controller.state)
    return planner


def test_reference_and_prompt_bias_cells():
    p = _planner(reference_strength=0.5, prompt_adherence=0.3)
    rows = p.rows(40, 50, torch.device("cpu"), torch.float32)  # video rows
    # video -> refA (cols 5:15) += 0.5 ; refA -> video += 0.25
    assert torch.allclose(rows[:, 5:15], torch.full((10, 10), 0.5))
    # video -> text (0:5) += 0.3
    assert torch.allclose(rows[:, 0:5], torch.full((10, 5), 0.3))
    # audio rows -> text += 0.3 * 0.8
    arows = p.rows(30, 35, torch.device("cpu"), torch.float32)
    assert torch.allclose(arows[:, 0:5], torch.full((5, 5), 0.24), atol=1e-6)
    # refA -> video (cols 40:) += 0.5 * 0.5 = 0.25
    rrows = p.rows(5, 10, torch.device("cpu"), torch.float32)
    assert torch.allclose(rrows[:, 40:], torch.full((5, 60), 0.25))


def test_reference_multiplicative_mode_is_inert():
    p = _planner(reference_strength=0.5, reference_boost_mode="multiplicative")
    rows = p.rows(40, 45, torch.device("cpu"), torch.float32)
    assert float(rows[:, 5:15].abs().max()) == 0.0


def test_subject_separation_penalizes_cross_ref():
    p = _planner(subject_separation=0.2, reference_strength=0.0)
    rows = p.rows(5, 10, torch.device("cpu"), torch.float32)  # refA queries
    assert torch.allclose(rows[:, 15:25], torch.full((5, 10), -0.2))
    # own segment untouched
    assert float(rows[:, 5:15].abs().max()) == 0.0


def test_identity_global_includes_anchor_boost():
    p = _planner(identity_lock=0.4, identity_temporal_decay=0.95, reference_strength=0.5)
    rows = p.rows(40, 45, torch.device("cpu"), torch.float32)
    # anchor is refA (first ref_img): ref boost 0.5 + identity 0.4 * 0.95^0
    assert torch.allclose(rows[:, 5:15], torch.full((5, 10), 0.9), atol=1e-5)


def test_identity_local_decays_with_time():
    p = _planner(
        identity_lock=0.4,
        identity_temporal_decay=0.95,
        identity_spatial_scope="local",
        reference_strength=0.0,
    )
    rows = p.rows(40, 100, torch.device("cpu"), torch.float32)
    # local scope: weight 0.9^|t_q - t_anchor| ; anchor t = 0, video t grows per frame
    assert float(rows[0, 5:15].mean()) > float(rows[-1, 5:15].mean())
    assert float(rows[:, 5:15].max()) <= 0.4 + 1e-5


def test_temporal_video_window_in_frames():
    p = _planner(temporal_lock=0.4, temporal_window=2, temporal_mode="smooth")
    rows = p.rows(40, 100, torch.device("cpu"), torch.float32)  # all video rows
    # rows 40..49 are frame 0; rows 50..59 frame 1 -> d=1 -> 0.4*0.8
    assert torch.allclose(rows[0:10, 50:60], torch.full((10, 10), 0.4 * 0.8), atol=1e-5)
    # frame distance 4 (rows 40..49 vs 80..89) > window 2 -> 0
    assert float(rows[0:10, 80:90].abs().max()) == 0.0
    # same frame (d=0) excluded
    assert float(rows[0:10, 40:50].abs().max()) == 0.0


def test_temporal_strict_is_constant():
    p = _planner(temporal_lock=0.4, temporal_window=1, temporal_mode="strict")
    rows = p.rows(40, 50, torch.device("cpu"), torch.float32)
    assert torch.allclose(rows[0:10, 50:60], torch.full((10, 10), 0.4), atol=1e-5)


def test_audio_video_sync_pairs_frames():
    p = _planner(audio_strength=0.5, audio_video_sync=True)
    # audio rows 30..39 (2 rows/frame => frames 0,0,1,1,2,2,3,3,4,4)
    rows = p.rows(30, 40, torch.device("cpu"), torch.float32)
    # audio frames 0/1 map near video frames 0/1 -> video frame 0 columns match
    assert torch.allclose(rows[0:4, 40:50], torch.full((4, 10), 0.5), atol=1e-5)
    # audio frame 2 maps to video ~frame 2 -> no match with video frame 0
    assert float(rows[4, 40:50].abs().max()) == 0.0
    # audio -> ref_audio += 0.7 * 0.5 (from mechanism 6)
    assert torch.allclose(rows[:, 25:30], torch.full((10, 5), 0.35), atol=1e-5)


def test_bias_scale_and_clamp():
    p = _planner(reference_strength=0.5, attention_bias_scale=2.0, bias_clamp_value=0.4)
    rows = p.rows(40, 45, torch.device("cpu"), torch.float32)
    # 0.5 * 2.0 = 1.0 clamped to 0.4 ; text 0.3*2=0.6 -> 0.4 (clamped too)
    assert torch.allclose(rows[:, 5:15], torch.full((5, 10), 0.4), atol=1e-6)
    assert float(rows.abs().max()) <= 0.4 + 1e-6


def test_planner_cache_changes_with_step():
    cfg = AttentionControlConfig(identity_lock=0.5, identity_temporal_decay=0.9)
    controller = H3AttentionController(config=cfg, schedule=None)
    layout = FakeLayout()
    from h3_attention.native import get_planner

    p1 = get_planner(controller, layout)
    k1 = p1.cache_key(controller)
    controller.state.step = 10
    controller.state.total_steps = 30
    p2 = get_planner(controller, layout)
    assert p2 is not p1  # identity decay depends on step -> rebuild
    assert p1.cache_key(controller) != k1 or p2.cache_key(controller) != k1
