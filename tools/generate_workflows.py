#!/usr/bin/env python3
"""Generate the h3-attention example workflows.

Writes LiteGraph JSON in ComfyUI's workflow format (version 0.4):
6-element links ``[id, src_id, src_slot, tgt_id, tgt_slot, type]``,
socket inputs only (widgets live exclusively in ``widgets_values``),
populated ``outputs[].links`` (``null`` when unused).

Node/widget schemas verified against:
- ComfyUI core (UNETLoader, CLIPLoader, VAELoader, LoadImage, MiniMaxH3ReferenceToVideo,
  BasicGuider, BasicScheduler, RandomNoise, KSamplerSelect, SamplerCustomAdvanced,
  VAEDecode, VAEDecodeAudio, CreateVideo, SaveVideo, PreviewImage, BlockSparseAttention)
- kijai/ComfyUI-KJNodes (PathchSageAttentionKJ, ModelPreviewOverrideKJ)
- xmarre/ComfyUI-Spectrum-MiniMax-H3 (SpectrumApplyMiniMaxH3)
- this repo (H3AttentionController/Settings/Debug)

Run:  python tools/generate_workflows.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "workflows"

UNET_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

SPECTRUM_W = [True, 0.5, 4, 0.1, 2, 0.75, 5, 1, 8, False, "system_ram"]
SAGE_W = ["auto", False]
# BlockSparseAttention (core, io.DynamicCombo): selection+tau, start, end,
# dense_blocks, min_tokens, extra_tokens, sink_conditioning, verbose == 9 widgets.
SPARSE_W = ["sol-attn", 1.3, 0.2, 1.0, "", 12288, 256, "exact_kv_and_rows", False]
PREVIEW_OVR_W = [1024, 80, True, 1, 12, "none"]

KINDS = {
    "maximum_coherence": dict(
        out_name="h3_attention_maximum_coherence.json",
        settings_w=[
            "maximum_coherence",
            0.4,
            "additive",
            0.5,
            0.97,
            "global",
            0.3,
            0.35,
            7,
            "smooth",
            0.3,
            0.4,
            0.2,
            True,
            False,
            0.4,
            0.08,
            False,
            10,
        ],
        debug_path="h3_attention_metrics.json",
        save_prefix="video/h3_attention_test",
        ref_images=["refs/mikhael_ref.png", "refs/kingpin_ref.png", "refs/vesper_ref.png"],
        prompt=(
            "subject_definitions:\n"
            "<Picture 1> is Mikhael, tall male with silver hair, black tactical vest, "
            "dark cargo pants, combat boots.\n"
            "<Picture 2> is Kingpin, massive bald male in white suit, gold accessories, cane.\n"
            "<Picture 3> is Vesper, athletic female with red hair in ponytail, leather jacket, jeans.\n"
            "\n"
            "summary:\n"
            "Three characters dance duranguense together in a cantina. Mikhael leads with "
            "precise footwork, Kingpin moves with surprising grace despite size, Vesper spins "
            "between them. All maintain exact clothing and identity from references. Camera "
            "circles the trio at eye level. Energetic norteno music with accordion and bajo sexto."
        ),
    ),
    "single_character": dict(
        out_name="h3_attention_single_character.json",
        settings_w=[
            "identity_preservation",
            0.25,
            "additive",
            0.5,
            0.97,
            "global",
            0.2,
            0.25,
            5,
            "smooth",
            0.1,
            0.3,
            0.15,
            True,
            False,
            0.4,
            0.08,
            False,
            10,
        ],
        debug_path="h3_attention_single_char_metrics.json",
        save_prefix="video/h3_attention_single_char",
        ref_images=["refs/mikhael_ref.png"],
        prompt=(
            "subject_definitions:\n"
            "<Picture 1> is Mikhael, tall male with silver hair, black tactical vest, "
            "dark cargo pants, combat boots.\n"
            "\n"
            "summary:\n"
            "Mikhael performs a solo duranguense dance in a cantina. Precise footwork, "
            "confident posture, silver hair catching the light. Camera circles at eye level. "
            "Energetic norteno music with accordion and bajo sexto."
        ),
    ),
}


class Graph:
    """Builds one workflow: nodes with socket inputs/outputs, then 6-elem links."""

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._by_id: dict[int, dict] = {}
        self._next_node = 1

    def node(self, type_, pos, size, inputs, outputs, widgets=None, title=None):
        nid = self._next_node
        self._next_node += 1
        node = {
            "id": nid,
            "type": type_,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": 0,
            "mode": 0,
            # fill link fields after all links are known
            "inputs": [dict(i, link=None) for i in inputs],
            "outputs": [dict(o, links=None) for o in outputs],
            "properties": {"Node name for S&R": type_},
            "widgets_values": list(widgets) if widgets is not None else [],
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        self._by_id[nid] = node
        return nid

    def link(self, src, src_slot, tgt, tgt_slot, type_):
        lid = len(self.links) + 1
        self.links.append([lid, src, src_slot, tgt, tgt_slot, type_])
        return lid

    def _resolve(self):
        for lid, src, ss, tgt, ts, type_ in self.links:
            out = self._by_id[src]["outputs"][ss]
            inp = self._by_id[tgt]["inputs"][ts]
            assert out["type"] == type_ == inp["type"], (
                f"link {lid}: type mismatch {out['type']} / {type_} / {inp['type']}"
            )
            out["links"] = (out["links"] or []) + [lid]
            inp["link"] = lid

    def _orders(self):
        # Kahn topological sort, stable by insertion id
        indeg = {n["id"]: 0 for n in self.nodes}
        adj: dict[int, list[int]] = {n["id"]: [] for n in self.nodes}
        for _, src, _, tgt, _, _ in self.links:
            adj[src].append(tgt)
            indeg[tgt] += 1
        ready = sorted(i for i, d in indeg.items() if d == 0)
        order = 0
        while ready:
            nid = ready.pop(0)
            self._by_id[nid]["order"] = order
            order += 1
            for nxt in adj[nid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()
        assert order == len(self.nodes), "graph has a cycle"

    def render(self, slug):
        self._resolve()
        self._orders()
        return {
            "id": slug,
            "revision": 0,
            "last_node_id": max(n["id"] for n in self.nodes),
            "last_link_id": max(l[0] for l in self.links),
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }


def _sock(name, type_, shape=None):
    s = {"name": name, "type": type_}
    if shape is not None:
        s["shape"] = shape
    return s


def build(kind: str) -> dict:
    cfg = KINDS[kind]
    g = Graph()

    # ---- loaders -----------------------------------------------------------
    unet = g.node(
        "UNETLoader", (0, 0), (315, 160), [], [_sock("MODEL", "MODEL")], [UNET_NAME, "default"]
    )
    clip = g.node(
        "CLIPLoader",
        (0, 300),
        (315, 180),
        [],
        [_sock("CLIP", "CLIP")],
        [CLIP_NAME, "minimax", "default"],
    )
    vaev = g.node("VAELoader", (0, 600), (315, 110), [], [_sock("VAE", "VAE")], [VIDEO_VAE])
    vaea = g.node("VAELoader", (0, 850), (315, 110), [], [_sock("VAE", "VAE")], [AUDIO_VAE])

    # ---- acceleration chain (UNET -> Spectrum -> Sage -> Sol-Attn -> preview)
    spectrum = g.node(
        "SpectrumApplyMiniMaxH3",
        (380, 0),
        (315, 300),
        [_sock("model", "MODEL")],
        [_sock("MODEL", "MODEL")],
        SPECTRUM_W,
    )
    sage = g.node(
        "PathchSageAttentionKJ",
        (740, 0),
        (315, 140),
        [_sock("model", "MODEL")],
        [_sock("MODEL", "MODEL")],
        SAGE_W,
    )
    sparse = g.node(
        "BlockSparseAttention",
        (1100, 0),
        (315, 340),
        [_sock("model", "MODEL")],
        [_sock("MODEL", "MODEL")],
        SPARSE_W,
    )
    povr = g.node(
        "ModelPreviewOverrideKJ",
        (1460, 0),
        (315, 320),
        [_sock("model", "MODEL"), _sock("vae", "VAE", 7), _sock("audio_vae", "VAE", 7)],
        [_sock("MODEL", "MODEL")],
        PREVIEW_OVR_W,
    )

    # ---- h3-attention nodes ------------------------------------------------
    settings = g.node(
        "H3AttentionSettings",
        (1460, 400),
        (330, 580),
        [],
        [_sock("settings", "H3_ATTENTION_SETTINGS")],
        cfg["settings_w"],
    )
    debug = g.node(
        "H3AttentionDebug",
        (1840, 400),
        (330, 200),
        [_sock("controller", "H3_ATTENTION_CONTROLLER")],
        [_sock("debug_node", "H3_ATTENTION_DEBUG"), _sock("metrics_summary", "STRING")],
        [True, True, cfg["debug_path"]],
    )
    controller = g.node(
        "H3AttentionController",
        (1840, 0),
        (330, 230),
        [
            _sock("model", "MODEL"),
            _sock("settings", "H3_ATTENTION_SETTINGS"),
            _sock("debug_node", "H3_ATTENTION_DEBUG"),
        ],
        [_sock("model", "MODEL"), _sock("controller", "H3_ATTENTION_CONTROLLER")],
        ["custom", "balanced"],
    )

    # ---- reference images --------------------------------------------------
    ref_ids = []
    for i, fname in enumerate(cfg["ref_images"]):
        y = 1400 + 260 * i
        ref_ids.append(
            g.node(
                "LoadImage",
                (0, y),
                (315, 150),
                [],
                [_sock("IMAGE", "IMAGE"), _sock("MASK", "MASK")],
                [fname, "image"],
            )
        )

    # ---- conditioning + latent --------------------------------------------
    ref_inputs = [
        _sock("clip", "CLIP"),
        _sock("vae", "VAE", 7),
        _sock("audio_vae", "VAE", 7),
    ]
    for i in range(len(ref_ids)):
        ref_inputs.append(_sock(f"ref_images.ref_image_{i}", "IMAGE", 7))
        ref_inputs[-1]["label"] = f"ref_image_{i}"
    ref2v = g.node(
        "MiniMaxH3ReferenceToVideo",
        (420, 1400),
        (370, 360),
        ref_inputs,
        [_sock("CONDITIONING", "CONDITIONING"), _sock("LATENT", "LATENT")],
        [cfg["prompt"], 1344, 768, 124, "match"],
    )

    # ---- sampling ----------------------------------------------------------
    guider = g.node(
        "BasicGuider",
        (2260, 0),
        (315, 130),
        [_sock("model", "MODEL"), _sock("conditioning", "CONDITIONING")],
        [_sock("GUIDER", "GUIDER")],
    )
    sched = g.node(
        "BasicScheduler",
        (2260, 200),
        (315, 160),
        [_sock("model", "MODEL")],
        [_sock("SIGMAS", "SIGMAS")],
        ["beta", 20, 1.0],
    )
    noise = g.node(
        "RandomNoise", (2260, 430), (315, 140), [], [_sock("NOISE", "NOISE")], [42, "fixed"]
    )
    select = g.node(
        "KSamplerSelect",
        (2260, 640),
        (315, 100),
        [],
        [_sock("SAMPLER", "SAMPLER")],
        ["res_multistep"],
    )
    sca = g.node(
        "SamplerCustomAdvanced",
        (2660, 200),
        (330, 180),
        [
            _sock("noise", "NOISE"),
            _sock("guider", "GUIDER"),
            _sock("sampler", "SAMPLER"),
            _sock("sigmas", "SIGMAS"),
            _sock("latent_image", "LATENT"),
        ],
        [_sock("output", "LATENT"), _sock("denoised_output", "LATENT")],
    )

    # ---- decode / save -----------------------------------------------------
    decode = g.node(
        "VAEDecode",
        (3060, 0),
        (315, 110),
        [_sock("samples", "LATENT"), _sock("vae", "VAE")],
        [_sock("IMAGE", "IMAGE")],
    )
    decode_a = g.node(
        "VAEDecodeAudio",
        (3060, 260),
        (315, 110),
        [_sock("samples", "LATENT"), _sock("vae", "VAE")],
        [_sock("AUDIO", "AUDIO")],
    )
    create = g.node(
        "CreateVideo",
        (3440, 100),
        (315, 250),
        [_sock("images", "IMAGE"), _sock("audio", "AUDIO", 7)],
        [_sock("VIDEO", "VIDEO")],
        [24, "auto", "sRGB", "none"],
    )
    save = g.node(
        "SaveVideo",
        (3820, 100),
        (330, 290),
        [_sock("video", "VIDEO")],
        [_sock("video", "VIDEO")],
        [cfg["save_prefix"], "auto", "auto", "auto"],
    )
    prev = g.node("PreviewImage", (3060, 540), (315, 280), [_sock("images", "IMAGE")], [])

    # ---- links -------------------------------------------------------------
    g.link(unet, 0, spectrum, 0, "MODEL")
    g.link(spectrum, 0, sage, 0, "MODEL")
    g.link(sage, 0, sparse, 0, "MODEL")
    g.link(sparse, 0, povr, 0, "MODEL")
    g.link(povr, 0, controller, 0, "MODEL")
    g.link(vaev, 0, povr, 1, "VAE")
    g.link(settings, 0, controller, 1, "H3_ATTENTION_SETTINGS")
    g.link(debug, 0, controller, 2, "H3_ATTENTION_DEBUG")
    # debug.controller stays unconnected (avoids a cycle: Debug is OUTPUT_NODE)

    g.link(clip, 0, ref2v, 0, "CLIP")
    g.link(vaev, 0, ref2v, 1, "VAE")
    g.link(vaea, 0, ref2v, 2, "VAE")
    for i, lid in enumerate(ref_ids):
        g.link(lid, 0, ref2v, 3 + i, "IMAGE")

    g.link(controller, 0, guider, 0, "MODEL")
    g.link(ref2v, 0, guider, 1, "CONDITIONING")
    g.link(controller, 0, sched, 0, "MODEL")

    g.link(noise, 0, sca, 0, "NOISE")
    g.link(guider, 0, sca, 1, "GUIDER")
    g.link(select, 0, sca, 2, "SAMPLER")
    g.link(sched, 0, sca, 3, "SIGMAS")
    g.link(ref2v, 1, sca, 4, "LATENT")

    g.link(sca, 0, decode, 0, "LATENT")
    g.link(sca, 0, decode_a, 0, "LATENT")
    g.link(vaev, 0, decode, 1, "VAE")
    g.link(vaea, 0, decode_a, 1, "VAE")

    g.link(decode, 0, create, 0, "IMAGE")
    g.link(decode_a, 0, create, 1, "AUDIO")
    g.link(create, 0, save, 0, "VIDEO")
    g.link(decode, 0, prev, 0, "IMAGE")

    slug = f"h3-attention-{kind.replace('_', '-')}"
    return g.render(slug)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for kind in KINDS:
        wf = build(kind)
        path = OUT_DIR / KINDS[kind]["out_name"]
        path.write_text(json.dumps(wf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"wrote {path.relative_to(ROOT)} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)"
        )


if __name__ == "__main__":
    main()
