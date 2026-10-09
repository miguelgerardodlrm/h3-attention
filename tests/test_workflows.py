"""Structural validation of the example workflows (example_workflows/*.json).

Checks the LiteGraph serialization format, link/slot consistency, node
whitelist, acyclicity, and the h3-attention-specific wiring (Settings and
Debug connected to the Controller, acceleration chain, sampling chain).
"""

import json
from pathlib import Path

import pytest

from h3_attention import PRESETS

WF_DIR = Path(__file__).resolve().parent.parent / "example_workflows"

WORKFLOWS = {
    "maximum_coherence": WF_DIR / "h3_attention_maximum_coherence.json",
    "single_character": WF_DIR / "h3_attention_single_character.json",
}

# Every node type the workflows may use (core ComfyUI, KJNodes, Spectrum-MiniMax-H3,
# this repo). Anything else would be a nonexistent node on a correctly set-up install.
ALLOWED_TYPES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "SpectrumApplyMiniMaxH3",
    "PathchSageAttentionKJ",
    "BlockSparseAttention",
    "ModelPreviewOverrideKJ",
    "H3AttentionSettings",
    "H3AttentionDebug",
    "H3AttentionController",
    "LoadImage",
    "MiniMaxH3ReferenceToVideo",
    "BasicGuider",
    "BasicScheduler",
    "RandomNoise",
    "KSamplerSelect",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
    "PreviewImage",
}

# Node types that must never appear: LoRA loaders and negative-conditioning paths.
# (The whitelist above already excludes these; this is a belt-and-braces check.)
FORBIDDEN_TYPES = {
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ConditioningZeroOut",
    "CLIPTextEncode",
    "KSampler",
}


def load(kind):
    return json.loads(WORKFLOWS[kind].read_text(encoding="utf-8"))


def nodes_by_id(wf):
    return {n["id"]: n for n in wf["nodes"]}


def by_type(wf, type_):
    found = [n for n in wf["nodes"] if n["type"] == type_]
    assert found, f"missing required node {type_}"
    return found


def sole(wf, type_):
    found = by_type(wf, type_)
    assert len(found) == 1, f"expected exactly one {type_}, got {len(found)}"
    return found[0]


@pytest.fixture(params=sorted(WORKFLOWS))
def wf(request):
    return load(request.param), request.param


class TestTemplateFolder:
    """ComfyUI's template browser indexes example_workflows/ (single level, JSON only).

    The index endpoint globs custom_nodes/*/<folder>/*.json, so extra files or
    nested directories here would either never show up or show up broken.
    """

    def test_single_level_json_only(self):
        assert WF_DIR.is_dir(), f"canonical template folder missing: {WF_DIR}"
        for path in WF_DIR.iterdir():
            assert path.is_file(), f"nested entries unsupported by ComfyUI: {path.name}"
            assert path.suffix in {".json", ".jpg"}, f"unexpected file: {path.name}"

    def test_exactly_the_two_templates(self):
        found = sorted(p.name for p in WF_DIR.glob("*.json"))
        assert found == [
            "h3_attention_maximum_coherence.json",
            "h3_attention_single_character.json",
        ]


class TestFormat:
    def test_top_level_shape(self, wf):
        wf, _ = wf
        for key in (
            "id",
            "revision",
            "last_node_id",
            "last_link_id",
            "nodes",
            "links",
            "groups",
            "config",
            "extra",
            "version",
        ):
            assert key in wf, f"missing top-level key {key}"
        assert wf["version"] == 0.4
        assert wf["nodes"] and wf["links"]

    def test_links_are_six_element_tuples(self, wf):
        wf, _ = wf
        for link in wf["links"]:
            assert len(link) == 6, f"link must have 6 elements: {link}"
            lid, src, ss, tgt, ts, type_ = link
            assert all(isinstance(v, int) for v in (lid, src, ss, tgt, ts))
            assert ts is not None, f"target slot must not be null: {link}"
            assert isinstance(type_, str) and type_

    def test_link_ids_unique_and_bounded(self, wf):
        wf, _ = wf
        ids = [l[0] for l in wf["links"]]
        assert len(ids) == len(set(ids))
        assert max(ids) == wf["last_link_id"]

    def test_node_ids_unique_and_bounded(self, wf):
        wf, _ = wf
        ids = [n["id"] for n in wf["nodes"]]
        assert len(ids) == len(set(ids))
        assert max(ids) == wf["last_node_id"]

    def test_node_layout_fields(self, wf):
        wf, _ = wf
        orders = []
        for n in wf["nodes"]:
            assert isinstance(n["pos"], list) and len(n["pos"]) == 2
            assert isinstance(n["size"], list) and len(n["size"]) == 2
            assert n["mode"] == 0
            orders.append(n["order"])
        assert sorted(orders) == list(range(len(wf["nodes"])))


class TestLinks:
    def test_endpoints_exist_and_slots_in_range(self, wf):
        wf, _ = wf
        by_id = nodes_by_id(wf)
        for lid, src, ss, tgt, ts, type_ in wf["links"]:
            assert src in by_id, f"link {lid}: unknown src {src}"
            assert tgt in by_id, f"link {lid}: unknown tgt {tgt}"
            src_n, tgt_n = by_id[src], by_id[tgt]
            assert ss < len(src_n["outputs"]), f"link {lid}: src slot OOB"
            assert ts < len(tgt_n["inputs"]), f"link {lid}: tgt slot OOB"

    def test_types_match_on_both_ends(self, wf):
        wf, _ = wf
        by_id = nodes_by_id(wf)
        for lid, src, ss, tgt, ts, type_ in wf["links"]:
            out_t = by_id[src]["outputs"][ss]["type"]
            in_t = by_id[tgt]["inputs"][ts]["type"]
            assert out_t == type_ == in_t, (
                f"link {lid}: {by_id[src]['type']}.{out_t} -> "
                f"{by_id[tgt]['type']}.{in_t} declared {type_}"
            )

    def test_node_fields_agree_with_links(self, wf):
        wf, _ = wf
        by_id = nodes_by_id(wf)
        for lid, src, ss, tgt, ts, type_ in wf["links"]:
            src_n, tgt_n = by_id[src], by_id[tgt]
            out_links = src_n["outputs"][ss].get("links") or []
            assert lid in out_links, f"link {lid} missing from src output"
            assert tgt_n["inputs"][ts]["link"] == lid, f"link {lid} not recorded on tgt input"

    def test_no_dangling_input_links(self, wf):
        wf, _ = wf
        valid = {l[0] for l in wf["links"]}
        for n in wf["nodes"]:
            for i, inp in enumerate(n["inputs"]):
                link = inp.get("link")
                if link is not None:
                    assert link in valid, f"{n['type']} input {i} references missing link {link}"

    def test_outputs_links_lists_reference_valid_links(self, wf):
        wf, _ = wf
        valid = {l[0] for l in wf["links"]}
        for n in wf["nodes"]:
            for o, out in enumerate(n["outputs"]):
                for link in out.get("links") or []:
                    assert link in valid, f"{n['type']} output {o} references missing link {link}"


class TestGraph:
    def test_whitelisted_node_types(self, wf):
        wf, kind = wf
        for n in wf["nodes"]:
            assert n["type"] in ALLOWED_TYPES, (
                f"{kind}: nonexistent/unsupported node type {n['type']}"
            )

    def test_no_forbidden_nodes(self, wf):
        wf, _ = wf
        for n in wf["nodes"]:
            assert n["type"] not in FORBIDDEN_TYPES, f"forbidden node {n['type']}"

    def test_acyclic(self, wf):
        wf, _ = wf
        indeg = {n["id"]: 0 for n in wf["nodes"]}
        adj = {n["id"]: [] for n in wf["nodes"]}
        for _, src, _, tgt, _, _ in wf["links"]:
            adj[src].append(tgt)
            indeg[tgt] += 1
        ready = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while ready:
            nid = ready.pop()
            seen += 1
            for nxt in adj[nid]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        assert seen == len(wf["nodes"]), "workflow graph contains a cycle"

    def test_no_orphan_nodes(self, wf):
        wf, _ = wf
        touched = set()
        for _, src, _, tgt, _, _ in wf["links"]:
            touched.add(src)
            touched.add(tgt)
        orphans = [n["type"] for n in wf["nodes"] if n["id"] not in touched]
        assert not orphans, f"unconnected nodes: {orphans}"

    def test_orders_form_topological_sequence(self, wf):
        wf, _ = wf
        by_id = nodes_by_id(wf)
        for _, src, _, tgt, _, _ in wf["links"]:
            assert by_id[src]["order"] < by_id[tgt]["order"], (
                f"order of {by_id[src]['type']} must precede {by_id[tgt]['type']}"
            )


def input_link(node, name):
    for inp in node["inputs"]:
        if inp["name"] == name:
            return inp["link"]
    raise AssertionError(f"{node['type']} has no input {name!r}")


class TestH3AttentionWiring:
    def test_settings_connected_to_controller(self, wf):
        wf, _ = wf
        controller = sole(wf, "H3AttentionController")
        settings = sole(wf, "H3AttentionSettings")
        link = input_link(controller, "settings")
        assert link is not None, "Controller.settings must be connected"
        src = next(l for l in wf["links"] if l[0] == link)
        assert src[1] == settings["id"]

    def test_debug_connected_to_controller_without_cycle(self, wf):
        wf, _ = wf
        controller = sole(wf, "H3AttentionController")
        debug = sole(wf, "H3AttentionDebug")
        link = input_link(controller, "debug_node")
        assert link is not None, "Controller.debug_node must be connected"
        src = next(l for l in wf["links"] if l[0] == link)
        assert src[1] == debug["id"]
        # Debug.controller must stay unconnected (Debug is an OUTPUT_NODE;
        # wiring its controller input back would create a graph cycle).
        assert input_link(debug, "controller") is None, (
            "Debug.controller must not be connected (cycle)"
        )

    def test_controller_widgets(self, wf):
        wf, kind = wf
        controller = sole(wf, "H3AttentionController")
        preset, schedule = controller["widgets_values"]
        assert preset == "custom", (
            f"{kind}: preset must be 'custom' so the Settings node drives the config"
        )
        assert schedule == "balanced"

    def test_settings_widgets(self, wf):
        wf, kind = wf
        settings = sole(wf, "H3AttentionSettings")
        w = settings["widgets_values"]
        assert len(w) == 19, f"{kind}: Settings must serialize 19 widgets, got {len(w)}"
        preset = w[0]
        assert preset in PRESETS, f"unknown Settings preset {preset!r}"
        expected = "maximum_coherence" if kind == "maximum_coherence" else "identity_preservation"
        assert preset == expected, f"{kind}: preset should be {expected}"
        assert w[14] is False, f"{kind}: adaptive_mode must default to False"
        assert w[17] is False, f"{kind}: debug_attention default False"

    def test_debug_widgets(self, wf):
        wf, _ = wf
        debug = sole(wf, "H3AttentionDebug")
        w = debug["widgets_values"]
        assert len(w) == 3
        assert isinstance(w[2], str) and w[2].endswith(".json")

    def test_controller_model_fans_out(self, wf):
        wf, _ = wf
        controller = sole(wf, "H3AttentionController")
        targets = {t for _, src, _, t, _, t_ in wf["links"] if src == controller["id"]}
        guider = sole(wf, "BasicGuider")
        sched = sole(wf, "BasicScheduler")
        assert guider["id"] in targets, "Controller model must feed BasicGuider"
        assert sched["id"] in targets, "Controller model must feed BasicScheduler"


class TestAccelerationChain:
    def test_chain_order(self, wf):
        wf, _ = wf
        order = [
            sole(wf, t)["order"]
            for t in (
                "UNETLoader",
                "SpectrumApplyMiniMaxH3",
                "PathchSageAttentionKJ",
                "BlockSparseAttention",
                "ModelPreviewOverrideKJ",
                "H3AttentionController",
            )
        ]
        assert order == sorted(order), "acceleration chain out of order"
        # and consecutive: each feeds the next
        by_id = nodes_by_id(wf)
        chain = [
            "UNETLoader",
            "SpectrumApplyMiniMaxH3",
            "PathchSageAttentionKJ",
            "BlockSparseAttention",
            "ModelPreviewOverrideKJ",
            "H3AttentionController",
        ]
        for a, b in zip(chain, chain[1:]):
            src = sole(wf, a)
            tgt = sole(wf, b)
            assert input_link(tgt, "model") is not None, f"{b}.model unconnected"
            link = next(l for l in wf["links"] if l[0] == input_link(tgt, "model"))
            assert link[1] == src["id"], f"{b}.model must come from {a}"

    def test_sage_widgets(self, wf):
        wf, _ = wf
        w = sole(wf, "PathchSageAttentionKJ")["widgets_values"]
        assert len(w) == 2
        assert w[0] == "auto"
        assert w[1] is False

    def test_block_sparse_widgets_sol_attn(self, wf):
        wf, _ = wf
        w = sole(wf, "BlockSparseAttention")["widgets_values"]
        assert len(w) == 9, f"expected 9 widgets, got {len(w)}"
        assert w[0] == "sol-attn"
        assert w[1] == 1.3
        assert w[7] == "exact_kv_and_rows"

    def test_preview_override_widgets(self, wf):
        wf, _ = wf
        w = sole(wf, "ModelPreviewOverrideKJ")["widgets_values"]
        assert w == [1024, 80, True, 1, 12, "none"]

    def test_spectrum_widgets(self, wf):
        wf, _ = wf
        w = sole(wf, "SpectrumApplyMiniMaxH3")["widgets_values"]
        assert len(w) == 11
        assert w[-1] == "system_ram"


class TestReferenceAndSampling:
    def test_loaders(self, wf):
        wf, _ = wf
        unet = sole(wf, "UNETLoader")
        assert unet["widgets_values"][0] == ("minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        clip = sole(wf, "CLIPLoader")
        cw = clip["widgets_values"]
        assert len(cw) == 3, "CLIPLoader needs [name, type, device]"
        assert cw[1] == "minimax", "CLIPLoader type must be 'minimax'"

    def test_vae_loaders(self, wf):
        wf, _ = wf
        vaes = by_type(wf, "VAELoader")
        names = sorted(v["widgets_values"][0] for v in vaes)
        assert names == [
            "minimax_h3_audio_vae_fp32.safetensors",
            "minimax_h3_video_vae_fp16.safetensors",
        ]

    def test_reference_images(self, wf):
        wf, kind = wf
        expected = 3 if kind == "maximum_coherence" else 1
        images = by_type(wf, "LoadImage")
        assert len(images) == expected, f"{kind}: expected {expected} reference images"
        ref2v = sole(wf, "MiniMaxH3ReferenceToVideo")
        for i in range(expected):
            name = f"ref_images.ref_image_{i}"
            link = input_link(ref2v, name)
            assert link is not None, f"{name} must be connected"
            src = next(l for l in wf["links"] if l[0] == link)
            assert by_id_type(wf, src[1]) == "LoadImage"
            # 0-based indexing: ref_image_0 exists; ref_image_{n} must not
        with pytest.raises(AssertionError):
            input_link(ref2v, f"ref_images.ref_image_{expected}")

    def test_ref2v_widgets_and_prompt_tags(self, wf):
        wf, kind = wf
        ref2v = sole(wf, "MiniMaxH3ReferenceToVideo")
        w = ref2v["widgets_values"]
        assert len(w) == 5, "ref2v widgets: [prompt, width, height, length, ref_image_size]"
        prompt, width, height, length, size = w
        assert (width, height, length, size) == (1344, 768, 124, "match")
        assert "<Picture 1>" in prompt, "prompt must use 1-based <Picture i> tags"
        n_refs = len(by_type(wf, "LoadImage"))
        if n_refs > 1:
            assert "<Picture 2>" in prompt
        if n_refs > 2:
            assert "<Picture 3>" in prompt
        # no stale tag style
        assert "<Subject" not in prompt

    def test_sampling_chain(self, wf):
        wf, _ = wf
        sca = sole(wf, "SamplerCustomAdvanced")
        ref2v = sole(wf, "MiniMaxH3ReferenceToVideo")
        guider = sole(wf, "BasicGuider")
        # every SamplerCustomAdvanced input connected
        for name in ("noise", "guider", "sampler", "sigmas", "latent_image"):
            assert input_link(sca, name) is not None, f"SCA.{name} unconnected"
        # latent comes straight from ref2v (slot 1), conditioning via BasicGuider
        assert input_link(sca, "latent_image") == next(
            l[0] for l in wf["links"] if l[1] == ref2v["id"] and l[2] == 1
        )
        assert input_link(guider, "conditioning") == next(
            l[0] for l in wf["links"] if l[1] == ref2v["id"] and l[2] == 0
        )
        sched = sole(wf, "BasicScheduler")
        assert sched["widgets_values"] == ["beta", 20, 1.0]
        assert sole(wf, "KSamplerSelect")["widgets_values"] == ["res_multistep"]
        assert sole(wf, "RandomNoise")["widgets_values"] == [42, "fixed"]

    def test_output_chain(self, wf):
        wf, kind = wf
        decode = sole(wf, "VAEDecode")
        decode_a = sole(wf, "VAEDecodeAudio")
        create = sole(wf, "CreateVideo")
        save = sole(wf, "SaveVideo")
        prev = sole(wf, "PreviewImage")
        assert input_link(create, "images") is not None
        assert input_link(create, "audio") is not None
        assert input_link(save, "video") is not None
        assert input_link(prev, "images") is not None
        cw = create["widgets_values"]
        assert cw[:3] == [24, "auto", "sRGB"], f"CreateVideo widgets {cw}"
        sw = save["widgets_values"]
        expected_prefix = (
            "video/h3_attention_test"
            if kind == "maximum_coherence"
            else "video/h3_attention_single_char"
        )
        assert sw[:3] == [expected_prefix, "auto", "auto"], f"SaveVideo widgets {sw}"
        # both decodes take their LATENT from the same SCA output slot
        sca = sole(wf, "SamplerCustomAdvanced")
        for dec in (decode, decode_a):
            src = next(l for l in wf["links"] if l[0] == input_link(dec, "samples"))
            assert (src[1], src[2]) == (sca["id"], 0)


def by_id_type(wf, nid):
    return nodes_by_id(wf)[nid]["type"]
