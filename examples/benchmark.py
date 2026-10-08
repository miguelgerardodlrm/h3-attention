#!/usr/bin/env python3
"""
Benchmark script for H3 Attention Control.

Compares baseline H3 vs attention-controlled variants across multiple metrics.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np

from h3_attention import (
    H3AttentionController,
    AttentionControlConfig,
    AttentionSchedule,
    PRESETS,
    SCHEDULE_PRESETS,
    install_controller_on_model,
    compute_reference_affinity,
    compute_prompt_affinity,
    compute_temporal_affinity,
    aggregate_metrics,
)


def load_h3_model(model_path: str, variant: str = "ref2va", dtype: torch.dtype = torch.bfloat16):
    """Load MiniMax H3 model from diffusers."""
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks, MiniMaxH3Blocks

    if variant == "ref2va":
        pipe = MiniMaxH3Ref2VABlocks().init_pipeline(model_path)
    else:
        pipe = MiniMaxH3Blocks().init_pipeline(model_path)

    pipe.load_components(dtype=dtype)
    pipe.to("cuda")
    return pipe


def run_generation(
    pipe,
    prompt: str,
    references: List[Dict],
    num_frames: int = 124,
    num_steps: int = 30,
    seed: int = 42,
    controller: H3AttentionController = None,
) -> Dict[str, Any]:
    """Run a single generation and collect metrics."""

    generator = torch.Generator("cuda").manual_seed(seed)

    start_time = time.time()

    if controller:
        with controller.patch(
            pipe.transformer if hasattr(pipe, "transformer") else pipe.transformer_ref
        ):
            result = pipe(
                prompt=prompt,
                references=references,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                generator=generator,
            )
    else:
        result = pipe(
            prompt=prompt,
            references=references,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
        )

    elapsed = time.time() - start_time

    return {
        "video": result.get("videos"),
        "audio": result.get("audio"),
        "sampling_rate": result.get("sampling_rate"),
        "time_seconds": elapsed,
        "metrics": controller.get_metrics_summary() if controller else {},
    }


def compute_video_metrics(
    video_tensor: torch.Tensor, reference_tensors: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute quality metrics for generated video.

    In practice, you'd use:
    - LPIPS for perceptual similarity
    - CLIP-I for identity consistency
    - FVD for video quality
    - Custom identity/classifier metrics
    """
    # Placeholder - implement with actual metrics
    return {
        "lpips_to_ref": 0.0,
        "clip_identity_score": 0.0,
        "fvd": 0.0,
        "temporal_consistency": 0.0,
    }


def run_experiment(
    model_path: str,
    variant: str,
    prompt: str,
    references: List[Dict],
    config_name: str,
    num_frames: int = 124,
    num_steps: int = 30,
    seed: int = 42,
    runs: int = 3,
) -> Dict[str, Any]:
    """Run a single experiment configuration multiple times."""

    print(f"\n{'=' * 60}")
    print(f"Experiment: {config_name}")
    print(f"Variant: {variant}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Runs: {runs}")
    print(f"{'=' * 60}")

    # Load model once
    pipe = load_h3_model(model_path, variant)

    # Get config
    if config_name == "baseline":
        config = AttentionControlConfig()
        schedule = None
    else:
        config = PRESETS.get(config_name, PRESETS["maximum_coherence"]).copy()
        schedule = SCHEDULE_PRESETS.get("balanced")

    all_results = []
    all_metrics = []

    for run in range(runs):
        print(f"\nRun {run + 1}/{runs} (seed={seed + run})...")

        controller = None
        if config.is_active:
            controller = H3AttentionController(config=config, schedule=schedule)

        result = run_generation(
            pipe, prompt, references, num_frames, num_steps, seed + run, controller
        )

        all_results.append(result)
        if controller:
            all_metrics.extend(controller.state.metrics_history)

        print(f"  Time: {result['time_seconds']:.1f}s")
        if controller:
            summary = controller.get_metrics_summary()
            print(f"  Ref Affinity: {summary.get('avg_reference_affinity', 0):.3f}")
            print(f"  Prompt Affinity: {summary.get('avg_prompt_affinity', 0):.3f}")
            print(f"  Temporal Affinity: {summary.get('avg_temporal_affinity', 0):.3f}")

    # Aggregate
    avg_time = np.mean([r["time_seconds"] for r in all_results])

    return {
        "config": config_name,
        "variant": variant,
        "runs": runs,
        "avg_time_seconds": float(avg_time),
        "individual_times": [r["time_seconds"] for r in all_results],
        "aggregated_metrics": aggregate_metrics(all_metrics) if all_metrics else {},
    }


def main():
    parser = argparse.ArgumentParser(description="H3 Attention Control Benchmark")
    parser.add_argument("--model-path", required=True, help="Path to MiniMax-H3 model")
    parser.add_argument("--variant", choices=["fl2va", "ref2va"], default="ref2va")
    parser.add_argument("--prompt", default="A person dancing in a room")
    parser.add_argument("--reference-dir", help="Directory with reference images/videos")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "reference_only", "identity_preservation", "maximum_coherence"],
    )
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="benchmark_results.json")

    args = parser.parse_args()

    # Prepare references
    references = []
    if args.reference_dir:
        ref_path = Path(args.reference_dir)
        for img_path in sorted(ref_path.glob("*.png")) + sorted(ref_path.glob("*.jpg")):
            references.append({"type": "image", "path": str(img_path)})

    if not references:
        print("Warning: No references provided, using text-only generation")
        references = []

    # Run experiments
    all_experiments = []

    for config_name in args.configs:
        try:
            exp_result = run_experiment(
                args.model_path,
                args.variant,
                args.prompt,
                references,
                config_name,
                args.num_frames,
                args.num_steps,
                args.seed,
                args.runs,
            )
            all_experiments.append(exp_result)
        except Exception as e:
            print(f"Error in {config_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    results = {
        "model_path": args.model_path,
        "variant": args.variant,
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "runs_per_config": args.runs,
        "experiments": all_experiments,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {args.output}")

    # Print summary table
    print("\nSummary:")
    print(
        f"{'Config':<25} {'Avg Time (s)':<15} {'Ref Affinity':<15} {'Prompt Affinity':<15} {'Temp Affinity':<15}"
    )
    print("-" * 85)

    for exp in all_experiments:
        m = exp.get("aggregated_metrics", {})
        print(
            f"{exp['config']:<25} {exp['avg_time_seconds']:<15.1f} "
            f"{m.get('mean_reference_affinity', 0):<15.3f} "
            f"{m.get('mean_prompt_affinity', 0):<15.3f} "
            f"{m.get('mean_temporal_affinity', 0):<15.3f}"
        )


if __name__ == "__main__":
    main()
