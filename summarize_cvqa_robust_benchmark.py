#!/usr/bin/env python3
"""Summarize PixVQA phase-level JSON, wall times and NVIDIA GPU samples."""
import csv
import glob
import json
import pathlib
import re
import statistics
import sys


def load_kv(path):
    return dict(line.strip().split('=', 1) for line in path.read_text().splitlines() if '=' in line)


def summarize_run(prefix):
    data = json.loads(pathlib.Path(str(prefix) + '_detail.json').read_text())
    perf = data['performance']
    t = load_kv(pathlib.Path(str(prefix) + '_time.txt'))
    wall = float(t['wall_sec'])
    startup = float(perf['model_load_seconds'])
    preprocess = float(perf['preprocessing_seconds'])
    infer = float(perf['inference_seconds'])
    adapter_elapsed = float(data['elapsed_seconds'])
    residual = wall - startup - preprocess - infer
    samples = []
    with open(str(prefix) + '_gpu.csv', newline='') as f:
        for line in f:
            parts = [item.strip() for item in line.strip().split(',')]
            if len(parts) < 3:
                continue
            try:
                samples.append((parts[0], float(parts[1]), float(parts[2])))
            except ValueError:
                pass
    peak_util = max(samples, key=lambda s: s[1]) if samples else ('', float('nan'), float('nan'))
    peak_mem = max(samples, key=lambda s: s[2]) if samples else ('', float('nan'), float('nan'))
    util_avg = statistics.mean(s[1] for s in samples) if samples else float('nan')
    sampling = data.get('sampling', {})
    sampled = sampling.get('sampled_frames', data.get('sampled_frames'))
    match = re.search(r'interval_(1|0\.5|0\.25)_rep_(\d+)$', prefix.name)
    if not match:
        raise ValueError(f'Unexpected filename: {prefix.name}')
    return {
        'interval_s': match.group(1), 'repeat': int(match.group(2)),
        'sampled_frames': sampled, 'quality_score': data['quality_score'],
        'wall_s': wall, 'effective_video_fps': 3600 / wall,
        'real_time_factor': 60 / wall,
        'model_load_s': startup, 'preprocessing_s': preprocess,
        'inference_s': infer, 'adapter_elapsed_s': adapter_elapsed,
        'other_s': residual,
        'other_note': 'Includes Python imports, preflight, process startup, JSON/metrics overhead and measurement-boundary differences',
        'gpu_util_mean_pct': util_avg, 'gpu_util_peak_pct': peak_util[1],
        'gpu_util_peak_timestamp': peak_util[0],
        'device_peak_mib': peak_mem[2], 'device_peak_timestamp': peak_mem[0],
        'torch_peak_allocated_mib': perf.get('peak_cuda_allocated_bytes', 0)/1048576,
        'torch_peak_reserved_mib': perf.get('peak_cuda_reserved_bytes', 0)/1048576,
        'host_rss_peak_mib': float(t['peak_host_rss_kb'])/1024,
        'gpu_monitor_samples': len(samples),
    }


def fmt(v):
    return f'{v:.2f}' if isinstance(v, float) else str(v)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 summarize_cvqa_robust_benchmark.py BENCHMARK_DIRECTORY')
    out = pathlib.Path(sys.argv[1])
    paths = sorted(out.glob('interval_*_rep_*_detail.json'))
    if not paths:
        raise SystemExit(f'No detailed JSON files found in {out}')
    rows = [summarize_run(pathlib.Path(str(p).removesuffix('_detail.json'))) for p in paths]
    csv_path = out / 'all_runs.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nPer-run results: {csv_path}\n')
    print('Sampling | n | wall mean +/- sd [s] | preprocess mean [s] | model mean [s] | GPU infer mean [s] | GPU util avg [%] | VRAM peak max [MiB] | quality mean +/- sd')
    for interval in ('1','0.5','0.25'):
        these = [r for r in rows if r['interval_s']==interval]
        if not these:
            continue
        avg = lambda key: statistics.mean(r[key] for r in these)
        sd = lambda key: statistics.stdev(r[key] for r in these) if len(these)>1 else 0.0
        print(f'{interval:>8} | {len(these)} | {avg("wall_s"):.2f} +/- {sd("wall_s"):.2f} | '
              f'{avg("preprocessing_s"):.2f} | {avg("model_load_s"):.2f} | {avg("inference_s"):.2f} | '
              f'{avg("gpu_util_mean_pct"):.1f} | {max(r["device_peak_mib"] for r in these):.0f} | '
              f'{avg("quality_score"):.4f} +/- {sd("quality_score"):.4f}')
    print('\nNote: Inference includes device transfer, forward, CPU score transfer and CUDA synchronization.')
    print('Residual "other_s" is not independently profiled; do not call it CPU decoding.')
    print('GPU peak times are samples, not evidence that a specific kernel caused the peak.')
    print('Effective video fps uses 3600 frames / wall seconds (offline throughput, not live latency).')


if __name__ == '__main__':
    main()
