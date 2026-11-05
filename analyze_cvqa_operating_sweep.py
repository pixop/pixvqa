#!/usr/bin/env python3
"""Analyze PixVQA parameter sweep. Plot only configs with EVERY repetition faster than real-time.

Usage: python3 analyze_cvqa_operating_sweep.py cvqa_sweep_YYYYMMDD_HHMMSS
Requires matplotlib for plots; produces CSV reports even without matplotlib.
"""
import csv
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict


def kv_file(path):
    return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)


def mean(x): return statistics.mean(x)
def sd(x): return statistics.stdev(x) if len(x)>1 else 0.0


def write_csv(path, rows, fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gpu_stats(path):
    util, mem = [], []
    with path.open() as f:
        for line in f:
            parts=[v.strip() for v in line.split(',')]
            if len(parts)<3: continue
            try:
                util.append((float(parts[1]),parts[0]))
                mem.append((float(parts[2]),parts[0]))
            except ValueError:
                continue
    if not util:
        raise ValueError(f'No GPU samples in {path}')
    up=max(util)
    mp=max(mem)
    return mean([u for u,_ in util]), up[0], up[1], mp[0], mp[1]


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python3 analyze_cvqa_operating_sweep.py RESULTS_DIRECTORY')
    out=pathlib.Path(sys.argv[1]).resolve()
    if not out.is_dir(): raise SystemExit(f'No such directory: {out}')
    config=kv_file(out/'config.txt')
    source_fps=float(config['SOURCE_FPS'])
    manifest=list(csv.DictReader((out/'manifest.csv').open()))
    runs=[]
    for m in manifest:
        prefix=out/m['prefix']
        detail=pathlib.Path(str(prefix)+'_detail.json')
        if not detail.is_file():
            print(f'WARNING: missing detailed JSON: {detail.name}',file=sys.stderr)
            continue
        exit_path=pathlib.Path(str(prefix)+'_exit_code.txt')
        if exit_path.exists() and exit_path.read_text().strip()!='0':
            print(f'WARNING: unsuccessful: {prefix.name}',file=sys.stderr)
            continue
        data=json.loads(detail.read_text())
        samp=data.get('sampling',{})
        video_frames=int(samp.get('opencv_frame_count',0))
        actual_video_fps=float(samp.get('opencv_fps',0))
        if video_frames<=0 or actual_video_fps<=0:
            raise ValueError(f'Invalid video metadata in {detail}')
        if abs(source_fps-actual_video_fps)>0.01:
            raise ValueError(f'SOURCE_FPS={source_fps} but source is {actual_video_fps} fps')
        t=kv_file(pathlib.Path(str(prefix)+'_time.txt'))
        wall=float(t['wall_sec'])
        uavg,upeak,utime,mpeak,mtime=gpu_stats(pathlib.Path(str(prefix)+'_gpu.csv'))
        perf=data.get('performance',{})
        row={'frame_interval_s':float(m['interval_sec']), 'sampling_fps':float(m['fps']),
             'chunk_s':float(m['chunk_sec']), 'repeat':int(m['repeat']),
             'planned_batch':int(m['planned_batch']),
             'actual_batch':samp.get('chunk_max_sampled_frames'),
             'sampled_frames':samp.get('sampled_frames'),
             'wall_s':wall, 'video_frames':video_frames,
             'source_fps':actual_video_fps, 'effective_fps':video_frames/wall,
             'realtime_factor':(video_frames/actual_video_fps)/wall,
             'gpu_util_avg_pct':uavg, 'gpu_util_peak_pct':upeak,
             'gpu_util_peak_time':utime, 'peak_device_vram_mib':mpeak,
             'peak_device_vram_time':mtime,
             'peak_torch_allocated_mib':perf.get('peak_cuda_allocated_bytes',0)/1048576,
             'host_rss_peak_mib':float(t['peak_host_rss_kb'])/1024,
             'model_load_s':perf.get('model_load_seconds'),
             'preprocessing_s':perf.get('preprocessing_seconds'),
             'inference_s':perf.get('inference_seconds'),
             'quality_score':data.get('quality_score'), 'prefix':m['prefix']}
        runs.append(row)
    if not runs: raise SystemExit('No completed runs found')
    write_csv(out/'all_runs.csv',runs,list(runs[0]))
    by_config=defaultdict(list)
    for r in runs: by_config[(r['sampling_fps'],r['chunk_s'])].append(r)
    expected_repeats=int(config['REPEATS'])
    summary=[]
    for (fps,chunk), rr in sorted(by_config.items()):
        throughput=[r['effective_fps'] for r in rr]
        util=[r['gpu_util_avg_pct'] for r in rr]
        peakmem=[r['peak_device_vram_mib'] for r in rr]
        quality=[r['quality_score'] for r in rr if r['quality_score'] is not None]
        # Conservative criterion: all required repetitions must exceed source fps.
        valid=len(rr)==expected_repeats and len({r['repeat'] for r in rr})==expected_repeats
        rt=valid and min(throughput)>source_fps
        summary.append({
            'frame_interval_s':rr[0]['frame_interval_s'],'sampling_fps':fps,'chunk_s':chunk,
            'repeats':len(rr),'expected_repeats':expected_repeats,
            'avg_fps':mean(throughput),'fps_sd':sd(throughput),'min_fps':min(throughput),
            'wall_mean_s':mean([r['wall_s'] for r in rr]),
            'gpu_util_avg_pct':mean(util), 'gpu_util_sd_pct':sd(util),
            'gpu_util_peak_pct':max(r['gpu_util_peak_pct'] for r in rr),
            'vram_peak_gib':max(peakmem)/1024,
            'vram_peak_mean_gib':mean(peakmem)/1024,
            'quality_mean':mean(quality) if quality else float('nan'),
            'quality_sd':sd(quality) if quality else float('nan'),
            'realtime_pass':rt,
            'reason':('PASS' if rt else 'INCOMPLETE' if not valid else 'SLOW'),
        })
    write_csv(out/'all_operating_points.csv',summary,list(summary[0]))
    good=[r for r in summary if r['realtime_pass']]
    write_csv(out/'realtime_operating_points.csv',good,list(summary[0]))
    pareto=[]
    for a in good:
        dominated=any((b['gpu_util_avg_pct']<=a['gpu_util_avg_pct'] and
                       b['vram_peak_gib']<=a['vram_peak_gib'] and
                       (b['gpu_util_avg_pct']<a['gpu_util_avg_pct'] or
                        b['vram_peak_gib']<a['vram_peak_gib'])) for b in good)
        if not dominated: pareto.append(a)
    write_csv(out/'pareto_operating_points.csv',pareto,list(summary[0]))
    print(f'Completed runs: {len(runs)}/{len(manifest)}')
    print(f'Operating points: {len(summary)}; faster-than-real-time: {len(good)}; Pareto: {len(pareto)}')
    print('Criterion: ALL repetitions must exceed source-video fps; device VRAM uses max across repetitions.')
    print('\nInterval [s] | Chunk [s] | Avg source fps | Avg GPU [%] | Peak VRAM [GiB] | Status')
    for r in summary:
        print(f'{r["frame_interval_s"]:>12.6f} | {r["chunk_s"]:>9g} | {r["avg_fps"]:>14.2f} | '
              f'{r["gpu_util_avg_pct"]:>11.1f} | {r["vram_peak_gib"]:>15.2f} | {r["reason"]}')
    if not good:
        print('\nNo feasible configurations: charts not generated.')
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print('matplotlib not installed: CSV outputs written, plot skipped. Install via pip install matplotlib')
        return
    rates=sorted({r['sampling_fps'] for r in good})
    chunks=sorted({r['chunk_s'] for r in good})
    colors=plt.get_cmap('viridis')
    markers=['o','s','^','D','P','X','v','<','>']
    color_by_rate={fps:colors(i/max(1,len(rates)-1)) for i,fps in enumerate(rates)}
    marker_by_chunk={c:markers[i%len(markers)] for i,c in enumerate(chunks)}

    fig,ax=plt.subplots(figsize=(12.5,8.5))
    for idx,r in enumerate(good):
        x=r['gpu_util_avg_pct'];y=r['vram_peak_gib']
        ax.scatter(x,y,s=95,c=[color_by_rate[r['sampling_fps']]],marker=marker_by_chunk[r['chunk_s']],
                   edgecolors='black',linewidths=.5,zorder=4)
        label=f'({r["frame_interval_s"]:.4g}s, {r["chunk_s"]:g}s, {r["avg_fps"]:.0f}fps)'
        dx=6 if idx%2==0 else -6
        dy=8 if idx%3==0 else -11 if idx%3==1 else 2
        ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',
                    fontsize=7.2,ha='left' if dx>0 else 'right')
    if len(pareto)>1:
        pp=sorted(pareto,key=lambda r:r['gpu_util_avg_pct'])
        ax.plot([r['gpu_util_avg_pct'] for r in pp],[r['vram_peak_gib'] for r in pp],
                linestyle='--',alpha=.65,color='gray',label='Pareto frontier')
    ax.set(xlabel='Average GPU utilization (%)', ylabel='Peak device-wide GPU memory (GiB)',
           title=f'CompressedVQA-HDR-NR: faster-than-real-time operating points (>{source_fps:g} source fps)')
    ax.grid(alpha=.2)
    rate_handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=color_by_rate[f],
                        markeredgecolor='black',markersize=8,label=f'{f:g} sampled fps') for f in rates]
    chunk_handles=[Line2D([0],[0],marker=marker_by_chunk[c],color='black',linestyle='none',
                         markersize=8,label=f'{c:g}s chunk') for c in chunks]
    legend1=ax.legend(handles=rate_handles,title='Sampling frequency',loc='upper left',bbox_to_anchor=(1.02,1))
    ax.add_artist(legend1)
    ax.legend(handles=chunk_handles,title='Chunk duration',loc='lower left',bbox_to_anchor=(1.02,0))
    fig.text(.07,.02,'Point label: (frame interval, chunk duration, mean source-video processing fps). '
                  'VRAM is device-wide; all repeats must exceed real time.',fontsize=9)
    fig.tight_layout(rect=(0,.04,.81,1))
    fig.savefig(out/'gpu_util_vs_vram_realtime.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/'gpu_util_vs_vram_realtime.svg',bbox_inches='tight')
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10.5,6.5))
    for fps in rates:
        rr=sorted((r for r in good if r['sampling_fps']==fps),key=lambda r:r['chunk_s'])
        ax.plot([r['gpu_util_avg_pct'] for r in rr],[r['avg_fps'] for r in rr],
                marker='o',color=color_by_rate[fps],label=f'{fps:g} sampled fps')
        for r in rr:
            ax.annotate(f'{r["chunk_s"]:g}s',(r['gpu_util_avg_pct'],r['avg_fps']),
                        xytext=(3,5),textcoords='offset points',fontsize=8)
    ax.axhline(source_fps,color='red',linestyle='--',alpha=.6,label='Real-time threshold')
    ax.set(xlabel='Average GPU utilization (%)',ylabel='Average source-video processing fps',
           title='Real-time feasible configurations: throughput vs GPU utilization')
    ax.grid(alpha=.2)
    ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'throughput_vs_gpu_util_realtime.png',dpi=190)
    plt.close(fig)
    print('\nPlots: gpu_util_vs_vram_realtime.png / .svg; throughput_vs_gpu_util_realtime.png')
    print('CSVs: all_runs.csv, all_operating_points.csv, realtime_operating_points.csv, pareto_operating_points.csv')


if __name__=='__main__':
    main()
