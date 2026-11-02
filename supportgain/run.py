"""TEF — single-protocol entry point.

Usage
-----
    python run.py --model model1 --phase phase1
    python run.py --model qwen25vl --phase phase2 --gpu 0 --start 0 --end 250
    python run.py --model hawkeye --phase phase1 --steps 1,2,3,4

Steps
-----
    1  full_inference        (a_full + grounding + IoU)
    2  segment_preparation   (R_pred + R_oracle with N seeds)
    3  intervention          (Keep/Remove for every segment family + Blind)
    4  metric_computation    (the older step-4 metric suite, kept as-is; no
                             paper number comes from it — see README,
                             "Legacy key names")

Per-sample inference count: 7 + 4*N_random   (=19 when N=3).

Adding a new model
------------------
    1) Copy ``models/model1/`` to ``models/<your_name>/``.
    2) Implement load/decode_video/predict_qa/predict_grounding in
       ``models/<your_name>/adapter.py``.
    3) Run ``python run.py --model <your_name> --phase phase1``.
"""
import argparse
import importlib
import os
import sys

# Make TEF root importable as the package root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_steps(s):
    if s == 'all':
        return [1, 2, 3, 4]
    return [int(x) for x in s.split(',')]


def _shard_path(base, shard):
    """Append `_shard` before the .json extension."""
    if not shard:
        return base
    root, ext = os.path.splitext(base)
    return f'{root}_{shard}{ext}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='Subdir under models/, e.g. "model1" or "qwen25vl"')
    p.add_argument('--phase', default='phase1',
                   choices=['phase1', 'phase2', 'phase3'])
    p.add_argument('--gpu', default='0')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=-1)
    p.add_argument('--steps', default='all',
                   help='comma list, e.g. "1,3,4" or "all"')
    p.add_argument('--shard', default='',
                   help='Suffix for sharded output files (e.g. "gpu0"). '
                        'When set, step 1/3 write to '
                        'step1_<shard>.json / step3_<shard>.json so that '
                        'multiple GPUs can run the same model in parallel '
                        'without clobbering each other. Steps 2 and 4 still '
                        'read/write the un-sharded files; run '
                        '``scripts/merge_shards.py`` between step 1 and 2.')
    p.add_argument('--families', default='all',
                   choices=['all', 'self', 'gt'],
                   help='Which metric families step 3 runs. "self" = E_hat + '
                        'R_pred (Support Gain on the predicted window); '
                        '"gt" = E_star + R_oracle (the positive control, plus '
                        'the gt-dependent step-4 metrics); "all" = both '
                        '(default). '
                        'Run "self" first, then "gt" later to top-up GT '
                        'metrics without recomputing the self inferences '
                        '(label-level merge into the existing step3.json).')
    args = p.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    steps = _parse_steps(args.steps)

    from supportgain.config.settings import RESULTS_DIR

    # Lazy-import the adapter so steps 2 and 4 (no model needed) can run
    # without loading the model into memory.
    adapter = None
    needs_model = any(s in steps for s in (1, 3))
    if needs_model:
        mod = importlib.import_module(f'models.{args.model}.adapter')
        adapter = mod.Adapter()
        adapter.load(device='cuda:0')
        print(f'[run] Loaded adapter: {adapter.name}', flush=True)

    if 1 in steps:
        from supportgain.protocol import step1_full
        out = _shard_path(
            os.path.join(RESULTS_DIR, args.model, 'step1.json'), args.shard,
        )
        step1_full.run(adapter, args.model, start=args.start, end=args.end,
                       out_path=out)

    if 2 in steps:
        from supportgain.protocol import step2_segments
        step2_segments.run(args.model)

    if 3 in steps:
        from supportgain.protocol import step3_intervention
        out = _shard_path(
            os.path.join(RESULTS_DIR, args.model, args.phase, 'step3.json'),
            args.shard,
        )
        step3_intervention.run(adapter, args.model, phase=args.phase,
                               start=args.start, end=args.end,
                               out_path=out, families=args.families)

    if 4 in steps:
        from supportgain.protocol import step4_metrics
        step4_metrics.run(args.model, phase=args.phase)


if __name__ == '__main__':
    main()
