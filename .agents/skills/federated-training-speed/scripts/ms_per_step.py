"""Milliseconds per optimizer step of a federated run, from its committed per-round JSON records.

Each record (one JSON file per committed round) is expected to hold a list of per-client entries with the
seconds that client's local epoch took and its step count; key names are options so any project schema
fits. With W worker processes the per-process ms/step is what to compare against a benchmark; the round
wall time also contains aggregation and evaluation.

    python ms_per_step.py RUN_DIR [--glob 'complete/round_*.json'] [--clients clients] [--seconds seconds]
                          [--steps applied,skipped] [--wall sec]
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir'); p.add_argument('--glob', default='complete/round_*.json')
    p.add_argument('--clients', default='clients'); p.add_argument('--seconds', default='seconds')
    p.add_argument('--steps', default='applied,skipped', help='comma list of keys summed into the step count')
    p.add_argument('--wall', default='sec', help='round wall-time key (optional)')
    a = p.parse_args(); keys = a.steps.split(',')
    files = sorted(Path(a.run_dir).glob(a.glob))
    if not files: raise SystemExit(f'no records matching {a.glob} under {a.run_dir}')
    print('round | clients | client-seconds | steps | ms/step | round wall s')
    total_s = total_n = 0
    for f in files:
        row = json.loads(f.read_text()); clients = row[a.clients]
        sec = sum(c[a.seconds] for c in clients); steps = sum(sum(c[k] for k in keys) for c in clients)
        total_s += sec; total_n += steps
        print(f"{row.get('round', f.stem)} | {len(clients)} | {sec:.0f} | {steps} | {1000 * sec / max(steps, 1):.2f} | "
              f"{row.get(a.wall, '')}")
    print(f'all | | {total_s:.0f} | {total_n} | {1000 * total_s / max(total_n, 1):.2f} |')


if __name__ == '__main__':
    main()
