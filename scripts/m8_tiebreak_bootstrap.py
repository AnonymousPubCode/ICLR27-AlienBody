"""n=8 AFMB vs Enumerate+Verify: per-env paired bootstrap CI on SR difference.

Data: results/m8_enum_fixed (45/50) vs results/m8_afmb_s0 (50/50), same 50 test envs.
jsonl may contain appended duplicates (run_eval without --resume appends); keep the
LAST record per env_id. Report counts per env too (any success vs all).
"""
import json, glob, sys
from collections import defaultdict

def per_env(paths):
    """Return {env_id: [success,...]} keeping all records; dedupe by last occurrence."""
    recs = {}
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                recs[d['env_id']] = d  # later file/line wins
    return recs

def load(globpat):
    return per_env(glob.glob(globpat))

enum = load('code/results/m8_enum_fixed/*/family4_test.jsonl')
afmb = load('code/results/m8_afmb_s0/*/family4_test.jsonl')
enum2 = load('code/results/m8_afmb_a2_passive/*/family4_test.jsonl')  # passive ablation
noeig = load('code/results/m8_afmb_a7_noeig/*/family4_test.jsonl')   # no_eig ablation

print('n envs: enum', len(enum), 'afmb', len(afmb), 'passive', len(enum2), 'noeig', len(noeig))

common = sorted(set(enum) & set(afmb))
print('common envs:', len(common))

def sr(recs):
    s = sum(1 for e in common if recs[e].get('success'))
    return s, len(common)

se, ne = sr(enum); sa, na = sr(afmb)
print(f'Enumerate SR: {se}/{ne} = {100*se/ne:.1f}%')
print(f'AFMB SR:      {sa}/{na} = {100*sa/na:.1f}%')

# which envs differ
diffs = [e for e in common if enum[e].get('success') != afmb[e].get('success')]
print('disagreeing envs:', len(diffs), diffs[:10])
for e in diffs:
    print(f'  {e}: enum={enum[e].get("success")} afmb={afmb[e].get("success")} | enum P1={enum[e].get("phase1_steps")} afmb P1={afmb[e].get("phase1_steps")}')

# paired env bootstrap on difference
import random
rng = random.Random(1234)
B = 200_000
diffs_sr = []
for _ in range(B):
    ids = [rng.choice(common) for _ in range(len(common))]
    d = (sum(1 for e in ids if afmb[e]['success']) - sum(1 for e in ids if enum[e]['success'])) / len(ids) * 100
    diffs_sr.append(d)
diffs_sr.sort()
lo, hi = diffs_sr[int(0.025*B)], diffs_sr[int(0.975*B)]
print(f'AFMB - Enumerate: +{100*(sa/se - 1)*0 if False else (sa-na)/na*0 + (sa-se)/na*100:.1f}pp [bootstrap 95% CI {lo:.1f}, {hi:.1f}]')
# fraction of bootstrap diffs <= 0 (one-sided p)
p_less = sum(1 for d in diffs_sr if d <= 0) / B
print(f'P(diff<=0) = {p_less:.4f}')

# also P1 (phase-1 steps) comparison on all envs
import statistics
p1_enum = [enum[e]['phase1_steps'] for e in common]
p1_afmb = [afmb[e]['phase1_steps'] for e in common]
print(f'P1 mean: enum {statistics.mean(p1_enum):.1f} vs afmb {statistics.mean(p1_afmb):.1f}')

# ablations
for name, recs in [('passive', enum2), ('no_eig', noeig)]:
    if recs:
        c = sorted(set(recs) & set(common))
        s = sum(1 for e in c if recs[e].get('success'))
        print(f'{name}: {s}/{len(c)} = {100*s/len(c):.1f}%')
