"""Versioned extensions. Frozen source is imported, never edited in place."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'frozen'))
import frozen_abm as abm
from frozen_estimators import projection_games, kernels, complete_u, bootstrap_u

# Preserve the original seed namespace bytes for exact replay of archived runs.
# Hex notation is a compatibility constant, not a new randomization namespace.
NS = bytes.fromhex('69706d2d636f6e74656e742d676f7665726e616e63652d32303236303931362d7631').decode('ascii')
TOPOLOGIES = ['BA', 'ER', 'WS', 'SBM']
REGIMES = ['single_shot', 'burst', 'continuous']
STREAMS = ['network', 'initialization', 'seeding', 'sharing', 'noise', 'inoculation']
BASE = yaml.safe_load((ROOT / 'inputs/base_config.yaml').read_text(encoding='utf-8'))['simulation']
ORIGINAL_BUILD = abm.build_network
ORIGINAL_INOCULATION = abm.apply_inoculation


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seeds_for(experiment, topology, replication):
    key = f'{NS}|{experiment}|{topology}|{replication}'
    master = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
    values = {s: int(hashlib.sha256(f'{master}|{s}'.encode()).hexdigest()[:15], 16)
              for s in STREAMS}
    return {'master_seed': master, 'seed_net': values['network'],
            'seed_attr': values['initialization'], 'seed_seeding': values['seeding'],
            'seed_sharing': values['sharing'], 'seed_noise': values['noise'],
            'seed_inoc': values['inoculation'], 'seed_sim': None}


_graphs = {}


def cached_build(topology, n, opinions, seed):
    # The frozen simulator reads graph structure but never mutates a built graph.
    key = (topology, n, seed, hashlib.sha256(opinions.tobytes()).hexdigest())
    if key not in _graphs:
        _graphs[key] = ORIGINAL_BUILD(topology, n, opinions, seed)
    return _graphs[key]


def corrected_inoculation(g, agents, strategy, budget_fraction, kappa,
                          compliance_fraction, seed):
    n = len(agents.opinions)
    if not (0 <= budget_fraction <= 1 and 0 <= compliance_fraction <= 1 and 0 <= kappa <= 1):
        raise ValueError('Budget, compliance and kappa must lie in [0,1].')
    k = int(budget_fraction * n)
    if strategy == 'none' or k == 0 or compliance_fraction == 0:
        return
    rng = np.random.default_rng(seed)
    if strategy == 'random':
        idx = rng.choice(n, size=k, replace=False)
    elif strategy == 'hub':
        nodes = np.arange(n)
        degrees = np.array([g.degree(int(i)) for i in nodes])
        idx = np.lexsort((nodes, -degrees))[:k]
    else:
        raise ValueError(strategy)
    if compliance_fraction < 1:
        n_comply = int(compliance_fraction * k)
        if n_comply == 0:
            return
        idx = rng.choice(idx, size=n_comply, replace=False)
    agents.inoculated[idx] = True
    agents.susceptibility[idx] *= kappa


def policies():
    out = [dict(policy='none', mechanism='none', inoculation='none', budget=0.,
                compliance=1., label_lambda=0., label_loss=False, selector='corrected')]
    for lam, loss in itertools.product([.3, .7], [False, True]):
        out.append(dict(policy=f'label_L{int(lam*100):02d}_{"lost" if loss else "retained"}',
                        mechanism='label', inoculation='none', budget=0., compliance=1.,
                        label_lambda=lam, label_loss=loss, selector='corrected'))
    for strategy, budget, compliance in itertools.product(['random', 'hub'], [.05, .2], [.5, 1.]):
        out.append(dict(policy=f'{strategy}_B{int(budget*100):02d}_C{int(compliance*100):03d}',
                        mechanism='inoculation', inoculation=strategy, budget=budget,
                        compliance=compliance, label_lambda=0., label_loss=False,
                        selector='corrected'))
    for strategy in ['random', 'hub']:
        out.append(dict(policy=f'combined_{strategy}', mechanism='combined',
                        inoculation=strategy, budget=.05, compliance=1.,
                        label_lambda=.7, label_loss=False, selector='corrected'))
    legacy = [dict(p, policy=p['policy'] + '_legacy', selector='legacy')
              for p in out if p['inoculation'] == 'hub']
    return out + legacy


def make_tasks(suite, reps=30):
    tasks = []
    if suite == 'E1':
        for rep, topo, mu, dose, regime, strength in itertools.product(
                range(reps), TOPOLOGIES, [0., .05], [.05, .5, .95], REGIMES, [.06, .14, .18]):
            tasks.append(dict(suite=suite, replication=rep, topology=topo, dose=dose,
                              seeding_regime=regime, strength=strength, reversion=mu))
    elif suite == 'E2':
        for rep, topo, dose, regime, policy in itertools.product(
                range(reps), TOPOLOGIES, [.05, .5], ['single_shot', 'continuous'], policies()):
            tasks.append(dict(suite=suite, replication=rep, topology=topo, dose=dose,
                              seeding_regime=regime, strength=.14, reversion=0., **policy))
    elif suite == 'E3':
        for rep, topo, stubbornness in itertools.product(range(reps), TOPOLOGIES, [.2, .8]):
            tasks.append(dict(suite=suite, replication=rep, topology=topo, stubbornness=stubbornness,
                              dose=0., seeding_regime='none', strength=0.))
            for dose, regime, strength in itertools.product([.05, .5, .95], REGIMES, [.1, .3]):
                tasks.append(dict(suite=suite, replication=rep, topology=topo, stubbornness=stubbornness,
                                  dose=dose, seeding_regime=regime, strength=strength))
    else:
        raise ValueError(suite)
    for task in tasks:
        task['run_id'] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]
    return tasks


def run_frozen_extension(task):
    abm.build_network = cached_build
    cfg = dict(BASE)
    cfg['seeding_total_fractions'] = {r: task['dose'] for r in REGIMES}
    cfg['opinion_reversion_mu'] = task['reversion']
    inoculation = task.get('inoculation', 'none')
    labeled = task.get('label_lambda', 0.) > 0
    cfg['disclosure_lambda'] = task.get('label_lambda', 0.)
    cfg['inoculation_budget_fraction'] = task.get('budget', 0.)
    cfg['compliance_fraction'] = task.get('compliance', 1.)
    cfg['label_loss_sharing'] = task.get('label_loss', False)
    abm.apply_inoculation = (ORIGINAL_INOCULATION if task.get('selector') == 'legacy'
                            else corrected_inoculation)
    seed = seeds_for(task['suite'], task['topology'], task['replication'])
    result = abm.run_single(task['topology'], 'labeled_ai' if labeled else 'synthetic_message',
                            task['strength'], inoculation, task['seeding_regime'], cfg,
                            task['replication'], seed_override=seed)
    if result['total_direct_exposures'] != round(task['dose'] * cfg['n_agents']):
        raise AssertionError('Assigned direct dose not realized exactly.')
    # This AUC is a new outcome calculated from the already recorded trajectory.
    result['mean_trajectory_shift'] = float(np.mean(json.loads(result['cumulative_total_shift_by_step'])))
    result['compliant_target_count'] = (int(cfg['n_agents'] * task.get('budget', 0.) *
                                          task.get('compliance', 1.)) if inoculation != 'none' else 0)
    result.update(task)
    result['model_version'] = ('frozen-dynamics-no-inoculation' if task['suite'] == 'E1'
                               else 'frozen-dynamics-' + task['selector'] + '-selector')
    return result


@lru_cache(maxsize=256)
def fj_setup(topology, rep, stubbornness):
    seed = seeds_for('E3', topology, rep)
    agents = abm.init_agents(1000, seed['seed_attr'])
    x0 = agents.opinions.copy()
    g = cached_build(topology, 1000, x0, seed['seed_net'])
    adj = nx.to_scipy_sparse_array(g, nodelist=range(1000), dtype=float, format='csr')
    deg = np.asarray(adj.sum(axis=1)).ravel()
    if np.any(deg == 0):
        adj = adj + sparse.diags((deg == 0).astype(float))
        deg = np.asarray(adj.sum(axis=1)).ravel()
    weights = sparse.diags(1 / deg) @ adj
    control = x0.copy()
    trajectory = []
    for _ in range(50):
        control = stubbornness * x0 + (1 - stubbornness) * (weights @ control)
        trajectory.append(control.copy())
    return x0, weights, np.array(trajectory), seed


def run_fj(task):
    x0, weights, controls, seed = fj_setup(task['topology'], task['replication'], task['stubbornness'])
    x = x0.copy()
    rng = np.random.default_rng(seed['seed_seeding'])
    exposures = np.zeros(1000, dtype=int)
    curve = []
    direct_by_step = []
    for t in range(50):
        frac = (0. if task['dose'] == 0 else abm.get_seed_fraction_at_step(
            task['seeding_regime'], t, 50, .05, False, task['dose']))
        k = int(round(frac * 1000))
        e = np.zeros(1000)
        if k:
            idx = rng.choice(1000, k, replace=False)
            e[idx] = 1
            exposures[idx] += 1
        b = task['strength'] * e
        peer = weights @ x
        x = task['stubbornness'] * x0 + (1 - task['stubbornness']) * ((1-b)*peer + b*.6)
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1 + 1e-12:
            raise AssertionError('Invalid bounded FJ update.')
        curve.append(float(np.mean(np.abs(x - controls[t]))))
        direct_by_step.append(k)
    opposed = x0 < 0
    ctrl = controls[-1]
    delta = x - ctrl
    return dict(task, **seed, model_version='FJ-inspired-peer-averaging-v1',
                N_agents=1000, T_steps=50,
                mean_total_shift=float(np.mean(np.abs(delta))),
                mean_signed_shift=float(np.mean(delta * np.sign(.6-x0))),
                conversion_rate=float(np.mean(x[opposed] > 0) - np.mean(ctrl[opposed] > 0)),
                raw_total_shift=float(np.mean(np.abs(x-x0))),
                raw_conversion_rate=float(np.mean(x[opposed] > 0)),
                control_conversion_rate=float(np.mean(ctrl[opposed] > 0)),
                mean_trajectory_shift=float(np.mean(curve)),
                total_direct_exposures=int(exposures.sum()),
                unique_directly_exposed=int(np.count_nonzero(exposures)),
                cumulative_total_shift_by_step=json.dumps(curve, separators=(',', ':')),
                direct_exposures_by_step=json.dumps(direct_by_step, separators=(',', ':')))


def execute_task(task):
    return run_fj(task) if task['suite'] == 'E3' else run_frozen_extension(task)


def run_suite(suite, workers=6, reps=30):
    path = ROOT / 'results' / f'{suite}_runs.jsonl'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing run output: {path}')
    tasks = make_tasks(suite, reps)
    pd.DataFrame(tasks).to_csv(ROOT / 'inputs' / f'{suite}_design_matrix.csv', index=False)
    start = time.perf_counter()
    rows = []
    with open(path, 'w', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(execute_task, tasks, chunksize=30), 1):
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + '\n')
            if i % 300 == 0 or i == len(tasks):
                f.flush()
                print(f'{suite}: {i}/{len(tasks)} runs; {time.perf_counter()-start:.1f}s', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'results' / f'{suite}_runs.csv.gz', index=False, compression='gzip')
    seed_cols = ['run_id', 'suite', 'topology', 'replication'] + [c for c in frame if c.startswith('seed_') or c == 'master_seed']
    frame[seed_cols].to_csv(ROOT / 'results' / f'{suite}_seed_manifest.csv', index=False)
    summary = dict(suite=suite, rows=len(frame), replications=reps,
                   elapsed_seconds=time.perf_counter()-start,
                   unique_run_ids=int(frame.run_id.nunique()),
                   outcome_finite=bool(np.isfinite(frame[['mean_total_shift','mean_signed_shift','conversion_rate']]).all().all()))
    (ROOT/'validation'/f'{suite}_execution.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


def verify_source(source):
    source = Path(source)
    expected = source / 'code' / 'run_revision.py'
    assert sha_file(expected) == sha_file(ROOT/'frozen/frozen_abm.py')
    data = pd.read_csv(source/'data/combined_runs_120960.csv.gz')
    sample = []
    checks = []
    hub_rows = []
    cfg = dict(BASE)
    abm.build_network = ORIGINAL_BUILD
    abm.apply_inoculation = ORIGINAL_INOCULATION
    for topo, regime in itertools.product(TOPOLOGIES, ['single_shot','continuous']):
        row = data[(data.design=='CS-0') & (data.topology==topo) &
                   (data.seeding_regime==regime) & (data.message_condition=='high_quality_ai') &
                   (data.inoculation=='hub') & (data.replication==0)].iloc[0]
        seed_columns = {'master_seed':'master_seed', 'seed_net':'network_seed',
                        'seed_attr':'initialization_seed', 'seed_seeding':'seeding_seed',
                        'seed_sharing':'sharing_seed', 'seed_noise':'noise_seed',
                        'seed_inoc':'inoculation_seed'}
        seed = {k: int(row[c]) for k,c in seed_columns.items()}
        seed['seed_sim'] = None
        result = abm.run_single(topo,'high_quality_ai',.18,'hub',regime,cfg,0,seed_override=seed)
        numeric = ['mean_total_shift','mean_signed_shift','conversion_rate','total_direct_exposures','total_secondhand_exposures']
        matched = {key: bool(np.isclose(row[key],result[key],rtol=1e-10,atol=1e-12)) for key in numeric}
        for key in ['direct_exposures_by_step','secondhand_exposures_by_step','cumulative_total_shift_by_step']:
            matched[key] = bool(np.allclose(json.loads(row[key]),json.loads(result[key]),rtol=1e-10,atol=1e-12))
        checks.append(dict(topology=topo, regime=regime, checks=matched, passed=all(matched.values())))
        sample.append(row.to_dict())
        agents = abm.init_agents(1000, seed['seed_attr'])
        graph = ORIGINAL_BUILD(topo,1000,agents.opinions,seed['seed_net'])
        original_ids = np.argsort(np.array([d for _,d in graph.degree()]))[-50:]
        degree_by_id = np.array([graph.degree(i) for i in range(1000)])
        correct_ids = np.lexsort((np.arange(1000), -degree_by_id))[:50]
        cutoff = np.sort(degree_by_id)[-50]
        hub_rows.append(dict(topology=topo, regime=regime,
                             node_iteration_is_id_order=list(graph.nodes())==list(range(1000)),
                             selected_mean_degree_legacy=float(degree_by_id[original_ids].mean()),
                             selected_mean_degree_corrected=float(degree_by_id[correct_ids].mean()),
                             legacy_selected_below_top_k_cutoff=int(np.sum(degree_by_id[original_ids]<cutoff)),
                             overlap_count=len(set(original_ids)&set(correct_ids))))
    pd.DataFrame(sample).to_csv(ROOT/'inputs/archived_replay_sample.csv.gz', index=False)
    pd.DataFrame(hub_rows).to_csv(ROOT/'validation/hub_selector_audit.csv', index=False)
    provenance = dict(source_root=str(source), original_model_sha256=sha_file(expected),
                      frozen_estimator_sha256=sha_file(ROOT/'frozen/frozen_estimators.py'),
                      original_archive_sha256=sha_file(source/'data/combined_runs_120960.csv.gz'),
                      source_rows=len(data), source_designs=int(data.design.nunique()),
                      replay_checks=checks, replay_passed=all(c['passed'] for c in checks),
                      python=sys.version, platform=platform.platform(),
                      package_versions={p:__import__(p).__version__ for p in ['numpy','pandas','scipy','networkx','matplotlib','yaml']},
                      protocol_sha256=sha_file(ROOT/'PROTOCOL.md'))
    (ROOT/'validation/source_verification.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(provenance,ensure_ascii=False,indent=2))
    if not provenance['replay_passed']:
        raise AssertionError('Frozen replay failed; do not start extensions.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite', choices=['E1','E2','E3'])
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--verify-source')
    args=p.parse_args()
    if args.verify_source:
        verify_source(args.verify_source)
    if args.suite:
        run_suite(args.suite,args.workers)


if __name__=='__main__':
    main()
