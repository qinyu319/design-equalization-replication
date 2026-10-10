import itertools
import json
import networkx as nx
import numpy as np
import pandas as pd
from experiments import (abm, corrected_inoculation, make_tasks, run_fj, run_frozen_extension,
                         projection_games, kernels, complete_u, seeds_for, ORIGINAL_INOCULATION)


def test_hub_selection_uses_node_ids_not_positions():
    graph=nx.Graph()
    graph.add_nodes_from([3,1,0,2])
    graph.add_edges_from([(3,0),(3,1),(3,2),(0,1)])
    agents=abm.init_agents(4,17)
    corrected_inoculation(graph,agents,'hub',.25,.2,1.,8)
    assert np.flatnonzero(agents.inoculated).tolist()==[3]


def test_zero_intervention_boundaries_and_full_compliance():
    graph=nx.path_graph(100)
    for budget,compliance in [(0.,1.),(.05,0.)]:
        agents=abm.init_agents(100,1)
        old=agents.susceptibility.copy()
        corrected_inoculation(graph,agents,'random',budget,.2,compliance,2)
        assert not agents.inoculated.any()
        np.testing.assert_array_equal(old,agents.susceptibility)
    agents=abm.init_agents(100,1)
    corrected_inoculation(graph,agents,'random',.2,.2,.5,2)
    assert agents.inoculated.sum()==10


def test_shapley_truth_pair_identity_and_covariance_subtraction():
    cells=pd.DataFrame(itertools.product([-1.,1.],repeat=4),columns=list('ABCD'))
    a,b,c,_=cells.to_numpy().T
    mu=.8*a+.5*b+.6*a*b+.3*c
    fh,h=projection_games(cells,list('ABCD'))
    np.testing.assert_allclose(np.einsum('c,kcd,d->k',mu,fh,mu),[.82,.43,.09,0],atol=1e-12)
    y=mu[:,None]+np.random.default_rng(8).normal(size=(16,7))
    k=kernels(y,h)
    brute=np.stack([y[:,r]@h@y[:,s] for r in range(7) for s in range(7) if r!=s]).mean(0)
    estimate=complete_u(k)
    np.testing.assert_allclose(estimate,brute,atol=1e-12)
    covariance=np.cov(y,bias=False)
    corrected=y.mean(1)@h@y.mean(1)-np.einsum('kij,ji->k',h,covariance)/7
    np.testing.assert_allclose(estimate,corrected,atol=1e-12)
    assert abs(estimate[:4].sum()-estimate[-1])<1e-12


def test_schedule_dose_conservation():
    for regime,dose in itertools.product(['single_shot','burst','continuous'],[.05,.5,.95]):
        counts=[int(1000*abm.get_seed_fraction_at_step(regime,t,50,.05,False,dose)) for t in range(50)]
        assert sum(counts)==round(1000*dose)


def test_second_model_no_input_is_exact_zero_relative_to_control():
    task=dict(suite='E3',topology='BA',replication=999,stubbornness=.2,dose=0.,
              seeding_regime='none',strength=0.,run_id='test-control')
    result=run_fj(task)
    assert result['mean_total_shift']==result['mean_signed_shift']==result['conversion_rate']==0


def test_complete_design_counts_and_ids():
    for suite,n in [('E1',6480),('E2',9600),('E3',4560)]:
        tasks=make_tasks(suite)
        assert len(tasks)==n
        assert len(set(t['run_id'] for t in tasks))==n


def test_random_selector_preserves_legacy_at_nonzero_interior_parameters():
    graph=nx.path_graph(1000)
    for budget,compliance in itertools.product([.05,.2],[.5,1.]):
        first=abm.init_agents(1000,10)
        second=abm.init_agents(1000,10)
        ORIGINAL_INOCULATION(graph,first,'random',budget,.2,compliance,41)
        corrected_inoculation(graph,second,'random',budget,.2,compliance,41)
        np.testing.assert_array_equal(first.susceptibility,second.susceptibility)


def test_new_replication_streams_are_distinct_but_pair_within_scenario():
    assert seeds_for('E1','BA',0)==seeds_for('E1','BA',0)
    assert seeds_for('E1','BA',0)!=seeds_for('E1','BA',1)
    assert seeds_for('E1','BA',0)!=seeds_for('E2','BA',0)


def test_independent_contrast_estimator_and_bootstrap_match_projection_implementation():
    from validate_estimator import contrast_system, estimate, bootstrap
    from experiments import bootstrap_u
    cells,X,W=contrast_system()
    _,h=projection_games(pd.DataFrame(cells,columns=list('ABCD')),list('ABCD'))
    rng=np.random.default_rng(4321)
    mixing=rng.normal(size=(16,16))
    y=rng.normal(size=(16,1))+mixing@rng.normal(size=(16,10))
    z=X.T@y/16
    u,ins,se=estimate(z,W)
    k=kernels(y,h)
    np.testing.assert_allclose(u,complete_u(k)[:4],atol=1e-12)
    np.testing.assert_allclose(ins,y.mean(1)@h[:4]@y.mean(1),atol=1e-12)
    counts=rng.multinomial(10,np.ones(10)/10,51)
    np.testing.assert_allclose(bootstrap(z,W,counts),bootstrap_u(k,counts)[:,:4],atol=1e-12)


def test_cached_network_and_archived_new_runs_replay():
    from experiments import ROOT, ORIGINAL_BUILD, cached_build, execute_task
    for topo in ['BA','ER','WS','SBM']:
        agents=abm.init_agents(1000,91)
        a=ORIGINAL_BUILD(topo,1000,agents.opinions,18)
        b=cached_build(topo,1000,agents.opinions,18)
        assert list(a.nodes())==list(b.nodes())
        assert list(a.edges())==list(b.edges())
    for suite in ['E1','E2','E3']:
        tasks=make_tasks(suite)
        data=pd.read_csv(ROOT/'results'/f'{suite}_runs.csv.gz').set_index('run_id')
        for index in [0,len(tasks)//2,len(tasks)-1]:
            task=tasks[index]
            result=execute_task(task)
            stored=data.loc[task['run_id']]
            for key in ['mean_total_shift','mean_signed_shift','conversion_rate','mean_trajectory_shift']:
                np.testing.assert_allclose(result[key],stored[key],atol=1e-12,rtol=1e-10)


def test_seed_namespace_matches_archived_manifests():
    from experiments import ROOT
    for suite in ['E1', 'E2', 'E3']:
        records = pd.read_csv(ROOT / 'results' / f'{suite}_seed_manifest.csv')
        columns = ['topology', 'replication', 'master_seed', 'seed_net',
                   'seed_attr', 'seed_seeding', 'seed_sharing', 'seed_noise', 'seed_inoc']
        records = records[columns].drop_duplicates()
        for row in records.itertuples(index=False):
            actual = seeds_for(suite, row.topology, row.replication)
            for name in columns[2:]:
                assert actual[name] == getattr(row, name), (suite, row.topology, row.replication, name)
