from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml
from experiments import ROOT, abm, cached_build, corrected_inoculation, sha_file


def replay(task):
    abm.build_network=cached_build
    abm.apply_inoculation=corrected_inoculation
    row=abm.run_single(task['topology'],task['message_condition'],task['strength'],'hub',
                       task['seeding_regime'],task['simulation'],task['replication'],
                       seed_override=task['seeds'])
    row['source_row_index']=task['source_row_index']
    row['design']=task['design']
    return row


def run(source,workers=6):
    source=Path(source)
    start=time.perf_counter()
    data=pd.read_csv(source/'data/combined_runs_120960.csv.gz')
    registry=pd.read_csv(source/'analysis_outputs/configuration_replay_registry.csv')
    config_lookup={}
    for entry in registry.itertuples(index=False):
        config=yaml.safe_load((source/entry.config).read_text(encoding='utf-8'))
        for rep in range(entry.replication_start,entry.replication_start+entry.replications):
            config_lookup[entry.design,rep]=config
    seed_columns={'master_seed':'master_seed','seed_net':'network_seed','seed_attr':'initialization_seed',
                  'seed_seeding':'seeding_seed','seed_sharing':'sharing_seed','seed_noise':'noise_seed',
                  'seed_inoc':'inoculation_seed'}
    tasks=[]
    # Group by source cell to make network cache reuse across designs effective.
    hub=data[data.inoculation=='hub'].sort_values(['topology','message_condition','seeding_regime','replication','design'])
    for idx,row in hub.iterrows():
        config=config_lookup[row.design,int(row.replication)]
        seeds={key:int(row[col]) for key,col in seed_columns.items()}
        seeds['seed_sim']=None
        tasks.append(dict(source_row_index=int(idx),topology=row.topology,message_condition=row.message_condition,
                          seeding_regime=row.seeding_regime,replication=int(row.replication),design=row.design,
                          strength=config['design']['message_conditions'][row.message_condition],
                          simulation=config['simulation'],seeds=seeds))
    assert len(tasks)==40320
    path=ROOT/'results/E5_corrected_hub_runs.jsonl'
    if path.exists():
        raise FileExistsError(path)
    replacements=[]
    with open(path,'w',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(replay,tasks,chunksize=30),1):
            replacements.append(row)
            f.write(json.dumps(row)+'\n')
            if i%3000==0 or i==len(tasks):
                f.flush()
                print(f'E5: {i}/{len(tasks)} correction runs; {time.perf_counter()-start:.1f}s',flush=True)
    repl=pd.DataFrame(replacements).set_index('source_row_index')
    # Retain original schema, design IDs, seeds and metadata. Replace model outputs only.
    immutable={'design','topology','message_condition','inoculation','seeding_regime','replication',
               'master_seed','network_seed','initialization_seed','seeding_seed','sharing_seed','noise_seed','inoculation_seed'}
    shared=[c for c in repl.columns if c in data.columns and c not in immutable]
    corrected=data.copy()
    for column in shared:
        corrected.loc[repl.index,column]=repl[column]
    numeric=['mean_total_shift','mean_signed_shift','conversion_rate','total_direct_exposures','total_secondhand_exposures']
    contrasts=[]
    for outcome in numeric:
        contrasts.append(pd.DataFrame(dict(source_row_index=repl.index,
                                           design=data.loc[repl.index,'design'].to_numpy(),
                                           topology=data.loc[repl.index,'topology'].to_numpy(),
                                           replication=data.loc[repl.index,'replication'].to_numpy(),
                                           outcome=outcome,
                                           original=data.loc[repl.index,outcome].to_numpy(),
                                           corrected=corrected.loc[repl.index,outcome].to_numpy())))
    differences=pd.concat(contrasts,ignore_index=True)
    differences['difference']=differences.corrected-differences.original
    differences.to_csv(ROOT/'results/E5_run_differences.csv.gz',index=False,compression='gzip')
    repl.to_csv(ROOT/'results/E5_corrected_hub_runs.csv.gz',compression='gzip')
    corrected.to_csv(ROOT/'results/E5_corrected_archive_120960.csv.gz',index=False,compression='gzip')
    # Original point table is a compact auditable reference; keep its source hash.
    original_points=pd.read_csv(source/'analysis_outputs/complete_u_point_estimates.csv')
    original_points.to_csv(ROOT/'inputs/original_complete_u_point_estimates.csv',index=False)
    summary=dict(corrected_runs=len(repl),archive_rows=len(corrected),elapsed_seconds=time.perf_counter()-start,
                 original_archive_unchanged=sha_file(source/'data/combined_runs_120960.csv.gz')==
                    json.loads((ROOT/'validation/source_verification.json').read_text(encoding='utf-8'))['original_archive_sha256'],
                 nonhub_rows_unchanged=bool(data[data.inoculation!='hub'].equals(corrected[corrected.inoculation!='hub'])),
                 direct_exposure_unchanged=bool(np.array_equal(data.total_direct_exposures,corrected.total_direct_exposures)))
    (ROOT/'validation/E5_execution.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    run(args.source,args.workers)
