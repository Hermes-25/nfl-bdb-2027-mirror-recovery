"""R8 supplement: exact canonical OL population with explicit missing-curve gate.

Reconstructs the original scalar recipe from preserved functions, then compares
on the immutable38-player common folds. Missing curve features remain missing
until fold-training imputation; missing NFL pressure labels are never zeros.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_functional import (load_data,extract_curves,BASE,FAMILY,CURVE,SEED,ALPHAS,
                            id_folds,ridge,brier,bootstrap_difference,make_figures,safe_savefig)
from legacy_scalar import build_features

MODES=['constant','conventional','original_scalar_family','missingness_gate','task_demands_missing','curve2_missing']


def feature_pair38(train,test,attempts,mode):
    columns=BASE[:]
    if mode=='original_scalar_family':columns+=['original_'+x for x in FAMILY]
    if mode=='missingness_gate':columns+=['curve_missing']
    if mode in ['task_demands_missing','curve2_missing']:columns+=FAMILY[1:]+['curve_missing']
    xx,xt=train[columns].to_numpy(float),test[columns].to_numpy(float)
    if mode=='curve2_missing':
        means=attempts.groupby('player_id')[CURVE].mean()
        curves=means.reindex(train.player_id).to_numpy(float)
        held=means.reindex(test.player_id).to_numpy(float)
        med=np.nanmedian(curves,axis=0);med=np.where(np.isfinite(med),med,0)
        curves,held=np.where(np.isnan(curves),med,curves),np.where(np.isnan(held),med,held)
        pca=PCA(n_components=2,svd_solver='full').fit(curves)
        xx=np.column_stack([xx,pca.transform(curves)]);xt=np.column_stack([xt,pca.transform(held)])
    return xx,xt


def canonical_nested(cohort,attempts,foldmap):
    cohort=cohort.sort_values(['draft_year','nfl_id']).reset_index(drop=True)
    outer=id_folds(cohort,foldmap['outer_folds'])
    prediction={m:np.full(len(cohort),np.nan) for m in MODES};records=[];fold_id=np.full(len(cohort),-1,int)
    for fold,(ii,jj) in enumerate(outer):
        train,test=cohort.iloc[ii].reset_index(drop=True),cohort.iloc[jj].reset_index(drop=True)
        inner=id_folds(train,foldmap['outer_folds'][fold]['inner_folds'])
        fold_id[jj]=fold;prediction['constant'][jj]=train.rate.mean()
        record={'outer_fold':fold,'train_ids':train.nfl_id.astype(int).tolist(),'test_ids':test.nfl_id.astype(int).tolist(),
                'inner_folds':foldmap['outer_folds'][fold]['inner_folds'],'selection':{}}
        for mode in MODES[1:]:
            losses=[]
            for alpha in ALPHAS:
                pp=np.full(len(train),np.nan)
                for aa,bb in inner:
                    a,b=train.iloc[aa],train.iloc[bb]
                    xx,xt=feature_pair38(a,b,attempts,mode)
                    pp[bb]=ridge(xx,a.rate.to_numpy(float),xt,alpha)
                losses.append(float(brier(train.rate.to_numpy(float),pp).mean()))
            alpha=sorted(zip(losses,ALPHAS),key=lambda z:(z[0],-z[1]))[0][1]
            xx,xt=feature_pair38(train,test,attempts,mode)
            prediction[mode][jj]=ridge(xx,train.rate.to_numpy(float),xt,alpha)
            record['selection'][mode]={'alpha':alpha,'inner_brier':losses}
        records.append(record)
    y=cohort.rate.to_numpy(float)
    metrics={m:{'equal_player_brier':float(brier(y,pp).mean()),'equal_player_mae':float(np.abs(y-pp).mean()),
                'opportunity_weighted_brier':float(np.average(brier(y,pp),weights=cohort.n))} for m,pp in prediction.items()}
    return cohort,fold_id,prediction,metrics,records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--fold-map',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    results,tracking,outcomes=load_data(args.raw_dir,[2023,2024])
    legacy,_,_,_=build_features(tracking)
    legacy=legacy[['draft_year','nfl_id']+FAMILY].rename(columns={c:'original_'+c for c in FAMILY})
    attempts,events,audit=extract_curves(tracking)
    new=attempts.groupby(['draft_year','nfl_id'],as_index=False)[FAMILY].median()
    cohort=results.merge(legacy,on=['draft_year','nfl_id'],validate='one_to_one').merge(
        outcomes,on=['draft_year','nfl_id'],validate='one_to_one')
    cohort=cohort[cohort.n.ge(50)].merge(new,on=['draft_year','nfl_id'],how='left',validate='one_to_one')
    cohort['player_id']=cohort.nfl_id.astype(int).astype(str)
    cohort['curve_missing']=cohort.normalized_deficit.isna().astype(float)
    foldmap=json.loads(args.fold_map.read_text())
    cc,fold_id,prediction,metrics,records=canonical_nested(cohort,attempts,foldmap)
    assert len(cc)==38
    result={'status':'executed_exploratory_R8_canonical38_supplement','years':[2023,2024],
            'eligible_players':len(cc),'charted_observations':int(cc.n.sum()),'pressures':int(cc.k.sum()),
            'curve_available_players':int(cc.curve_missing.eq(0).sum()),'curve_missing_ids':cc.loc[cc.curve_missing.eq(1),'nfl_id'].astype(int).tolist(),
            'fold_map_sha256':hashlib.sha256(args.fold_map.read_bytes()).hexdigest(),
            'metrics':metrics,'comparisons_vs_conventional':{m:bootstrap_difference(cc.rate.to_numpy(float),prediction['conventional'],prediction[m]) for m in MODES[2:]},
            'curve_increment_over_task_and_missing_gate':bootstrap_difference(cc.rate.to_numpy(float),prediction['task_demands_missing'],prediction['curve2_missing']),
            'same_session_repeatability':'Not estimable: every official player has only one mirror attempt; no EB forecast advertised.',
            'missing_policy':'One missing curve retains player in validation: explicit availabilityindicator; imputation and PCA fit in inner/outer training only.',
            'interpretation':'Development diagnostic only; original2025 negative result preserved and no new holdout claim.',
            'original_scalar_recipe':'Exact prior extraction functions retained in legacy_scalar.py, without original source mutation.',
            'limits':['Shared game/draft-cohort dependence remains.','Fixed-prediction bootstrap omits full refits, research selection and detector uncertainty.',
                      'One missingcurve example cannot train a generalizable availability effect.','Normalized curves reveal execution/task variation, not reaction time, forces or causal ability.']}
    (out/'CANONICAL38_RESULT.json').write_text(json.dumps(result,indent=2))
    (out/'FOLD_RECORDS.json').write_text(json.dumps(records,indent=2))
    rows=[]
    for i,row in cc.iterrows():
        rows.append({'player_id':row.player_id,'draft_year':int(row.draft_year),'outer_fold':int(fold_id[i]),
                     'n':int(row.n),'k':int(row.k),'rate':float(row.rate),'curve_missing':int(row.curve_missing),
                     **{m:float(pp[i]) for m,pp in prediction.items()}})
    (out/'OOF_PREDICTIONS.json').write_text(json.dumps(rows,indent=2))
    make_figures(attempts,events,cc,out)
    fig,ax=plt.subplots(figsize=(7.5,4.3),constrained_layout=True)
    vals=[metrics[m]['equal_player_brier'] for m in MODES]
    ax.scatter(vals,np.arange(len(MODES)),color='#126d9a',s=65)
    ax.set_yticks(np.arange(len(MODES)),['Constant','Conventional','Original scalar family','Missing-curve indicator','Task demands +indicator','Two curve components +controls'])
    ax.set(xlabel='Nested equal-player Brier loss (lower is better)',title='All38rookie OL: missing curves stay in evaluation')
    ax.invert_yaxis();safe_savefig(fig,out/'canonical38_baselines.png',dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
