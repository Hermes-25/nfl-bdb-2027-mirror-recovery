"""Executed R8: fixed37-player universe; detector sensitivity and cohort transfer.

Variants never change the target population. If a detector loses a player's
features, that player remains in validation with fold-training median imputation.
This is exploratory;2025 is historically observed and excluded from this analysis.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_functional import (load_data,extract_curves,BASE,FAMILY,CURVE,SEED,ALPHAS,
                            nested_evaluate,feature_pair,ridge,brier,bootstrap_difference,make_figures,safe_savefig)


def cross_cohort(cohort,attempts):
    modes=["constant","conventional","task_demands","scalar_family_complete_windows","curve_pca2"]
    output=[]
    for year in sorted(cohort.draft_year.unique()):
        train=cohort[cohort.draft_year.ne(year)].sort_values("nfl_id").reset_index(drop=True)
        test=cohort[cohort.draft_year.eq(year)].sort_values("nfl_id").reset_index(drop=True)
        inner=list(KFold(3,shuffle=True,random_state=SEED+int(year)).split(train))
        metric={"constant":{"brier":float(brier(test.rate.to_numpy(float),np.full(len(test),train.rate.mean())).mean())}}
        for mode in modes[1:]:
            scores=[]
            for alpha in ALPHAS:
                prediction=np.full(len(train),np.nan)
                for ii,jj in inner:
                    a,b=train.iloc[ii],train.iloc[jj]
                    xx,xt=feature_pair(a,b,attempts,mode)
                    prediction[jj]=ridge(xx,a.rate.to_numpy(float),xt,alpha)
                scores.append(float(brier(train.rate.to_numpy(float),prediction).mean()))
            alpha=sorted(zip(scores,ALPHAS),key=lambda z:(z[0],-z[1]))[0][1]
            xx,xt=feature_pair(train,test,attempts,mode)
            prediction=ridge(xx,train.rate.to_numpy(float),xt,alpha)
            metric[mode]={"brier":float(brier(test.rate.to_numpy(float),prediction).mean()),"alpha":alpha}
        output.append({"train_years":sorted(map(int,train.draft_year.unique())),"test_year":int(year),
                       "train_players":len(train),"test_players":len(test),"metrics":metric,
                       "interpretation":"Retrospective cross-cohort diagnostic; backward2024->2023 direction is not a forecast."})
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--fold-map",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    foldmap=json.loads(args.fold_map.read_text())
    ids=set(map(int,foldmap['outer_folds'][0]['train_ids']+foldmap['outer_folds'][0]['test_ids']))
    results,tracking,outcomes=load_data(args.raw_dir,[2023,2024])
    base=results.merge(outcomes,on=['draft_year','nfl_id'],validate='one_to_one')
    base=base[base.nfl_id.isin(ids)].copy();base['player_id']=base.nfl_id.astype(int).astype(str)
    assert len(base)==len(ids) and base.n.ge(50).all()
    records=[];primary=None
    for window in [5,7,9]:
        for crop in [.75,1.0,1.25]:
            attempts,events,quality=extract_curves(tracking,smoothing_window=window,exit_threshold=crop)
            scalar=attempts.groupby(['draft_year','nfl_id'],as_index=False)[FAMILY].median()
            cohort=base.merge(scalar,on=['draft_year','nfl_id'],how='left',validate='one_to_one')
            cc,folds,predictions,metrics,selections=nested_evaluate(cohort,attempts,foldmap)
            record={'smoothing_window_frames':window,'forward_crop_yards':crop,'fixed_evaluation_players':len(cc),
                    'players_with_usable_curve':int(cc.normalized_deficit.notna().sum()),
                    'feature_missing_players':cc.loc[cc.normalized_deficit.isna(),'nfl_id'].astype(int).tolist(),
                    'usable_attempts_all_OL':len(attempts),'quality_counts':quality.quality.value_counts().to_dict(),
                    'metrics':metrics,'curve_vs_conventional':bootstrap_difference(cc.rate.to_numpy(float),predictions['conventional'],predictions['curve_pca2']),
                    'curve_vs_task_demands':bootstrap_difference(cc.rate.to_numpy(float),predictions['task_demands'],predictions['curve_pca2']),
                    'prediction_imputation':'Missing detector features retained; all imputation fit inside inner/outer training folds'}
            records.append(record)
            print(json.dumps({k:record[k] for k in ['smoothing_window_frames','forward_crop_yards','players_with_usable_curve','curve_vs_conventional']}),flush=True)
            if window==5 and crop==1.0:
                primary=(cc,attempts,events,metrics,predictions)
    cc,attempts,events,metrics,predictions=primary
    make_figures(attempts,events,cc,out)
    transport=cross_cohort(cc,attempts)
    summary={'status':'executed_exploratory_detector_and_cross_cohort_audit','years':[2023,2024],
             'evaluation_players':len(cc),'charted_observations':int(cc.n.sum()),'pressures':int(cc.k.sum()),
             'fold_map_sha256':hashlib.sha256(args.fold_map.read_bytes()).hexdigest(),
             'variants':records,'cross_cohort':transport,
             'reliability':'Zero repeated attempts perplayer; between-attempt ICC/noise unidentifiable; EB equals ordinaryPCA fallback',
             'claim':'No validated forecast gain, stable-trait reliability or causal talent grade is established.'}
    (out/'ROUND8_RESULT.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    gains=np.array([record['curve_vs_conventional']['reference_minus_alternative'] for record in records])
    labels=[f"{r['smoothing_window_frames']}frames /{r['forward_crop_yards']}yd" for r in records]
    axes[0].barh(range(len(records)),gains,color=np.where(gains>=0,'#17846b','#a54837'))
    axes[0].axvline(0,color='black',lw=.8);axes[0].set_yticks(range(len(records)),labels)
    axes[0].set(xlabel='Conventional Brier minus curve Brier (higher favors curves)',title='Same37players across detector variants')
    order=['constant','conventional','task_demands','scalar_family_complete_windows','curve_pca2']
    vals=[metrics[m]['equal_player_brier'] for m in order]
    axes[1].scatter(vals,range(len(order)),color='#126d9a',s=60)
    axes[1].set_yticks(range(len(order)),['Constant','Conventional','+Task demands','+Scalar family','+Curve2components'])
    axes[1].set(xlabel='Nested equal-player Brier (lower is better)',title='Exploratory prediction: complexity does not win')
    safe_savefig(fig,out/'robustness_and_baselines.png',dpi=160);plt.close(fig)
    print(json.dumps({'cross_cohort':transport,'curve_gain_range':[float(gains.min()),float(gains.max())]},indent=2),flush=True)


if __name__=='__main__':main()
