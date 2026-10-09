"""OL mirror curve extraction and nested player-level exploratory comparison.

Usage: python run_functional.py --raw-dir /path/to/data --output-dir /path/to/out
Portable Python 3; requires numpy/pandas/scipy/scikit-learn/matplotlib.
Raw observations are never written to outputs. New results are development
evidence; the original entry already inspected 2025.
"""
from __future__ import annotations
import argparse, hashlib, json,io,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functional_eb import FunctionalEB, balanced_repeatability

BASE = ["combine_height", "combine_weight", "forty", "ten_yd_split"]
FAMILY = ["normalized_deficit", "incoming_speed", "lateral_amplitude"]
TIME_GRID = np.round(np.linspace(-0.4, 0.6, 11), 3)
CURVE = [f"c{i:02d}" for i in range(len(TIME_GRID))]
SEED = 87007
ALPHAS = [1.0, 10.0, 100.0]


def safe_savefig(figure,path,dpi=160):
    """Render completely in memory, verify PNG, then replace output atomically."""
    from PIL import Image
    buffer=io.BytesIO();figure.savefig(buffer,format='png',dpi=dpi)
    data=buffer.getvalue()
    Image.open(io.BytesIO(data)).verify()
    temporary=path.with_name(path.name+f'.{os.getpid()}.pending')
    temporary.write_bytes(data);temporary.replace(path)


def load_data(raw_dir, years):
    d = Path(raw_dir)
    players = pd.read_csv(d / "players.csv")
    players = players[players.draft_year.isin(years)]
    results = pd.read_csv(d / "combine_results.csv")
    results = results[results.draft_year.isin(years) & results.combine_position.eq("OL")]
    ids = set(results.nfl_id)
    if results.duplicated(["draft_year", "nfl_id"]).any():
        raise ValueError("Duplicate Combine participant key")
    games = pd.read_csv(d / "games.csv")
    games = games[games.season.isin(years) & games.season_type.eq("REG")]
    tracking_parts = []
    for q in pd.read_csv(d / "combine_tracking.csv", chunksize=200000, low_memory=False):
        q = q[q.draft_year.isin(years) & q.nfl_id.isin(ids) & q.entity_type.eq("PLAYER") &
              q.drill_name.str.replace("-", "_", regex=False).eq("PASS_PRO_MIRROR_DRILL")]
        if len(q): tracking_parts.append(q)
    tracking = pd.concat(tracking_parts, ignore_index=True)
    observed_parts = []
    for q in pd.read_csv(d / "player_play.csv", chunksize=100000, low_memory=False):
        q = q[q.nfl_id.isin(ids) & q.game_id.isin(set(games.game_id))]
        q = q.merge(players[["nfl_id", "draft_year"]], on="nfl_id", validate="many_to_one").merge(
            games[["game_id", "season"]], on="game_id", validate="many_to_one")
        q = q[q.season.eq(q.draft_year) & q.pressure_allowed.notna() &
              q.lined_up_position.isin(["C", "G", "T"]) &
              q.team_abbr.eq(q.possession_team) & q.play_nullified_by_penalty.eq("N")]
        labels = q.pressure_allowed.astype(str).str.lower()
        if not set(labels) <= {"true", "false"}: raise ValueError("Unexpected pressure labels")
        q = q.assign(pressure=labels.eq("true").astype(int))
        if len(q): observed_parts.append(q[["draft_year", "nfl_id", "game_id", "play_id", "pressure"]])
    observed = pd.concat(observed_parts, ignore_index=True)
    if observed.duplicated(["game_id", "play_id", "nfl_id"]).any():
        raise ValueError("Duplicate eligible player-play key")
    outcomes = observed.groupby(["draft_year", "nfl_id"], as_index=False).agg(
        n=("pressure", "size"), k=("pressure", "sum"))
    outcomes["rate"] = outcomes.k / outcomes.n
    return results, tracking, outcomes


def extract_curves(tracking, smoothing_window=5, exit_threshold=1.0):
    key_columns = ["draft_year", "event_id", "nfl_id", "attempt"]
    if tracking.duplicated(key_columns + ["time"]).any():
        raise ValueError("Duplicate tracking timestamp")
    if not tracking.x.between(0,120).all() or not tracking.y.between(0,53.5).all():
        raise ValueError("Coordinates outside documented football-field yard bounds")
    attempts, events, audit = [], [], []
    for key, group in tracking.groupby(key_columns, sort=True):
        group = group.sort_values("time")
        sec = (pd.to_datetime(group.time) - pd.to_datetime(group.time.iloc[0])).dt.total_seconds().to_numpy()
        x, y = group.x.to_numpy(float), group.y.to_numpy(float)
        quality = "ok"
        exits = np.flatnonzero((np.abs(x - np.median(x[:min(10, len(x))])) > exit_threshold) & (sec > 0.5))
        stop = int(exits[0]) if len(exits) else len(x)
        if stop < 15 or np.any(np.abs(np.diff(sec[:stop]) - 0.1) > 0.015): quality = "short_or_non_10hz"
        if not (np.isfinite(x).all() and np.isfinite(y).all()): quality = "nonfinite_coordinates"
        attempt_id = "|".join(map(str, key))
        player_id = str(int(key[2]))
        local = []
        if quality == "ok":
            smooth = savgol_filter(y[:stop], smoothing_window, 2)
            velocity = np.gradient(smooth, sec[:stop])
            signs = np.where(np.abs(velocity) > 0.05, np.sign(velocity), 0)
            nz = np.flatnonzero(signs)
            changes = [int(round((i+j)/2)) for i,j in zip(nz[:-1], nz[1:])
                       if signs[i]*signs[j] < 0 and j-i <= 10]
            last = -100
            for index in changes:
                # Do not extrapolate truncated pre/post windows into valid curves.
                if index < 5 or index+6 >= stop or index-last < 5: continue
                pre = float(np.median(velocity[index-5:index-1]))
                post = float(np.median(velocity[index+1:index+5]))
                if abs(pre) < 0.5 or abs(post) < 0.5 or pre*post >= 0: continue
                incoming = abs(pre)
                direction = np.sign(post)
                curve = np.interp(sec[index] + TIME_GRID, sec[:stop], direction*velocity) / incoming
                deficit = float(np.trapezoid(np.maximum(0, incoming - direction*velocity[index:index+6]),
                                             sec[index:index+6]-sec[index]) / incoming)
                row = dict(zip(key_columns, key), player_id=player_id, attempt_id=attempt_id,
                           reversal_time_seconds=float(sec[index]), incoming_speed=incoming,
                           normalized_deficit=deficit, direction=int(direction))
                row.update(dict(zip(CURVE, map(float, curve))))
                local.append(row); events.append(row); last = index
        if len(local) < 2 and quality == "ok": quality = "fewer_than_two_complete_curve_turns"
        audit.append(dict(zip(key_columns, key), quality=quality, frames=len(group),
                          mirror_frames=stop, complete_curve_turns=len(local)))
        if quality != "ok": continue
        event_frame = pd.DataFrame(local)
        row = dict(zip(key_columns, key), player_id=player_id, attempt_id=attempt_id,
                   n_turns=len(local), incoming_speed=float(event_frame.incoming_speed.median()),
                   normalized_deficit=float(event_frame.normalized_deficit.quantile(0.75)),
                   lateral_amplitude=float(np.percentile(y[:stop], 90) - np.percentile(y[:stop], 10)))
        row.update(dict(zip(CURVE, event_frame[CURVE].median().to_numpy(float))))
        attempts.append(row)
    return pd.DataFrame(attempts), pd.DataFrame(events), pd.DataFrame(audit)


def brier(rate, predictions):
    return rate*(1-predictions)**2 + (1-rate)*predictions**2


def ridge(train_x, train_y, test_x, alpha):
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0)
    train_x = np.where(np.isnan(train_x), median, train_x)
    test_x = np.where(np.isnan(test_x), median, test_x)
    mean, sd = train_x.mean(0), train_x.std(0)
    sd = np.where(sd > 1e-12, sd, 1)
    xx, xt = (train_x-mean)/sd, (test_x-mean)/sd
    ymean = train_y.mean()
    coefficient = np.linalg.solve(xx.T @ xx + alpha*np.eye(xx.shape[1]), xx.T @ (train_y-ymean))
    return np.clip(ymean + xt @ coefficient, 0.001, 0.999)


def feature_pair(train, test, attempts, mode):
    columns = BASE + (FAMILY if mode == "scalar_family_complete_windows" else [])
    if mode in ("task_demands", "curve_pca2", "curve_eb2"):
        columns = BASE + FAMILY[1:]
    train_x, test_x = train[columns].to_numpy(float), test[columns].to_numpy(float)
    if mode in ("curve_pca2", "curve_eb2"):
        train_attempts = attempts[attempts.player_id.isin(set(train.player_id))]
        test_attempts = attempts[attempts.player_id.isin(set(test.player_id))]
        if mode == "curve_eb2":
            estimator = FunctionalEB(CURVE, 2).fit(train_attempts)
            ft, fv = estimator.transform(train_attempts), estimator.transform(test_attempts)
            pc = ["functional_eb_pc1", "functional_eb_pc2"]
            tx = train[["player_id"]].merge(ft, on="player_id", how="left")[pc].to_numpy(float)
            vx = test[["player_id"]].merge(fv, on="player_id", how="left")[pc].to_numpy(float)
        else:
            ft = train_attempts.groupby("player_id")[CURVE].mean()
            fv = test_attempts.groupby("player_id")[CURVE].mean()
            tx = ft.reindex(train.player_id).to_numpy(float)
            vx = fv.reindex(test.player_id).to_numpy(float)
            median = np.nanmedian(tx, axis=0)
            tx, vx = np.where(np.isnan(tx), median, tx), np.where(np.isnan(vx), median, vx)
            estimator = PCA(n_components=2, svd_solver="full").fit(tx)
            tx, vx = estimator.transform(tx), estimator.transform(vx)
        train_x, test_x = np.column_stack([train_x, tx]), np.column_stack([test_x, vx])
    return train_x, test_x


def id_folds(frame, records):
    ids=frame.nfl_id.to_numpy(int)
    universe=set(ids);seen=[];folds=[]
    for record in records:
        train_ids=set(map(int,record.get("train_ids",record.get("train_player_ids",[]))))
        test_ids=set(map(int,record.get("test_ids",record.get("test_player_ids",[]))))
        if not train_ids or not test_ids or train_ids & test_ids:
            raise ValueError("Empty or overlapping common player fold")
        if train_ids | test_ids != universe:
            raise ValueError("Common fold universe differs from complete-curve eligible cohort")
        train=np.flatnonzero(np.isin(ids,list(train_ids)));test=np.flatnonzero(np.isin(ids,list(test_ids)))
        folds.append((train,test));seen.extend(ids[test].tolist())
    if sorted(seen) != sorted(ids.tolist()):
        raise ValueError("Every eligible player must be tested exactly once")
    return folds


def nested_evaluate(cohort, attempts, fold_manifest=None):
    cohort = cohort.sort_values(["draft_year", "nfl_id"]).reset_index(drop=True)
    if not cohort.nfl_id.is_unique: raise ValueError("Player-level cohort has repeated identifiers")
    outer_records=None
    if fold_manifest is None:
        outer=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(cohort,cohort.draft_year))
    else:
        if isinstance(fold_manifest,list): outer_records=fold_manifest
        else:
            outer_records=next((fold_manifest[k] for k in ["outer_folds","folds","outer"] if k in fold_manifest),None)
        if outer_records is None: raise ValueError("Common map needs explicit outer and inner train/test player-ID records")
        outer=id_folds(cohort,outer_records)
        if len(outer)!=5: raise ValueError("Five outer folds required")
    fold_id=np.full(len(cohort),-1,int)
    modes = ["constant", "conventional", "task_demands", "scalar_family_complete_windows", "curve_pca2", "curve_eb2"]
    predictions = {mode: np.full(len(cohort), np.nan) for mode in modes}
    selections = []
    for f,(tr,te) in enumerate(outer):
        fold_id[te]=f
        train,test=cohort.iloc[tr].reset_index(drop=True),cohort.iloc[te].reset_index(drop=True)
        predictions["constant"][te]=train.rate.mean()
        if outer_records is not None:
            inner_records=outer_records[f].get("inner_folds",outer_records[f].get("inner"))
            if inner_records is None: raise ValueError("Immutable inner player folds required for matched comparison")
            inner=id_folds(train,inner_records)
            if len(inner)!=3: raise ValueError("Three inner folds required")
        else:
            inner=list(StratifiedKFold(3,shuffle=True,random_state=SEED+100+f).split(train,train.draft_year))
        for mode in modes[1:]:
            losses = []
            for alpha in ALPHAS:
                pp = np.full(len(train), np.nan)
                for ii, jj in inner:
                    a, b = train.iloc[ii], train.iloc[jj]
                    xx, xt = feature_pair(a, b, attempts, mode)
                    pp[jj] = ridge(xx, a.rate.to_numpy(float), xt, alpha)
                losses.append(float(brier(train.rate.to_numpy(float), pp).mean()))
            best = sorted(zip(losses, ALPHAS), key=lambda z:(z[0], -z[1]))[0][1]
            xx, xt = feature_pair(train, test, attempts, mode)
            predictions[mode][te] = ridge(xx, train.rate.to_numpy(float), xt, best)
            selections.append({"outer_fold":f,"mode":mode,"alpha":best,"inner_brier":losses,
                               "train_ids":train.nfl_id.astype(int).tolist(),"test_ids":test.nfl_id.astype(int).tolist(),
                               "inner_folds":[{"train_ids":train.iloc[ii].nfl_id.astype(int).tolist(),
                                               "test_ids":train.iloc[jj].nfl_id.astype(int).tolist()} for ii,jj in inner]})
    y = cohort.rate.to_numpy(float)
    metrics = {mode:{"equal_player_brier":float(brier(y, pp).mean()),
                     "equal_player_mae":float(np.abs(y-pp).mean()),
                     "opportunity_weighted_brier":float(np.average(brier(y, pp), weights=cohort.n))}
               for mode,pp in predictions.items()}
    return cohort, fold_id, predictions, metrics, selections


def bootstrap_difference(y, reference, alternative, seed=SEED, n_boot=5000):
    differences = brier(y, reference) - brier(y, alternative)
    rng = np.random.default_rng(seed)
    boot = differences[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(1)
    return {"reference_minus_alternative":float(differences.mean()),
            "conditional_fixed_predictions_player_bootstrap_95":np.quantile(boot,[0.025,0.975]).tolist(),
            "limitations":"Does not refit folds, account for shared games or correct model search; exploratory diagnostic."}


def repeatability_report(attempts, value_column, n_boot=1000):
    report=balanced_repeatability(attempts,value_column)
    if not report.get("estimable"): return report
    pairs=[]
    for _,group in attempts.groupby("player_id",sort=True):
        sort_columns=[c for c in ["event_id","attempt"] if c in group] or ["attempt_id"]
        values=group.sort_values(sort_columns)[value_column].dropna().to_numpy(float)
        if len(values)>=2: pairs.append(values[:2])
    values=np.array(pairs);rng=np.random.default_rng(SEED)
    draws=values[rng.integers(0,len(values),size=(n_boot,len(values)))]
    means=draws.mean(2);grand=draws.mean((1,2))
    between=2*((means-grand[:,None])**2).sum(1)/(len(values)-1)
    within=((draws-means[:,:,None])**2).sum((1,2))/len(values)
    valid=between+within>1e-12
    icc=(between[valid]-within[valid])/(between[valid]+within[valid])
    te=np.std(draws[:,:,1]-draws[:,:,0],axis=1,ddof=1)/np.sqrt(2)
    report.update({"paired_player_bootstrap_icc_95":np.quantile(icc,[.025,.975]).tolist(),
                   "paired_player_bootstrap_typical_error_95":np.quantile(te,[.025,.975]).tolist(),
                   "bootstrap_replicates":n_boot,
                   "interpretation":"Within-session attempt consistency; shared instructions, coaches, sequence, and tracking error are not disentangled."})
    return report


def make_figures(attempts, events, cohort, out):
    usable = events[events.attempt_id.isin(set(attempts.attempt_id))]
    # Pick most repeated athlete, breaking ties by numeric ID rather than NFL outcome.
    counts = attempts.groupby("player_id").size()
    example = sorted(counts.index, key=lambda p:(-counts[p], int(p)))[0]
    figure, axes = plt.subplots(1,2, figsize=(11,4.2), constrained_layout=True)
    curves = attempts[attempts.player_id.eq(example)]
    selected_events=usable[usable.player_id.eq(example)]
    for _, row in selected_events.iterrows():
        axes[0].plot(TIME_GRID,row[CURVE].to_numpy(float),color="#9aabba",alpha=.65,lw=1)
    for _,row in curves.iterrows():
        axes[0].plot(TIME_GRID,row[CURVE].to_numpy(float),color="#126d9a",lw=2.5,
                     label=f"Attempt {row['attempt']} median; {int(row['n_turns'])} turns")
    axes[0].axvline(0,color="black",lw=0.8); axes[0].axhline(0,color="gray",lw=0.8)
    axes[0].set(xlabel="Seconds from kinematic reversal", ylabel="Outgoing-signed velocity / entry speed",
                title=f"One OL: individual turns and attempt median\nNFL ID {example}; selection ignores NFL outcome")
    axes[0].legend(fontsize=8)
    pairs = []
    for pid, group in attempts.groupby("player_id"):
        if len(group) >= 2:
            values = group.sort_values("attempt_id").normalized_deficit.iloc[:2].to_numpy()
            pairs.append(values)
    if pairs:
        pairs=np.array(pairs); axes[1].scatter(pairs[:,0],pairs[:,1],alpha=0.7)
        lim=[float(pairs.min()),float(pairs.max())]; axes[1].plot(lim,lim,color="gray",ls="--")
        axes[1].set(xlabel="First attempt recovery cost (seconds)",ylabel="Second attempt recovery cost (seconds)",
                    title="Same-session repeatability, not stable talent")
    else:
        points=axes[1].scatter(selected_events.reversal_time_seconds,selected_events.normalized_deficit,
                               c=selected_events.incoming_speed,cmap="viridis",s=80,edgecolor="black",lw=.5)
        figure.colorbar(points,ax=axes[1],label="Entry speed (yards/second)")
        axes[1].set(xlabel="Reversal time within attempt (seconds)",ylabel="Turn recovery cost (seconds)",
                    title="Turn-to-turn task variation\nOnly one attempt: reliability is unidentifiable")
    safe_savefig(figure,out/"functional_measurement.png",dpi=160);plt.close(figure)
    quantiles=np.quantile(attempts[CURVE].to_numpy(float),[.1,.5,.9],axis=0)
    figure,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.fill_between(TIME_GRID,quantiles[0],quantiles[2],alpha=.25,label="Attempt 10th–90th percentiles")
    ax.plot(TIME_GRID,quantiles[1],label="Attempt median")
    ax.axvline(0,color="black",lw=.8);ax.axhline(0,color="gray",lw=.8)
    ax.set(xlabel="Seconds from kinematic reversal",ylabel="Outgoing-signed velocity / entry speed",
           title="OL mirror movement curves: braking and recovery")
    ax.legend();safe_savefig(figure,out/"population_curve.png",dpi=160);plt.close(figure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"round7")
    parser.add_argument("--years",nargs="+",type=int,default=[2023,2024])
    parser.add_argument("--fold-manifest","--fold-map",type=Path)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    results,tracking,outcomes=load_data(args.raw_dir,args.years)
    attempts,events,audit=extract_curves(tracking)
    scalar=attempts.groupby(["draft_year","nfl_id"],as_index=False)[FAMILY].median()
    cohort=results.merge(scalar,on=["draft_year","nfl_id"],validate="one_to_one").merge(
        outcomes,on=["draft_year","nfl_id"],validate="one_to_one")
    cohort=cohort[cohort.n.ge(50)].copy();cohort["player_id"]=cohort.nfl_id.astype(int).astype(str)
    manifest=json.loads(args.fold_manifest.read_text()) if args.fold_manifest else None
    cohort,folds,predictions,metrics,selections=nested_evaluate(cohort,attempts,manifest)
    noise=FunctionalEB(CURVE,2).fit(attempts[attempts.player_id.isin(set(cohort.player_id))])
    sensitivity={}
    for window in [7,9]:
        aa,_,qq=extract_curves(tracking,smoothing_window=window)
        base=attempts.groupby("player_id").normalized_deficit.median()
        alt=aa.groupby("player_id").normalized_deficit.median()
        match=pd.concat([base,alt],axis=1,join="inner").dropna()
        sensitivity[str(window)]={"players_matched":len(match),"rank_correlation":float(spearmanr(match.iloc[:,0],match.iloc[:,1]).statistic),
                                  "median_absolute_change_seconds":float(np.median(np.abs(match.iloc[:,0]-match.iloc[:,1]))),
                                  "attempt_quality_counts":qq.quality.value_counts().to_dict()}
    output={"status":"exploratory_development_new_methods;2025_previously_observed",
            "years":args.years,"eligible_players":len(cohort),"charted_opportunities":int(cohort.n.sum()),
            "charted_pressures":int(cohort.k.sum()),"tracking_frames":len(tracking),
            "attempts":len(attempts),"turn_events":len(events),"quality_counts":audit.quality.value_counts().to_dict(),
            "metrics":metrics,"alpha_selections":selections,
            "nested_scheme":{"outer":5,"inner":3,"seed":SEED,"stratified_by":"draft_year","grouped_by":"nfl_id",
                             "fold_assignment_source":"immutable_root_player_ID_map" if args.fold_manifest else "candidate_local_generated_not_yet_matched",
                             "fold_map_sha256":hashlib.sha256(args.fold_manifest.read_bytes()).hexdigest() if args.fold_manifest else None},
            "same_session_repeatability":repeatability_report(attempts,"normalized_deficit"),
            "measurement_noise_estimable":noise.noise_estimable_,"training_repeated_players":noise.repeated_players_,
            "curve_eigenvalues":noise.eigenvalues_.tolist(),"detector_sensitivity":sensitivity,
            "comparisons":{m:bootstrap_difference(cohort.rate.to_numpy(float),predictions["conventional"],predictions[m])
                           for m in ["scalar_family_complete_windows","curve_pca2","curve_eb2"]},
            "increment_over_task_demands":{m:bootstrap_difference(cohort.rate.to_numpy(float),predictions["task_demands"],predictions[m])
                                           for m in ["curve_pca2","curve_eb2"]},
            "target":"Explicit non-null charted rookie REG pressures; non-nullified possession C/G/T rows; minimum50; equal-player assessment.",
            "limitations":["Turn events from one attempt are dependent; one median curve per attempt.",
                           "Scalar family follows the prior recipe but is measured on complete -.4..+.6s turn windows; it is not an exact rerun of the original scalar detector.",
                           "Repeated attempts share a session and unknown coach/sequence demands.",
                           "Player folds do not remove shared game dependence or draft-cohort shift.",
                           "No stimulus timing, foot contacts or force plates; kinematics cannot identify reaction time or braking impulse.",
                           "Fixed normalization windows and EB assumptions need sensitivity; this is not exact PACE.",
                           "No confirmatory claim after prior cohort use; bootstrap is conditional on fitted OOF predictions."]}
    (args.output_dir/"ROUND_RESULT.json").write_text(json.dumps(output,indent=2))
    prediction_rows=[]
    for i,row in cohort.iterrows():
        prediction_rows.append({"player_id":row.player_id,"draft_year":int(row.draft_year),"outer_fold":int(folds[i]),
                                "n":int(row.n),"k":int(row.k),"rate":float(row.rate),
                                **{m:float(pp[i]) for m,pp in predictions.items()}})
    (args.output_dir/"OOF_PREDICTIONS.json").write_text(json.dumps(prediction_rows,indent=2))
    (args.output_dir/"EXTRACTION_AUDIT.json").write_text(audit.to_json(orient="records",indent=2))
    make_figures(attempts,events,cohort,args.output_dir)
    print(json.dumps(output,indent=2))


if __name__=="__main__": main()
