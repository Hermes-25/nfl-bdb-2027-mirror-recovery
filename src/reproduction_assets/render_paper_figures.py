"""Journal-style displays of previously verified results; no models are refitted.

The only computations are plotting-scale changes and previously specified
descriptive summaries. Raw tracking, when available, is used only to redraw the
same deterministically selected film-review example. Captions are external.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#176B86"
ORANGE = "#BB613C"
GREY = "#667078"
PALE = "#CFD5D9"
PURPLE = "#7B6499"
GOLD = "#D0AA5B"


def paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "normal",
        "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": True,
    })


def axis_style(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E5E8EA", linewidth=.65)
    ax.tick_params(length=3.5, width=.7)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=10)


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=.08)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def draw_selection(transport, sensitivity, frozen, output):
    """All roster totals, seven differences, and every archived sensitivity point."""
    paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), layout="constrained")
    ax = axes[0, 0]
    rows = transport["original_2025_archived_not_recomputed"]["cohort_coverage"]
    x = np.arange(3); width = .34
    ax.bar(x-width/2, [r["combine_OL"] for r in rows], width, color=PALE,
           label="Combine OL roster")
    ax.bar(x+width/2, [r["eligible_min50"] for r in rows], width, color=BLUE,
           label="Usable mirror + ≥50 charted observations")
    for i, r in enumerate(rows):
        ax.annotate(f'{r["eligible_min50"]}/{r["combine_OL"]}',
                    (i+width/2, r["eligible_min50"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xticks=x, xticklabels=["2023", "2024", "2025"],
           ylabel="Number of prospects", ylim=(0, 58))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    panel(ax, "a", "Eligibility by draft cohort")
    axis_style(ax, "y")

    labels = {"combine_height": "Height", "combine_weight": "Weight",
              "forty": "40-yard time", "ten_yd_split": "10-yard split",
              "normalized_deficit": "Relative recovery cost",
              "incoming_speed": "Entry lateral speed",
              "lateral_amplitude": "Lateral excursion"}
    ax = axes[0, 1]
    selection = transport["R7"]["selection_smd"]
    y = np.arange(len(selection))
    values = np.array([r["standardized_mean_difference"] for r in selection])
    ax.barh(y, values, color=BLUE, height=.68)
    ax.axvline(0, color=GREY, linewidth=.8)
    ax.set(yticks=y, yticklabels=[labels[r["feature"]] for r in selection],
           xlabel="(Retained − excluded mean) / all-prospect SD", xlim=(-.61, .67))
    ax.invert_yaxis()
    for yy, vv in zip(y, values):
        ax.annotate(f"{vv:+.2f}", (vv, yy), xytext=(4 if vv >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if vv >= 0 else "right", fontsize=8)
    panel(ax, "b", "Development-cohort selection")
    axis_style(ax)

    ax = axes[1, 0]
    primary = transport["R8"]["primary"]["combine_minus_mirror"]
    ax.scatter([primary*1000], [0], color=GREY, marker="D", s=35, zorder=4)
    rng = np.random.default_rng(989)
    for yy, name, color in [(1, "splits", PURPLE), (2, "delete", BLUE)]:
        values = np.array([r["combine_minus_mirror"] for r in sensitivity[name]])
        ax.scatter(values*1000, yy+rng.uniform(-.09, .09, len(values)),
                   s=18, alpha=.82, color=color, linewidths=0)
    draws = np.array([r["baseline_minus_mirror"] for r in sensitivity["boot"]])
    lo, hi = np.quantile(draws, [.025, .975]); median = np.median(draws)
    ax.errorbar(median*1000, 3, xerr=np.array([[median-lo], [hi-median]])*1000,
                fmt="o", markersize=4.5, color=ORANGE, capsize=3.5, linewidth=1.25)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.set(yticks=[0, 1, 2, 3],
           yticklabels=["Primary split", "20 split repeats", "38 player deletions",
                        "500 nested resamples"], ylim=(-.45, 3.5),
           xlabel="Conventional − mirror Brier (×10⁻³)")
    ax.invert_yaxis(); panel(ax, "c", "Exploratory model-selection stability")
    axis_style(ax)

    ax = axes[1, 1]
    delta = frozen["brier_improvement"]
    for yy, key, color in [(0, "fixed_state_player_bootstrap95", ORANGE),
                           (1, "pooled_training_and_evaluation_refit_bootstrap95", BLUE)]:
        lo, hi = frozen[key]
        ax.errorbar(delta*1000, yy, xerr=np.array([[delta-lo], [hi-delta]])*1000,
                    fmt="o", markersize=4.5, color=color, capsize=3.5, linewidth=1.25)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.set(yticks=[0, 1], yticklabels=["Fixed fitted states", "Train + test player refits"],
           ylim=(-.55, 1.55), xlabel="Conventional − mirror Brier (×10⁻³)")
    ax.invert_yaxis(); panel(ax, "d", "Original frozen 2025 comparison")
    axis_style(ax)
    save_figure(fig, output)


def draw_frozen(frozen, output):
    """Immutable 2025 losses, archived uncertainty scopes, and calibration means."""
    paper_style()
    assert frozen["evaluation_players"] == 19
    assert frozen["charted_opportunities"] == 7801
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.25), layout="constrained")
    keys = ["constant", "baseline", "enhanced"]
    names = ["Constant", "Conventional", "+ mirror"]
    colors = [GREY, BLUE, ORANGE]; y = np.arange(3)
    ax = axes[0]
    loss = np.array([frozen[k]["player_equal_brier"] for k in keys])
    ax.scatter(loss*1000, y, s=35, color=colors, zorder=4)
    for yy, vv in zip(y, loss):
        ax.annotate(f"{vv:.6f}", (vv*1000, yy), xytext=(5, 7),
                    textcoords="offset points", fontsize=8.5)
    ax.axvline(loss[0]*1000, color=GREY, linestyle=":", linewidth=1)
    ax.set(yticks=y, yticklabels=names, ylim=(-.6, 2.6),
           xlim=(83.35, 85.15), xlabel="Equal-player Brier loss (×10⁻³)")
    ax.invert_yaxis(); panel(ax, "a", "Forecast loss")
    axis_style(ax)

    ax = axes[1]; delta = frozen["brier_improvement"]
    for yy, key, color in [(0, "fixed_state_player_bootstrap95", ORANGE),
                           (1, "pooled_training_and_evaluation_refit_bootstrap95", BLUE)]:
        lo, hi = frozen[key]
        ax.errorbar(delta*1000, yy, xerr=np.array([[delta-lo], [hi-delta]])*1000,
                    fmt="o", markersize=4.5, color=color, capsize=3.5, linewidth=1.25)
    ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
    ax.set(yticks=[0, 1], yticklabels=["Fixed fitted states", "Train + test\nplayer refits"],
           ylim=(-.6, 1.6), xlabel="Conventional − mirror Brier (×10⁻³)")
    ax.invert_yaxis(); panel(ax, "b", "Two uncertainty scopes")
    axis_style(ax)

    ax = axes[2]
    means = np.array([frozen[k]["mean_prediction"] for k in keys])*100
    observed = frozen["baseline"]["observed_mean_rate"]*100
    ax.scatter(means, y, s=35, color=colors, zorder=4)
    for yy, vv in zip(y, means):
        ax.annotate(f"{vv:.2f}", (vv, yy), xytext=(5, 7),
                    textcoords="offset points", fontsize=8.5)
    ax.axvline(observed, color=GREY, linestyle="--", linewidth=.8,
               label=f"Observed: {observed:.2f}")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set(yticks=y, yticklabels=names, ylim=(-.6, 2.6), xlim=(8.65, 12.15),
           xlabel="Mean pressures per 100 charted observations")
    ax.invert_yaxis(); panel(ax, "c", "Mean forecast and observed rate")
    axis_style(ax)
    save_figure(fig, output)


def draw_count(counts, intervals, output, primary=None, interval_rows=None):
    """Use exact runtime row data when supplied; archived aggregates otherwise.

    Local aggregate-only preview shows point losses, not invented confidence
    intervals. The notebook can supply the already fitted row tables to retain
    the original four fixed-OOF bootstrap intervals exactly.
    """
    paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), layout="constrained")
    models = ["nested_conventional_ridge", "beta_binomial_conventional",
              "beta_binomial_full", "nested_full_ridge"]
    names = ["Conventional ridge", "Conventional\nbeta-binomial",
             "Mirror beta-binomial", "Mirror ridge"]
    ax = axes[0]; y = np.arange(4)
    if primary is not None:
        wide = primary.pivot(index="player_id", columns="model", values="prediction")
        target = primary[primary.model.eq("constant_equal_player")].set_index("player_id").loc[wide.index]
        rate = (target.y/target.n).to_numpy()
        samples = np.random.default_rng(87007).integers(0, len(wide), size=(5000, len(wide)))
        means=[]; limits=[]
        for model in models:
            delta = (rate-wide.constant_equal_player.to_numpy())**2-(rate-wide[model].to_numpy())**2
            means.append(delta.mean()); limits.append(np.quantile(delta[samples].mean(1), [.025, .975]))
        means = np.array(means); limits = np.array(limits)
        ax.errorbar(means*1000, y,
                    xerr=np.vstack([means-limits[:, 0], limits[:, 1]-means])*1000,
                    fmt="o", color=BLUE, capsize=3.5, markersize=4.5, linewidth=1.25)
        ax.axvline(0, color=GREY, linestyle="--", linewidth=.8)
        ax.set(xlabel="Constant − model Brier (×10⁻³)")
        panel(ax, "a", "Fixed-OOF comparison")
    else:
        scores=np.array([counts["metrics"][model]["equal_player_brier"] for model in models])
        ax.scatter(scores*1000, y, s=35, color=BLUE, zorder=4)
        base=counts["metrics"]["constant_equal_player"]["equal_player_brier"]
        ax.axvline(base*1000, color=GREY, linestyle="--", linewidth=.8,
                   label=f"Constant: {base:.6f}")
        for yy, vv in zip(y, scores):
            ax.annotate(f"{vv:.6f}", (vv*1000, yy), xytext=(4, 7),
                        textcoords="offset points", fontsize=8)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.set(xlim=(82.79, 83.02), xlabel="Equal-player Brier loss (×10⁻³)")
        panel(ax, "a", "Exploratory forecast loss")
    ax.set(yticks=y, yticklabels=names, ylim=(-.55, 3.6))
    ax.invert_yaxis(); axis_style(ax)

    rows=[intervals[k] for k in ["beta_binomial_constant", "beta_binomial_full"]]
    x=np.arange(2); width=.33
    ax=axes[1]
    left=ax.bar(x-width/2, [r["central90_count_coverage"]*100 for r in rows],
                width, color=ORANGE, label="Beta-binomial central 90%")
    right=ax.bar(x+width/2, [r["jackknife_style_alpha05_empirical_coverage"]*100 for r in rows],
                 width, color=BLUE, label="Jackknife+ style, α = 0.05")
    for bars in [left, right]:
        for b in bars:
            ax.annotate(f"{round(b.get_height()/100*38)}/38",
                        (b.get_x()+b.get_width()/2, b.get_height()), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(xticks=x, xticklabels=["Constant prior", "Mirror prior"], ylim=(0, 110),
           ylabel="Empirical player coverage (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.29), frameon=False, fontsize=8)
    panel(ax, "b", "Empirical interval coverage"); axis_style(ax, "y")

    ax=axes[2]
    for yy, r in enumerate(rows):
        central=r["median_central90_width"]*100
        jack=r["median_jackknife_style_alpha05_width"]*100
        ax.plot([central, jack], [yy, yy], color=PALE, linewidth=1.5)
        ax.scatter([central, jack], [yy, yy], color=[ORANGE, BLUE], s=35, zorder=4)
        for vv, offset in [(central, -15), (jack, 8)]:
            ax.annotate(f"{vv:.2f}", (vv, yy), xytext=(0, offset),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.set(yticks=x, yticklabels=["Constant prior", "Mirror prior"], ylim=(-.6, 1.6),
           xlim=(4.9, 13.45), xlabel="Median interval width (percentage points)")
    ax.invert_yaxis(); panel(ax, "c", "Interval width"); axis_style(ax)
    save_figure(fig, output)


def draw_film(tracking, card, output):
    """Redraw the exact first retained turn without changing its detector."""
    from scipy.signal import savgol_filter
    import pandas as pd
    paper_style()
    turn=card["turns"][0]
    group=tracking[tracking.nfl_id.eq(card["nfl_id"]) &
                   tracking.event_id.astype(str).eq(turn["event_id"]) &
                   tracking.attempt.eq(turn["attempt"])].sort_values("time")
    sec=(pd.to_datetime(group.time)-pd.to_datetime(group.time.iloc[0])).dt.total_seconds().to_numpy()
    x,y=group.x.to_numpy(float),group.y.to_numpy(float)
    origin_x=np.median(x[:min(10,len(x))]); origin_y=y[0]
    exits=np.flatnonzero((np.abs(x-origin_x)>1)&(sec>.5))
    stop=int(exits[0]) if len(exits) else len(x)
    velocity=np.gradient(savgol_filter(y[:stop],5,2), sec[:stop])
    index=int(np.argmin(np.abs(sec-turn["reversal_time_from_attempt_start_seconds"])))
    direction=turn["outgoing_y_direction"]
    incoming=turn["incoming_lateral_speed_yards_per_second"]
    selected=slice(max(0,index-6),min(stop,index+9))
    relative=sec[selected]-sec[index]; oriented=direction*velocity[selected]
    window_time=sec[index:index+6]-sec[index]
    shortfall=np.maximum(0,incoming-direction*velocity[index:index+6])
    area=float(np.trapezoid(shortfall,window_time));cost=area/incoming
    assert np.isclose(cost,turn["normalized_half_second_recovery_cost_seconds"],atol=1e-12)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.55),layout="constrained")
    ax=axes[0]
    ax.plot(x[:stop]-origin_x,y[:stop]-origin_y,color=BLUE,lw=1.7,label="Mirror phase")
    if stop<len(x):
        end=min(len(x),stop+12)
        ax.plot(x[stop-1:end]-origin_x,y[stop-1:end]-origin_y,color=ORANGE,lw=1.3,ls="--",
                label="Excluded forward run")
        ax.scatter(x[stop]-origin_x,y[stop]-origin_y,s=30,marker="x",color=ORANGE,zorder=5)
        boundary=np.sign(x[stop]-origin_x)
        ax.axvline(boundary,color=GREY,ls=":",lw=.8)
        ax.text(boundary+.08,.64,f"Forward-exit boundary\n({boundary:+.0f} yd)",
                transform=ax.get_xaxis_transform(),va="top",ha="left",
                fontsize=8,color=GREY)
    ax.scatter(x[0]-origin_x,y[0]-origin_y,color=GREY,s=30,zorder=6)
    ax.annotate("Start",(x[0]-origin_x,y[0]-origin_y),xytext=(-25,-13),textcoords="offset points",
                arrowprops={"arrowstyle":"->","linewidth":.7},fontsize=8.5)
    ax.scatter(x[index]-origin_x,y[index]-origin_y,color=ORANGE,marker="s",s=30,zorder=7)
    ax.annotate(f"Reversal at {sec[index]:.1f} s",(x[index]-origin_x,y[index]-origin_y),xytext=(18,12),
                textcoords="offset points",arrowprops={"arrowstyle":"->","color":ORANGE,"linewidth":.7},
                fontsize=8.5,color=ORANGE)
    ax.set_aspect("equal",adjustable="datalim")
    ax.set(xlabel="Forward displacement from initial median (yd)",
           ylabel="Lateral displacement from start (yd)")
    ax.legend(frameon=False,loc="lower right",fontsize=8)
    panel(ax,"a","Selected mirror-drill trajectory");axis_style(ax,"both")
    ax=axes[1]
    ax.plot(relative,oriented,color=BLUE,lw=1.7,label="Outgoing-signed velocity")
    ax.axhline(incoming,color=PURPLE,ls="--",lw=1.2,label=f"Entry speed: {incoming:.2f} yd/s")
    ax.axhline(0,color=GREY,lw=.7);ax.axvline(0,color=ORANGE,lw=.8)
    ax.fill_between(window_time,incoming-shortfall,incoming,color=GOLD,alpha=.65,
                    label="Positive velocity shortfall, 0–0.5 s")
    ax.axvspan(0,.5,color=GOLD,alpha=.07)
    ax.text(.035,.96,f"Area: {area:.3f} yd\nArea / entry speed: {cost:.3f} s",transform=ax.transAxes,
            va="top",fontsize=8.5,bbox={"facecolor":"white","edgecolor":"#E1E5E7","alpha":.98})
    ax.set(xlabel="Time from sampled kinematic reversal (s)",
           ylabel="Outgoing-signed lateral velocity (yd/s)")
    ax.legend(frameon=False,loc="lower right",fontsize=8)
    panel(ax,"b","Half-second movement recovery cost");axis_style(ax,"both")
    save_figure(fig,output)
    return {"nfl_id":card["nfl_id"],"turn_seconds":float(sec[index]),
            "entry_speed_yards_s":incoming,"area_yards":area,"cost_seconds":cost,
            "matches_original_metric":True,"reliability_established":False}


def draw_film_from_file(tracking_path, functional_code_directory, output):
    """Small official-input extraction for the same deterministic film example.

    Only the already selected example is read into memory. No row table is
    written, no NFL outcomes are opened, and the curve detector is unchanged.
    """
    import sys
    import pandas as pd
    sys.path.insert(0,str(functional_code_directory))
    from run_functional import extract_curves
    fragments=[]
    for frame in pd.read_csv(tracking_path,chunksize=200000,low_memory=False):
        mask=(frame.nfl_id.eq(55870)&frame.entity_type.eq("PLAYER")&
              frame.drill_name.str.replace("-","_",regex=False).eq("PASS_PRO_MIRROR_DRILL"))
        if mask.any():
            fragments.append(frame.loc[mask])
    tracking=pd.concat(fragments,ignore_index=True)
    attempts,events,audit=extract_curves(tracking)
    turns=events[events.attempt_id.isin(set(attempts.attempt_id))].sort_values(["event_id","attempt","reversal_time_seconds"])
    turn=turns.iloc[0]
    card={"nfl_id":55870,"turns":[{
        "event_id":str(turn.event_id),"attempt":int(turn.attempt),
        "reversal_time_from_attempt_start_seconds":float(turn.reversal_time_seconds),
        "incoming_lateral_speed_yards_per_second":float(turn.incoming_speed),
        "normalized_half_second_recovery_cost_seconds":float(turn.normalized_deficit),
        "outgoing_y_direction":int(turn.direction),
    }]}
    return draw_film(tracking,card,output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-base",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--tracking",type=Path)
    parser.add_argument("--predictions",type=Path)
    parser.add_argument("--interval-rows",type=Path)
    args=parser.parse_args();base=args.source_base;out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    source_paths={
        "transport":base/"final_revision/verified_run/transport_appendix_outputs/TRANSPORT_APPENDIX_AGGREGATE.json",
        "frozen":base/"final_revision/verified_run/FINAL_2025_RESULT.json",
        "counts":base/"final_revision/verified_run/hierarchical_results/results.json",
        "intervals":base/"final_revision/verified_run/hierarchical_results/interval_coverage.json",
        "splits":base/"transport_validation/round8/split_sensitivity.json",
        "delete":base/"transport_validation/round8/deletion_influence.json",
        "boot":base/"transport_validation/round8/retuned_bootstrap.json",
    }
    d={k:json.loads(p.read_text()) for k,p in source_paths.items()}
    draw_selection(d["transport"],d,d["frozen"],out/"figure2_selection_stability.png")
    draw_frozen(d["frozen"],out/"figure3_frozen_result.png")
    primary=None; interval_rows=None; checks={}
    if args.predictions:
        import pandas as pd
        primary=pd.read_csv(args.predictions)
        n=primary[primary.model.eq("constant_equal_player")]
        assert n.player_id.nunique()==d["counts"]["n_players"]==38
        assert int(n.n.sum())==d["counts"]["observations"]==16165
        assert int(n.y.sum())==d["counts"]["pressures"]==1444
        maximum=0.0
        for model, rows in primary.groupby("model"):
            if model not in d["counts"]["metrics"]:
                continue
            rate=rows.y/rows.n
            brier=float((rate*(1-rows.prediction)**2+(1-rate)*rows.prediction**2).mean())
            diff=abs(brier-d["counts"]["metrics"][model]["equal_player_brier"])
            maximum=max(maximum,diff)
            assert diff<1e-8,(model,diff)
        checks["private_prediction_max_brier_difference_from_verified_aggregate"]=maximum
        source_paths["private_predictions"]=args.predictions
    if args.interval_rows:
        import pandas as pd
        interval_rows=pd.read_csv(args.interval_rows)
        maximum=0.0
        for model,rows in interval_rows.groupby("model"):
            if model not in d["intervals"]:
                continue
            rate=rows.y/rows.n
            values={
                "central90_count_coverage":float(((rate>=rows.conditional90_low)&(rate<=rows.conditional90_high)).mean()),
                "jackknife_style_alpha05_empirical_coverage":float(((rate>=rows.jackknife_plus_low)&(rate<=rows.jackknife_plus_high)).mean()),
                "median_central90_width":float((rows.conditional90_high-rows.conditional90_low).median()),
                "median_jackknife_style_alpha05_width":float((rows.jackknife_plus_high-rows.jackknife_plus_low).median()),
            }
            for key,value in values.items():
                diff=abs(value-d["intervals"][model][key]);maximum=max(maximum,diff)
                assert diff<1e-8,(model,key,diff)
        checks["private_interval_max_difference_from_verified_aggregate"]=maximum
        source_paths["private_intervals"]=args.interval_rows
    draw_count(d["counts"],d["intervals"],out/"figure4_count_uncertainty.png",primary,interval_rows)
    if args.tracking:
        check=draw_film_from_file(args.tracking,base/"final_revision/verified_run/functional_code",out/"figure1_film_card.png")
        original=json.loads((base/"final_revision/verified_run/functional_results/FILM_CARD_NUMERICAL_CHECK.json").read_text())
        for left,right in [("turn_seconds","selected_turn_seconds"),
                           ("entry_speed_yards_s","entry_speed_yards_s"),
                           ("area_yards","positive_shortfall_area_yards"),
                           ("cost_seconds","area_divided_by_entry_speed_seconds")]:
            assert abs(check[left]-original[right])<1e-12,(left,check[left],original[right])
        checks["film_card"]=check
        checks["film_card_matches_verified_run_check_atol"]=1e-12
        source_paths["private_tracking"]=args.tracking
    provenance={"display_only":True,"new_fit":False,"new_evaluation":False,
        "source_sha256":{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in source_paths.items()},
        "figure_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("figure*.png")},
        "figure4_local_panel_a":"Exact four original fixed-OOF bootstrap intervals, unchanged draws and seed." if primary is not None else "Verified aggregate point loss; original interval computation available via primary= runtime row table.",
        "figure1":"Regenerated from official private tracking and matched archived numerical check." if args.tracking else "Faithful callable renderer supplied; awaiting private tracking/card source or exact runtime extraction.",
        "checks":checks}
    (out/"VISUAL_PROVENANCE.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))


if __name__=="__main__":
    main()
