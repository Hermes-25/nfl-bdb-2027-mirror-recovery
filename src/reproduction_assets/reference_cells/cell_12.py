def input_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

input_ids={f:{'name':f,'bytes':(DATA_DIR/f).stat().st_size,'sha256':input_hash(DATA_DIR/f)} for f in REQUIRED}
selected_alpha_base=bundle['alphas']['conventional'];selected_alpha_full=bundle['alphas']['full_family']
config={'development_years':[2023,2024],'final_enabled':bool(FINAL_EVALUATION_ENABLED),'seed':SEED,'ridge_grid':GRID,
        'primary_endpoint':'charted_all','minimum_charted_opportunities':50,'base_features':BASE,'tracking_family':FAMILY,
        'baseline_alpha_selected_2023':selected_alpha_base,'enhanced_alpha_selected_2023':selected_alpha_full}
manifest={'candidate':'OL_speed_relative_mirror_recovery','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'code_fingerprint':CODE_FINGERPRINT,'dependencies':VERSIONS,'official_input_identifiers':input_ids,'config':config,
          'timing':'Only Combine measurements/tracking in primary predictors; NFL lineup role defines target exposure, not a prediction input.',
          'detector':{'drill_aliases':['PASS_PRO_MIRROR_DRILL','PASS_PRO-MIRROR_DRILL'],'coordinate_units':'yards',
                      'sampling':'10Hz; adjacentdtwithin0.015s of0.1s','lateral_axis':'y','forward_exit':'first abs(x-median(first10x))>1yard after0.5s',
                      'smoothing':'Savitzky-Golay5frames,polynomial2','stationary_bridge':'abs lateralvelocity<=.05yd/s;bridge<=1s',
                      'usable_turn':'preceding/following median speed>.5yd/s, opposite signs,0.5s spacing;>=2reversals',
                      'loss_window_seconds':.5,'recovery_summary':'P75 of reversal deficit/incoming speed;medianacross attempts',
                      'speed_summary':'median incoming speed','excursion':'P90(y)-P10(y) withinmirrorphase;medianacross attempts',
                      'event_keys':['draft_year','event_id','nfl_id','attempt'],'repeatability':'Notestablished'},
          'outcome_policy':'Combine_position OL participants; explicitnon-nullpressure_allowed,offensiveC/G/Tlineup,possessionteam,notnullified,rookie season==draft_year,REGonly;min50. Missingneverzero.',
          'metric_policy':'Primary equal-player mean binary Brier; logloss and calibration descriptive; matchedeligibleplayers; no correlation-based headline.',
          'final_recipe':{'enabled':False,'authorization_required':'OL_selected_by_root','fit_years':[2023,2024],'evaluate_year':2025,
                          'baseline_features':BASE,'enhanced_features':BASE+FAMILY,'baseline_alpha':selected_alpha_base,
                          'enhanced_alpha':selected_alpha_full,'preprocessing':'medianimpute/standardize fitted on pooled2023/24eligibleplayers only',
                          'player_weighting':'equal','prediction_clip':[.001,.999],'cohort_policy':'identical frozen endpoint/min50/drillfeature gates',
                          'no_holdout_model_switch':True},
          'uncertainty_scope':'250training+selectionplayerbootstraprefitsfixselectedalpha;5000fixedmodelplayerdraws;LOOalpha reselected. Excludesalpha/familyselection uncertainty and sharedteam/game dependence.'}
packet={'status':'completed','primary':primary,'denominator_sensitivities':sensitivities,'role_conditioned_R4_history':historical,
        'audit':audit,'config':config,'input_identifiers':input_ids,'freeze_manifest':manifest,'final_2025_opened':False,
        'history':'R2geometrybug repair;R3normalization/denominator audits;R4compactfamily/CV;R5drops unverifiedpre-draftNFLrole. Repeated2024selection; not original preregistration.'}
(WORK/'config.json').write_text(json.dumps(config,indent=2))
(WORK/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2))
(WORK/'ROUND5_RESULT.json').write_text(json.dumps(packet,indent=2,allow_nan=False))

u=primary['comparisons']['full_family']['uncertainty']['brier'];ci=u['training_selection_refit_bootstrap95']
bs=primary['models']['conventional']['selected_model_score'];fs=primary['models']['full_family']['selected_model_score']
article=f'''# Relative mirror recovery: a measured question about rookie pass protection

**Question.** Does a prospect's speed-relative recovery during the Combine mirror drill add information about first-year charted NFL pass protection beyond conventional Combine measurements?

**Current evidence.** The timing-safe profile has a Brier-loss improvement of {u['baseline_minus_enhanced']:+.6f} on 20 rookies in the 2024 exploratory selection cohort. Its training-and-selection player-refit interval is [{ci[0]:+.6f}, {ci[1]:+.6f}]. That uncertainty does not establish incremental predictive value. It also does not establish that tracking is useless. We propose a specific measurement for film review, not a talent ranking or automatic grade adjustment.

## A football measurement, not a reaction-time score

An offensive lineman repeatedly changes lateral direction in the mirror drill, then finishes with a forward run. We separate that run-out before measuring reversal cost. The recorded geometry supports lateral y and forward x; we have no body orientation, opponent trajectory, or stimulus timestamp. Consequently, the measure describes the prospect's own movement, not reaction latency, stance quality, or a causal athletic trait.

For each usable reversal we integrate positive movement shortfall over 0.5 seconds: the preceding lateral speed minus velocity toward the new direction, bounded below by zero. Dividing by incoming speed expresses the shortfall in **effective seconds of speed-relative movement loss**. We use the 75th percentile across reversals and the median across attempts. Two companion descriptors characterize the task: median incoming lateral speed in yards/second and the mirror-phase 90th-minus-10th percentile lateral excursion in yards. Excursion can reflect drill instructions; it is not an ability grade.

Positions are smoothed with a five-frame quadratic Savitzky–Golay filter; velocities come from coordinates and timestamps, not supplied speed or distance columns. Near-zero velocities are bridged to avoid missing stationary turns. Usable turns have opposite directions and median incoming/outgoing speeds above 0.5 yards/second; at least two are required. The forward run starts when x moves more than one yard from its initial reference. Full event and attempt identifiers prevent trajectories from being combined accidentally.

![Figure 1](figure1_geometry.png)

**Figure 1.** Geometry and discretized recovery loss for the smallest player identifier in each cohort and the longest 2023 recording. These cases were selected without NFL outcomes. The third case illustrates variable drill duration; recorded attempts are not repeated physiological trials. The full development cohorts contain 37 and 46 mirror participants.

## The NFL question is conditional on protection opportunity

We start with Combine position **OL**, link NFL identifiers, and retain only the prospect's draft-year regular season. An eligible opportunity has an explicit non-null pressure-allowed label, an offensive C/G/T alignment on the possession team, and a play that was not nullified. We require at least 50 charted observations per player. Missing labels are undefined; they are never pressure-free plays. Explicit False labels include some zero-rusher-encounter observations, so this is a charted-protection estimand rather than a universal pass-rush denominator.

The 2023 cohort has 18 eligible players, 8,182 opportunities and 733 charted pressures; 2024 has 20 players, 7,983 opportunities and 711 pressures. Playing opportunity and charting create selection: the results do not describe all 83 prospects' unconditional NFL ability or career success.

![Figure 2](figure2_coverage.png)

**Figure 2.** The participant funnel shows why thousands of plays do not provide thousands of independent prospects. Complete Combine measurement does not guarantee a meaningful rookie role. The effective model/evaluation samples are 18 and 20 players.

## Timing-safe comparison and uncertainty

The conventional baseline uses Combine height, weight, forty time and ten-yard split. The full model adds only the three frozen tracking descriptors. NFL roster position was removed after an information-timing audit: it is not verified as a pre-draft input. The earlier role-conditioned result is reproduced in the executable appendix and explicitly retained as exploratory history.

Each model independently chooses ridge strength from 1, 10 or 100 using 2023 leave-one-player-out cross-validation. Imputation and scaling are fitted inside every fold. The selected baseline alpha is {selected_alpha_base:g}; the full-profile alpha is {selected_alpha_full:g}. A 2023 model-selection score is not independent final accuracy. Every grid score is retained, and 2024 remains repeatedly used exploratory selection data. We do not select the endpoint or model using the most favorable 2024 result.

| 2024 timing-safe model | Equal-player Brier loss |
|---|---:|
| Training-mean constant | {primary['constant']['player_equal_brier']:.6f} |
| Conventional Combine measurements | {bs['player_equal_brier']:.6f} |
| Conventional measurements plus mirror profile | {fs['player_equal_brier']:.6f} |

The interval above resamples players independently within the training and selection cohorts and refits preprocessing and models 250 times while fixing the selected alphas. A separate 5,000-draw player bootstrap fixes the trained models. Neither accounts fully for earlier family/alpha choices or shared teams and games. Leaving out each training player and reselecting alpha changes some full-profile predictions by as much as {100*primary['comparisons']['full_family']['max_LOO_prediction_shift']:.2f} percentage points, emphasizing small-sample fragility.

![Figure 3](figure3_decision.png)

**Figure 3.** Proper-loss improvement and player-level uncertainty on identical 2024 participants. Normalized recovery alone and the speed/excursion pair are explanatory subsets, not replacement headline models. The notebook also retains zero-encounter exclusion, C/I/IN/S-code restriction, and their intersection as distinct conditional endpoint sensitivities. None is promoted for a favorable score.

## What a scouting department can use now

Attach the extracted turn timestamps and trajectory to available film. Review whether a large relative recovery loss recurs at comparable incoming speed and excursion, and use the clip to formulate a technical question. This workflow preserves observable movement rather than compressing it into an unsupported talent grade. These pilot data do not justify automatically adjusting a rookie pressure forecast or recommending a roster decision.

The executable audit checks aliases, linkage uniqueness, coordinate bounds, chart-label missingness, duplicate timestamps, truncated trajectories and empty cohorts. Development-only smoothing and exit-threshold sensitivities are reported, without choosing thresholds by NFL outcome scores. One recorded attempt per player means computational robustness cannot establish physiological reliability.

A final 2025 branch is disabled. Only the team's selected candidate may receive a frozen pooled-2023/2024 fit and a single 2025 confirmation under unchanged features, alphas and eligibility rules. If that confirmation fails, the failure must be reported without selecting a different candidate. This article is an internal candidate artifact; no public notebook or competition submission has been published.
'''
assert len(article.split())<=2000
(WORK/'article.txt').write_text(article)
print('Article words:',len(article.split()),'; figures:3; article tables:1')
print('Output packet, article, config and candidate freeze manifest written.')
