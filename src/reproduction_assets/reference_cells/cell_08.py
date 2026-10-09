DATA_DIR=locate_data()
inputs=load_inputs(DATA_DIR,years=(2023,2024),final_enabled=False)
prepared=prepare_cohort(inputs)
primary,bundle=development_models(prepared['eligible'])
print('Primary 2023 training / 2024 selection players:',primary['train_players'],primary['selection_players'])
print('Timing-safe predictors:',BASE+FAMILY)
print('2023 CV chosen alphas:',bundle['alphas'])
print('Constant, conventional, full-family Brier:',primary['constant']['player_equal_brier'],
      primary['models']['conventional']['selected_model_score']['player_equal_brier'],
      primary['models']['full_family']['selected_model_score']['player_equal_brier'])
print('Primary Brier improvement and uncertainty:',primary['comparisons']['full_family']['uncertainty']['brier'])
bundle['prediction_frame'].to_csv(WORK/'development_predictions.csv',index=False)

historical,_=development_models(prepared['eligible'],historical_role=True)
assert abs(historical['models']['conventional']['selected_model_score']['player_equal_brier']-0.08483422126191108)<1e-10
assert abs(historical['models']['full_family']['selected_model_score']['player_equal_brier']-0.08468259293121042)<1e-10
sensitivities={}
for endpoint in ['encountered_at_least_one','pass_results_C_I_IN_S','encountered_and_C_I_IN_S']:
    alt=prepare_cohort(inputs,endpoint=endpoint)
    rr,_=development_models(alt['eligible'])
    sensitivities[endpoint]={'funnel':alt['funnel'],'models':rr}
print('Role-conditioned R4 values reproduced and labeled historical only.')
print('Three denominator sensitivities retained; none promoted by 2024 score.')

# Meaningful adversarial checks. These do not train/test hypothetical outcomes.
audit={'input':inputs['audit'],'funnel':prepared['funnel'],'coordinate_velocity_source':'x/y/time; supplied s/a/dis ignored',
       'attempt_repeatability':'one recorded mirror attempt per player in development; no physiological reliability claim',
       'timing_correction':'NFL roster is_tackle removed from primary. Initial OL cohort now uses documented Combine_position OL.',
       'event_key_repair':'Full event_id+attempt keys retained for multiple-event safety; development metrics unchanged.'}
t=inputs['tracking'];first=t.sort_values(['draft_year','nfl_id','event_id','attempt','time']).iloc[0]
same=t[(t.draft_year==first.draft_year)&(t.nfl_id==first.nfl_id)&(t.event_id==first.event_id)&(t.attempt==first.attempt)].sort_values('time')
checks={}
try:extract_mirror(pd.concat([same,same.iloc[:1]],ignore_index=True))
except ValueError as e:checks['duplicate_timestamp_rejected']=str(e)
short,_,_=extract_mirror(same.iloc[:10])
assert short.quality.ne('ok').all() and short.recovery_deficit.isna().all()
checks['truncated_actual_trajectory']='First10 frames of a real2023 trajectory produce undefined metric, not0.'
bad=same.copy();bad.loc[bad.index[0],'x']=999
try:extract_mirror(bad)
except ValueError as e:checks['coordinate_unit_failure_rejected']=str(e)
try:fit_recipe(bundle['train'].iloc[:0],bundle['test'],10,10)
except ValueError as e:checks['empty_cohort_rejected']=str(e)
try:load_inputs(Path('/must_not_be_opened'),years=(2025,),final_enabled=False)
except PermissionError as e:checks['final_guard_rejected_before_fileaccess']=str(e)
assert len(checks)==5
audit['adversarial_checks']=checks
truth=prepared['rookie_rows'].copy()
truth['pressure_label']=truth.pressure_allowed.map(lambda x:'missing' if pd.isna(x) else str(x).lower())
truth['rusher_bucket']=np.select([truth.pass_rushers_encountered.isna(),truth.pass_rushers_encountered.eq(0),truth.pass_rushers_encountered.gt(0)],['missing','zero','positive'],default='other')
truth_table=truth.groupby(['draft_year','pressure_label','rusher_bucket']).size().rename('rows').reset_index()
audit['pressure_label_truth_table']=truth_table.to_dict('records')
audit['charted_false']='Explicit label, including0-encounter rows; missing is notFalse. Conditional charted protection estimand.'
audit['accepted_attempts_by_year']={str(y):int((prepared['attempts'].draft_year.eq(y)&prepared['attempts'].quality.eq('ok')).sum()) for y in [2023,2024]}
audit['no_role_primary_assertion']=all('tackle' not in c and c!='nfl_position' for c in BASE+FAMILY)
assert audit['no_role_primary_assertion']
state=fit_state(bundle['train'],BASE+FAMILY,bundle['alphas']['full_family'])
state_prediction=apply_state(state,bundle['test'])
assert np.allclose(state_prediction,bundle['predictions']['full_family'],rtol=0,atol=1e-14)
audit['fitted_state_prediction_equivalence_max_abs_difference']=float(abs(state_prediction-bundle['predictions']['full_family']).max())
print('Adversarial checks:',checks)

# Development-only mechanical sensitivity inherited from earlier audit; no NFL
# score is used to choose an exit threshold, smoothing window or family member.
geo_sensitivity=[]
reference=prepared['features'][prepared['features'].draft_year.eq(2023)]
for window,exit_yards in [(3,1.0),(7,1.0),(5,.75),(5,1.25)]:
    aa,ee,ex=extract(t[t.draft_year.eq(2023)],window,exit_yards)
    good=aa[aa.quality.eq('ok')];keys=['draft_year','event_id','nfl_id','attempt']
    e=ee.merge(good[keys],on=keys,validate='many_to_one');e['norm']=e.recovery_deficit_yards/e.incoming_speed_yards_s
    z=e.groupby(['draft_year','nfl_id'],as_index=False).agg(normalized_deficit=('norm',lambda z:np.percentile(z,75)),incoming_speed=('incoming_speed_yards_s','median'))
    amp=[]
    for _,row in good.iterrows():
        key=(int(row.draft_year),row.event_id,int(row.nfl_id),int(row.attempt));v=ex[key]['y'][:ex[key]['stop']]
        amp.append({'draft_year':int(row.draft_year),'nfl_id':int(row.nfl_id),'lateral_amplitude':float(np.percentile(v,90)-np.percentile(v,10))})
    z=z.merge(pd.DataFrame(amp).groupby(['draft_year','nfl_id'],as_index=False).median(),on=['draft_year','nfl_id'])
    paired=z.merge(reference,on=['draft_year','nfl_id'],suffixes=('_alternative','_primary'))
    geo_sensitivity.append({'smoothing_frames':window,'exit_yards':exit_yards,'common_players':len(paired),
        'rank_correlation':{field:float(scipy.stats.spearmanr(paired[field+'_alternative'],paired[field+'_primary']).statistic) for field in FAMILY}})
audit['geometry_sensitivity_2023_only']=geo_sensitivity
audit['dependency_versions']=VERSIONS
