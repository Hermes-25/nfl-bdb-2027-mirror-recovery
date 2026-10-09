# This branch is deliberately disabled in every Round5 candidate.
# Root alone may authorize the ONE selected candidate, record code/config hashes,
# fill the frozen alphas from freeze_manifest.json, and then execute this branch.
if FINAL_EVALUATION_ENABLED:
    if FINAL_SELECTION_AUTHORIZATION != 'OL_selected_by_root':
        raise PermissionError('Root selection authorization required before any final outcome loading.')
    if FROZEN_BASELINE_ALPHA not in GRID or FROZEN_ENHANCED_ALPHA not in GRID:
        raise ValueError('Both frozen alphas must be explicitly supplied before final outcomes are opened.')
    if FROZEN_BASELINE_ALPHA != selected_alpha_base or FROZEN_ENHANCED_ALPHA != selected_alpha_full:
        raise ValueError('Final alphas differ from frozen development recipe; evaluation stopped before loading2025.')
    # Fit and serialize BOTH states before opening any final outcomes.
    pooled=prepared['eligible'].reset_index(drop=True)
    baseline_state=fit_state(pooled,BASE,FROZEN_BASELINE_ALPHA)
    enhanced_state=fit_state(pooled,BASE+FAMILY,FROZEN_ENHANCED_ALPHA)
    preholdout_state={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_fingerprint':CODE_FINGERPRINT,'baseline':baseline_state,'enhanced':enhanced_state,
        'training_mean_pressure':float(pooled.rate.mean()),'config':config}
    serialized_state=json.dumps(preholdout_state,indent=2)
    (WORK/'PREHOLDOUT_FITTED_STATES.json').write_text(serialized_state)
    preholdout_hash=hashlib.sha256(serialized_state.encode()).hexdigest()
    # Only now may the single selected candidate load its final cohort.
    final_inputs=load_inputs(DATA_DIR,years=(2025,),final_enabled=True)
    final_prepared=prepare_cohort(final_inputs,years=(2025,))
    heldout=final_prepared['eligible'].reset_index(drop=True)
    if len(heldout)<10:raise ValueError('Final cohort fails frozen10-player feasibility gate; no replacement model allowed.')
    yy=heldout.rate.to_numpy();n=heldout.n.to_numpy()
    bp=apply_state(baseline_state,heldout);ep=apply_state(enhanced_state,heldout)
    final_result={'constant':scores(yy,np.full(len(heldout),pooled.rate.mean()),n),'baseline':scores(yy,bp,n),'enhanced':scores(yy,ep,n),
                  'preholdout_fitted_state_sha256':preholdout_hash}
    final_result['funnel']=final_prepared['funnel'];final_result['status']='one_frozen_confirmation_no_model_switch'
    (WORK/'FINAL_2025_RESULT.json').write_text(json.dumps(final_result,indent=2))
    print('Frozen2025confirmation:',final_result)
else:
    print('Final2025branch NOT EXECUTED. No2025 outcomes evaluated.')
