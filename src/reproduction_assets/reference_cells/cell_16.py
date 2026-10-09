
# This uncertainty calculation was specified before the first reserved look.
# The point models were already fitted and serialized before loading 2025.
assert FINAL_EVALUATION_ENABLED
assert baseline_state['alpha'] == 10 and enhanced_state['alpha'] == 1
delta = loss(yy, bp) - loss(yy, ep)
rng_final = np.random.default_rng(62025)
fixed_draws = np.array([delta[rng_final.integers(0, len(heldout), len(heldout))].mean()
                        for _ in range(5000)])
refit_draws = []
for _ in range(250):
    ti = rng_final.integers(0, len(pooled), len(pooled))
    ei = rng_final.integers(0, len(heldout), len(heldout))
    train_b = pooled.iloc[ti].reset_index(drop=True)
    test_b = heldout.iloc[ei].reset_index(drop=True)
    state_b = fit_state(train_b, BASE, FROZEN_BASELINE_ALPHA)
    state_e = fit_state(train_b, BASE + FAMILY, FROZEN_ENHANCED_ALPHA)
    pred_b = apply_state(state_b, test_b)
    pred_e = apply_state(state_e, test_b)
    refit_draws.append(float((loss(test_b.rate.to_numpy(), pred_b) -
                              loss(test_b.rate.to_numpy(), pred_e)).mean()))
final_result.update({
    'evaluation_type': 'Frozen reserved-cohort evaluation; retrospective temporal holdout',
    'training_years': [2023, 2024], 'evaluation_year': 2025,
    'training_players': len(pooled), 'evaluation_players': len(heldout),
    'charted_opportunities': int(heldout.n.sum()),
    'charted_pressures': int(np.rint((heldout.rate * heldout.n).sum())),
    'baseline_alpha': 10, 'enhanced_alpha': 1,
    'brier_improvement': float(delta.mean()),
    'fixed_state_player_bootstrap95': np.quantile(fixed_draws, [.025, .975]).tolist(),
    'pooled_training_and_evaluation_refit_bootstrap95': np.quantile(refit_draws, [.025, .975]).tolist(),
    'uncertainty_scope': 'Equal-player resampling; fixed features/alphas; excludes prior selection, detector uncertainty and shared teams/games.',
    'seed': 62025, 'fixed_state_draws': 5000, 'refit_draws': 250,
    'model_or_winner_changed_after_outcomes': False,
    'runtime_versions': VERSIONS,
})
(WORK / 'FINAL_2025_RESULT.json').write_text(json.dumps(final_result, indent=2))
pd.DataFrame({'nfl_id': heldout.nfl_id, 'charted_observations': heldout.n,
              'observed_pressure_rate': yy, 'conventional_prediction': bp,
              'mirror_profile_prediction': ep}).to_csv(WORK / 'FINAL_2025_predictions.csv', index=False)
print('FINAL RESERVED RESULT:', json.dumps(final_result, indent=2))

dev_u = primary['comparisons']['full_family']['uncertainty']['brier']
values = [dev_u['baseline_minus_enhanced'], final_result['brier_improvement']]
intervals = [dev_u['training_selection_refit_bootstrap95'],
             final_result['pooled_training_and_evaluation_refit_bootstrap95']]
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (value, interval, color) in enumerate(zip(values, intervals, ['tab:blue', 'tab:orange'])):
    ax.plot(interval, [i, i], color=color, lw=3)
    ax.scatter([value], [i], color=color, s=65, zorder=3)
ax.axvline(0, color='black', ls='--', lw=1)
ax.set_yticks([0, 1], ['2024 exploratory selection\n18 training / 20 evaluation players',
                      f'2025 frozen reserved evaluation\n{len(pooled)} training / {len(heldout)} evaluation players'])
ax.invert_yaxis()
ax.set_xlabel('Equal-player Brier improvement over conventional Combine measurements\nPositive favors the full mirror profile; intervals resample training and evaluation players')
ax.set_title('Does the selected mirror profile add NFL pressure information?\nSame frozen features and alphas; no selection after 2025 outcomes')
ax.grid(axis='x', alpha=.18)
fig.tight_layout()
fig.savefig(WORK / 'figure4_reserved_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()

ci_final = final_result['pooled_training_and_evaluation_refit_bootstrap95']
final_text = f"""## Frozen reserved-cohort evaluation

We selected this offensive-line entry after five internal research rounds, then recorded the notebook's analysis-code hash and fixed settings before opening NFL season 2025 outcomes. Both preprocessing/model states were fitted on the 38 eligible 2023/2024 players and serialized before loading 2025. The conventional alpha remained 10 and full-profile alpha 1. No feature, endpoint, eligibility, model or winner changed afterward.

The fixed recipe evaluated **{len(heldout)} eligible 2025 rookies**, with **{int(heldout.n.sum()):,} charted observations** and **{final_result['charted_pressures']:,} pressures**. Equal-player Brier loss was **{final_result['baseline']['player_equal_brier']:.6f}** for conventional measurements and **{final_result['enhanced']['player_equal_brier']:.6f}** with the mirror profile. Improvement was **{final_result['brier_improvement']:+.6f}**, with 250 paired pooled-training/evaluation-player refits giving interval **[{ci_final[0]:+.6f}, {ci_final[1]:+.6f}]**. The notebook also reports the constant baseline and a 5,000-draw fixed-state interval.

![Figure 4](figure4_reserved_evaluation.png)

**Figure 4.** The original selection result and one frozen reserved-cohort evaluation. Intervals condition on the selected features/alphas and omit earlier selection, detector uncertainty and shared-game dependence. These are retrospective outcomes supplied with the competition, not prospectively collected evidence. A favorable point estimate alone does not establish reliable scouting value; an unfavorable result is retained without a replacement model.

Data: National Football League, [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027), Kaggle (2026), CC BY-NC 4.0 and competition rules. Original analysis code: MIT. AI assistance was used for research, coding and review; all results are reproduced from official inputs with open-source CPU libraries.
"""
old_ending = 'A final 2025 branch is disabled.'
article = article.split(old_ending)[0].rstrip() + '\n\n' + final_text
assert len(article.split()) <= 2000
(WORK / 'article.txt').write_text(article)
print('Final article words:', len(article.split()), '; figures: 4; tables: 1')
