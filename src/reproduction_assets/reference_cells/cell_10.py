# Figure1: mechanically selected cases: smallest identifier in each year and
# longest2023 recording. No NFL outcomes determine the example selection.
attempts=prepared['attempts'];examples=prepared['examples']
cases=[attempts[attempts.draft_year.eq(y)&attempts.quality.eq('ok')].sort_values('nfl_id').iloc[0] for y in [2023,2024]]
cases.append(attempts[attempts.draft_year.eq(2023)].sort_values('frames',ascending=False).iloc[0])
fig,axes=plt.subplots(3,2,figsize=(11,10));example_records=[]
for i,a in enumerate(cases):
    key=(int(a.draft_year),a.event_id,int(a.nfl_id),int(a.attempt));e=examples[key];stop=e['stop'];sec=e['sec'];x=e['x'];yy=e['y']
    v=np.gradient(savgol_filter(yy[:stop],5,2),sec[:stop]);r=int(e['rev'][0]);pre=float(np.median(v[r-5:r-1]));post=float(np.median(v[r+1:r+5]));new=np.sign(post)*v[r:r+6];tt=sec[r:r+6]-sec[r]
    name=inputs['players'].loc[inputs['players'].nfl_id.eq(a.nfl_id),'display_name'].iloc[0]
    axes[i,0].plot(x,yy,color='.75',label='full recording');axes[i,0].plot(x[:stop],yy[:stop],label='mirror phase');axes[i,0].scatter(x[e['rev']],yy[e['rev']],color='tab:red',s=20,label='usable reversals')
    axes[i,0].set(xlabel='x (yards)',ylabel='y (yards)',title=f'{int(a.draft_year)}: {name}; {len(e["rev"])} reversals')
    axes[i,0].legend(fontsize=8)
    axes[i,1].plot(tt,new,label='velocity toward new direction');axes[i,1].axhline(abs(pre),color='black',ls='--',label='incoming speed')
    axes[i,1].fill_between(tt,np.minimum(new,abs(pre)),abs(pre),color='tab:orange',alpha=.4,label='discretized deficit')
    axes[i,1].set(xlabel='time after reversal (seconds)',ylabel='lateral velocity (yards/s)',title='First usable reversal; no reaction stimulus observed');axes[i,1].legend(fontsize=8)
    example_records.append({'year':int(a.draft_year),'nfl_id':int(a.nfl_id),'mirror_seconds':float(sec[stop-1]),'usable_reversals':len(e['rev'])})
fig.suptitle('Mirror movement and final forward run-out: 37/46 development participants\nThird row: longest 2023 recording; a protocol-duration edge case',fontsize=12)
fig.tight_layout();fig.savefig(WORK/'figure1_geometry.png',dpi=150,bbox_inches='tight');plt.show()
audit['deterministic_geometry_examples']=example_records

# Figure2: participant attrition rather than play counts as independentN.
stages=['Combine OL','Usable mirror','Any rookie REG','Charted endpoint','At least 50 observations']
keys=['Combine_OL_participants','usable_feature_players','any_rookie_REG_players','observed_endpoint_players','eligible_players_min50']
fig,ax=plt.subplots(figsize=(10,4.5));pos=np.arange(len(keys))
for offset,year,color in [(-.18,2023,'tab:blue'),(.18,2024,'tab:orange')]:
    vals=[prepared['funnel'][str(year)][k] for k in keys];bars=ax.bar(pos+offset,vals,width=.35,color=color,label=str(year))
    ax.bar_label(bars,padding=3)
ax.set_xticks(pos,stages);ax.set(ylabel='Players (evaluation units)',title='Complete mirror measurement does not imply NFL playing opportunity\n2023 training n=18; 2024 exploratory selection n=20')
ax.legend();ax.set_ylim(0,53);fig.tight_layout();fig.savefig(WORK/'figure2_coverage.png',dpi=150,bbox_inches='tight');plt.show()

# Figure3: timing-safe primary and physically motivated explanatory subsets.
fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.1,1]})
names=['normalized','speed_amplitude','full_family'];ax=axes[0]
for i,name in enumerate(names):
    u=primary['comparisons'][name]['uncertainty']['brier'];m=u['baseline_minus_enhanced'];f=u['fixed_model_player_bootstrap95'];b=u['training_selection_refit_bootstrap95']
    ax.plot(b,[i+.08,i+.08],color='tab:orange',lw=2);ax.plot(f,[i-.08,i-.08],color='tab:blue',lw=2);ax.scatter([m],[i],color='black',zorder=4)
ax.set_yticks(range(3),['Normalized recovery','Speed + excursion','Full mirror profile']);ax.invert_yaxis();ax.axvline(0,color='black',ls='--')
ax.plot([],[],color='tab:blue',label='95% fixed-training bootstrap');ax.plot([],[],color='tab:orange',label='95% training + selection refits');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.23))
ax.set(xlabel='Brier improvement over conventional Combine tests\n(positive favors tracking)',title='2024 exploratory selection: 20 players\n2023 player CV only; no NFL-role predictor')
u=primary['comparisons']['full_family']['uncertainty']['brier'];ci=u['training_selection_refit_bootstrap95'];shift=primary['comparisons']['full_family']['max_LOO_prediction_shift']
axes[1].axis('off');axes[1].text(0,1,'Decision evidence',fontsize=16,weight='bold',va='top')
text=(f'Football question\nDoes relative mirror recovery add protection\ninformation beyond conventional tests?\n\n'
      f'Primary estimate\nΔBrier: {u["baseline_minus_enhanced"]:+.6f}\n95% refit interval: [{ci[0]:+.6f}, {ci[1]:+.6f}]\n\n'
      f'Supported use\nAttach turn timestamps to film review; compare\nreversal cost at matched speed and excursion.\n\n'
      f'Limits\n18 training / 20 selection players; one attempt.\nMaximum prediction shift: {100*shift:.2f} pp.\nNo automatic grade or talent ranking.\nNo causal ability or reaction-time claim.')
axes[1].text(0,.88,text,fontsize=10.5,va='top',linespacing=1.18)
fig.tight_layout();fig.savefig(WORK/'figure3_decision.png',dpi=150,bbox_inches='tight');plt.show()
