"""Inspectable turn timestamps, execution intensity and explicit measurement gate.

The selection and cards use Combine tracking only. No NFL pressure label enters
a card, ranking or example choice. Cards are film-review aids, not talent grades.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_functional import extract_curves,CURVE,TIME_GRID,safe_savefig


def draw_film_card(tracking,card,output):
    """XY geometry and exact half-second recovery-area construction."""
    turn=card['turns'][0]  # Earliest retained turn; no NFL outcome used.
    group=tracking[tracking.nfl_id.eq(card['nfl_id']) & tracking.event_id.astype(str).eq(turn['event_id']) &
                   tracking.attempt.eq(turn['attempt'])].sort_values('time')
    sec=(pd.to_datetime(group.time)-pd.to_datetime(group.time.iloc[0])).dt.total_seconds().to_numpy()
    x,y=group.x.to_numpy(float),group.y.to_numpy(float)
    origin_x=np.median(x[:min(10,len(x))]);origin_y=y[0]
    exits=np.flatnonzero((np.abs(x-origin_x)>1)&(sec>.5));stop=int(exits[0]) if len(exits) else len(x)
    smooth=savgol_filter(y[:stop],5,2);velocity=np.gradient(smooth,sec[:stop])
    index=int(np.argmin(np.abs(sec-turn['reversal_time_from_attempt_start_seconds'])))
    direction=turn['outgoing_y_direction'];incoming=turn['incoming_lateral_speed_yards_per_second']
    selected=slice(max(0,index-6),min(stop,index+9));relative=sec[selected]-sec[index]
    oriented=direction*velocity[selected]
    window_time=sec[index:index+6]-sec[index];outgoing=direction*velocity[index:index+6]
    shortfall=np.maximum(0,incoming-outgoing)
    area=float(np.trapezoid(shortfall,window_time));cost=area/incoming
    assert np.isclose(cost,turn['normalized_half_second_recovery_cost_seconds'],atol=1e-12)
    fig,axes=plt.subplots(1,2,figsize=(11.2,5),constrained_layout=True)
    ax=axes[0]
    ax.plot(x[:stop]-origin_x,y[:stop]-origin_y,color='#126d9a',lw=2,label='Mirror phase')
    display_end=min(len(x),stop+12)
    if stop<len(x):
        ax.plot(x[stop-1:display_end]-origin_x,y[stop-1:display_end]-origin_y,color='#c86b2d',lw=1.8,ls='--',label='Forward run: excluded')
        ax.scatter(x[stop]-origin_x,y[stop]-origin_y,s=65,marker='x',color='#c86b2d',zorder=5)
        j=min(stop+5,len(x)-1)
        ax.annotate('Forward run',xy=(x[j]-origin_x,y[j]-origin_y),xytext=(8,15),textcoords='offset points',
                    arrowprops={'arrowstyle':'->','color':'#c86b2d'},fontsize=9,color='#8b4216')
        boundary=np.sign(x[stop]-origin_x)
        ax.axvline(boundary,color='#c86b2d',ls=':',lw=1)
    ax.scatter(x[0]-origin_x,y[0]-origin_y,color='#17846b',s=75,zorder=6)
    ax.annotate('Start',xy=(x[0]-origin_x,y[0]-origin_y),xytext=(-30,-15),textcoords='offset points',
                arrowprops={'arrowstyle':'->'},fontsize=9)
    ax.scatter(x[index]-origin_x,y[index]-origin_y,color='#a54837',marker='s',s=65,zorder=7)
    ax.annotate(f"Selected reversal: {sec[index]:.1f}s",xy=(x[index]-origin_x,y[index]-origin_y),
                xytext=(18,12),textcoords='offset points',arrowprops={'arrowstyle':'->','color':'#a54837'},fontsize=9,color='#a54837')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    ax.set(xlabel='Forward displacement from initial median (yards)',ylabel='Lateral displacement from start (yards)',
           title='Tracking geometry: equal yard scales')
    ax.legend(fontsize=8,loc='lower right')
    ax=axes[1]
    ax.plot(relative,oriented,color='#126d9a',lw=2.5,label='Outgoing-signed lateral velocity')
    ax.axhline(incoming,color='#7a5099',ls='--',lw=1.5,label=f'Entry speed: {incoming:.2f}yd/s')
    ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='#a54837',lw=1)
    ax.fill_between(window_time,incoming-shortfall,incoming,color='#d6a857',alpha=.65,
                    label='Sampled positive shortfall, 0–0.5s')
    ax.axvspan(0,.5,color='#d6a857',alpha=.1)
    ax.text(.03,.94,f'Area {area:.3f}yd ÷ entry speed {incoming:.2f}yd/s\n= recovery cost {cost:.3f}s',
            transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.95})
    ax.set(xlabel='Seconds from sampled kinematic reversal',ylabel='Outgoing-signed lateral velocity (yards/second)',
           title='Half-second deficit: movement cost, not reaction time')
    ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
    fig.suptitle(f"{card['display_name']} · {card['draft_year']} OL mirror · attempt {turn['attempt']}\n"
                 'Example selected by lowest NFL ID, then earliest usable turn; NFL outcomes ignored',fontsize=13)
    safe_savefig(fig,output/'film_review_card.png',dpi=180);plt.close(fig)
    (output/'FILM_CARD_NUMERICAL_CHECK.json').write_text(json.dumps({'nfl_id':card['nfl_id'],'selected_turn_seconds':float(sec[index]),
                                                                  'entry_speed_yards_s':incoming,'positive_shortfall_area_yards':area,
                                                                  'area_divided_by_entry_speed_seconds':cost,'matches_extracted_turn_metric':True,
                                                                  'reliability_established':False},indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--save-player-cards',action='store_true',help='Local audit only; omit for aggregate public outputs.')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    players=pd.read_csv(args.raw_dir/'players.csv',usecols=['nfl_id','display_name','draft_year'])
    results=pd.read_csv(args.raw_dir/'combine_results.csv')
    ids=set(results.loc[results.draft_year.isin([2023,2024])&results.combine_position.eq('OL'),'nfl_id'])
    pieces=[]
    for group in pd.read_csv(args.raw_dir/'combine_tracking.csv',chunksize=200000,low_memory=False):
        q=group[group.draft_year.isin([2023,2024])&group.nfl_id.isin(ids)&group.entity_type.eq('PLAYER')&
                group.drill_name.str.replace('-','_',regex=False).eq('PASS_PRO_MIRROR_DRILL')]
        if len(q):pieces.append(q)
    tracking=pd.concat(pieces,ignore_index=True)
    attempts,events,audit=extract_curves(tracking)
    cards=[]
    names=dict(zip(players.nfl_id.astype(int),players.display_name))
    for player_id,qa in audit.groupby('nfl_id',sort=True):
        aa=attempts[attempts.nfl_id.eq(player_id)]
        turns=events[events.attempt_id.isin(set(aa.attempt_id))].sort_values(['event_id','attempt','reversal_time_seconds'])
        card={'nfl_id':int(player_id),'display_name':str(names.get(int(player_id),'Unknown')),
              'draft_year':int(qa.draft_year.iloc[0]),'official_attempt_count':len(qa),
              'complete_curve_attempt_count':len(aa),'complete_curve_turns':len(turns),
              'quality_status':qa.quality.tolist(),
              'repeatability_established':False,
              'measurement_gate':'One official attempt: compare execution context and film; do not interpret as a stable talent grade.',
              'curve_window_seconds':TIME_GRID.tolist(),'turns':[]}
        for _,turn in turns.iterrows():
            card['turns'].append({'event_id':str(turn.event_id),'attempt':int(turn.attempt),
                                  'reversal_time_from_attempt_start_seconds':float(turn.reversal_time_seconds),
                                  'incoming_lateral_speed_yards_per_second':float(turn.incoming_speed),
                                  'normalized_half_second_recovery_cost_seconds':float(turn.normalized_deficit),
                                  'outgoing_y_direction':int(turn.direction),
                                  'outgoing_signed_speed_over_entry_speed_curve':[float(turn[c]) for c in CURVE]})
        if len(turns):
            card['incoming_speed_range_yards_per_second']=[float(turns.incoming_speed.min()),float(turns.incoming_speed.max())]
            card['recovery_cost_range_seconds']=[float(turns.normalized_deficit.min()),float(turns.normalized_deficit.max())]
            card['lateral_excursion_p90_p10_yards']=float(aa.lateral_amplitude.median())
        card['coach_actions']=['Review the listed turns in film and compare entry speed, excursion and cue sequence.',
                               'Collect another standardized attempt before estimating between-attempt consistency or ranking athletes.',
                               'Record coach stimulus timing separately if the intended quantity is reaction latency.']
        cards.append(card)
    output={'status':'executed_Combine_only_film_review_aids','NFL_outcomes_used_for_cards_or_selection':False,
            'players':len(cards),'official_attempts':len(audit),'usable_attempts':len(attempts),
            'repeated_player_attempts':int(audit.groupby('nfl_id').size().ge(2).sum()),
            'example_selection_rule':'Lowest NFL identifier among usable OL mirror players; independent of NFL outcome',
            'example_nfl_id':int(attempts.nfl_id.min()),'cards':cards}
    if args.save_player_cards:
        (args.output_dir/'SCOUTING_CARDS.json').write_text(json.dumps(output,indent=2))
    example=next(c for c in cards if c['nfl_id']==output['example_nfl_id'])
    if args.save_player_cards:
        (args.output_dir/'EXAMPLE_FILM_CARD.json').write_text(json.dumps(example,indent=2))
    draw_film_card(tracking,example,args.output_dir)
    aggregate={k:output[k] for k in output if k!='cards'}
    aggregate['example_display_name']=example['display_name']
    aggregate['example_entry_speed_range_yards_s']=example['incoming_speed_range_yards_per_second']
    aggregate['example_recovery_cost_range_seconds']=example['recovery_cost_range_seconds']
    aggregate['example_lateral_excursion_p90_p10_yards']=example['lateral_excursion_p90_p10_yards']
    (args.output_dir/'MEASUREMENT_AUDIT.json').write_text(json.dumps(aggregate,indent=2))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
