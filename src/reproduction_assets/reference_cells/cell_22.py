def read_selected(path,predicate):
    # Record selection examines only cohort/game/player keys. Discarded NFL
    # outcomes never enter a typed DataFrame, analysis, model, or output.
    buf=io.StringIO()
    with path.open(newline='') as f:
        reader=csv.reader(f);header=next(reader);index={k:i for i,k in enumerate(header)}
        writer=csv.writer(buf);writer.writerow(header)
        for row in reader:
            if len(row)!=len(header):raise ValueError('Malformed CSV record in '+path.name)
            if predicate(row,index):writer.writerow(row)
    buf.seek(0)
    return pd.read_csv(buf,low_memory=False)

def load_inputs(data_dir,years=(2023,2024),final_enabled=False):
    years=set(int(y) for y in years)
    if not years <= {2023,2024,2025}:raise ValueError('Unexpected cohort year.')
    if 2025 in years and not final_enabled:
        raise PermissionError('2025 loader blocked BEFORE any file is opened.')
    allowed={str(y) for y in years};d=Path(data_dir)
    players=read_selected(d/'players.csv',lambda r,k:r[k['draft_year']] in allowed)
    results=read_selected(d/'combine_results.csv',lambda r,k:r[k['draft_year']] in allowed)
    if not players.nfl_id.is_unique or results.duplicated(['draft_year','nfl_id']).any():
        raise ValueError('Non-unique player/Combine linkage key.')
    ol=results[results.combine_position.eq('OL')].copy()
    ids={str(int(x)) for x in ol.nfl_id};draft=dict(zip(players.nfl_id.astype(str),players.draft_year.astype(int)))
    games=read_selected(d/'games.csv',lambda r,k:r[k['season']] in allowed)
    if not games.game_id.is_unique:raise ValueError('Duplicate game identifiers.')
    reg=games[games.season_type.eq('REG')]
    gameyear=dict(zip(reg.game_id.astype(str),reg.season.astype(int)))
    tracking=read_selected(d/'combine_tracking.csv',lambda r,k:
        r[k['draft_year']] in allowed and r[k['nfl_id']] in ids and
        r[k['entity_type']]=='PLAYER' and r[k['drill_name']].replace('-','_')=='PASS_PRO_MIRROR_DRILL')
    def rookie_record(r,k):
        game=r[k['game_id']];nid=r[k['nfl_id']]
        return game in gameyear and nid in ids and gameyear[game]==draft.get(nid)
    player_play=read_selected(d/'player_play.csv',rookie_record)
    assert set(players.draft_year)<=years and set(results.draft_year)<=years
    assert set(tracking.draft_year)<=years and set(games.season)<=years
    for a in years:
        for b in years:
            if a<b:assert not set(players[players.draft_year.eq(a)].nfl_id)&set(players[players.draft_year.eq(b)].nfl_id)
    if player_play.duplicated(['game_id','play_id','nfl_id']).any():raise ValueError('Duplicate player-play key.')
    return {'players':players,'results':results,'games':games,'tracking':tracking,'player_play':player_play,
            'audit':{'years':sorted(years),'accepted_rows':{'players':len(players),'combine_results':len(results),
                     'combine_tracking_mirror':len(tracking),'games':len(games),'rookie_REG_player_play':len(player_play)},
                     'drill_aliases':sorted(tracking.drill_name.unique().tolist()),'initial_OL_cohort_source':'combine_results.combine_position == OL',
                     'outcomes_selected_by_keys_before_typing':True,'career_file_opened':False}}

def extract(t, window=5, exit_threshold=1.0):
    rows=[]; events=[]; examples={}
    for key,g in t.groupby(['draft_year','event_id','nfl_id','attempt'],sort=True):
        g=g.sort_values('time').copy()
        tm=pd.to_datetime(g.time); sec=(tm-tm.iloc[0]).dt.total_seconds().to_numpy()
        x=g.x.to_numpy(); y=g.y.to_numpy()
        start=np.median(x[:min(10,len(x))])
        exits=np.flatnonzero((np.abs(x-start)>exit_threshold)&(sec>0.5))
        stop=int(exits[0]) if len(exits) else len(x)
        quality='ok'
        if stop<15 or np.any(np.abs(np.diff(sec[:stop])-.1)>.015): quality='short_or_non_10hz'
        if np.isnan(x).any() or np.isnan(y).any(): quality='missing_coordinates'
        rev=[]; deficits=[]
        if quality=='ok':
            sy=savgol_filter(y[:stop],window,2)
            vel=np.gradient(sy,sec[:stop])
            # Zero-speed frames are real at turns: bridge near-zero intervals,
            # rather than requiring adjacent nonzero signs (which misses minima).
            signs=np.where(np.abs(vel)>.05,np.sign(vel),0)
            nz=np.flatnonzero(signs)
            changes=[int(round((i+j)/2)) for i,j in zip(nz[:-1],nz[1:])
                     if signs[i]*signs[j]<0 and j-i<=10]
            last=-100
            for i in changes:
                if i<5 or i+5>=stop or i-last<5: continue
                pre=np.median(vel[i-5:i-1]); post=np.median(vel[i+1:i+5])
                if abs(pre)<.5 or abs(post)<.5 or pre*post>=0: continue
                new=np.sign(post)*vel[i:i+6]
                deficit=float(trapezoid(np.maximum(0,abs(pre)-new),sec[i:i+6]-sec[i]))
                rev.append(i); deficits.append(deficit); last=i
                events.append({'draft_year':key[0],'event_id':key[1],'nfl_id':key[2],'attempt':key[3],
                               'reversal_time_seconds':float(sec[i]),'incoming_speed_yards_s':float(abs(pre)),
                               'recovery_deficit_yards':deficit})
        if len(deficits)<2 and quality=='ok': quality='fewer_than_two_usable_reversals'
        rows.append({'draft_year':key[0],'event_id':key[1],'nfl_id':key[2],'attempt':key[3],
                     'frames':len(g),'mirror_frames':stop,'usable_reversals':len(deficits),
                     'quality':quality,'recovery_deficit':float(np.percentile(deficits,75)) if quality=='ok' else np.nan,
                     'x_start':float(start),'x_range':float(np.ptp(x)),'y_range':float(np.ptp(y))})
        examples[key]={'sec':sec,'x':x,'y':y,'stop':stop,'rev':rev,'quality':quality}
    return pd.DataFrame(rows),pd.DataFrame(events),examples

def extract_mirror(t):
    keys=['draft_year','event_id','nfl_id','attempt','time']
    if t.duplicated(keys).any():raise ValueError('Duplicate tracking timestamp key; extraction refused.')
    if t.empty:raise ValueError('No mirror trajectories; metric is undefined, not zero.')
    if not t.x.between(0,120).all() or not t.y.between(0,53.5).all():
        raise ValueError('Coordinate bounds incompatible with documented field-yard units.')
    return extract(t)

def build_features(t):
    attempts,events,examples=extract_mirror(t)
    keys=['draft_year','event_id','nfl_id','attempt']
    good=attempts[attempts.quality.eq('ok')]
    if good.empty or events.empty:raise ValueError('No usable reversals; metric is undefined.')
    e=events.merge(good[keys],on=keys,validate='many_to_one')
    e['normalized_deficit']=e.recovery_deficit_yards/e.incoming_speed_yards_s
    a=e.groupby(keys,as_index=False).agg(normalized_deficit=('normalized_deficit',lambda z:np.percentile(z,75)),
        incoming_speed=('incoming_speed_yards_s','median'))
    task=[]
    for _,row in good.iterrows():
        key=(int(row.draft_year),row.event_id,int(row.nfl_id),int(row.attempt));ex=examples[key]
        yy=ex['y'][:ex['stop']]
        task.append(dict(zip(keys,key),lateral_amplitude=float(np.percentile(yy,90)-np.percentile(yy,10)),
                         phase_seconds=float(ex['sec'][ex['stop']-1])))
    z=good.merge(a,on=keys,validate='one_to_one').merge(pd.DataFrame(task),on=keys,validate='one_to_one')
    features=z.groupby(['draft_year','nfl_id'],as_index=False).agg(
        normalized_deficit=('normalized_deficit','median'),incoming_speed=('incoming_speed','median'),
        lateral_amplitude=('lateral_amplitude','median'),raw_deficit=('recovery_deficit','median'),
        usable_reversals=('usable_reversals','sum'),usable_attempts=('attempt','size'),phase_seconds=('phase_seconds','median'))
    return features,attempts,events,examples

def prepare_cohort(inputs,years=(2023,2024),endpoint='charted_all'):
    features,attempts,events,examples=build_features(inputs['tracking'][inputs['tracking'].draft_year.isin(years)])
    c=inputs['results'];p=inputs['players'];g=inputs['games']
    c=c[c.draft_year.isin(years)&c.combine_position.eq('OL')]
    base=p[p.draft_year.isin(years)].merge(c[['draft_year','nfl_id']+BASE],on=['draft_year','nfl_id'],validate='one_to_one')
    base=base.merge(features,on=['draft_year','nfl_id'],validate='one_to_one')
    # NFL position is retained ONLY for explicitly labeled historical sensitivity.
    base['historical_is_tackle']=base.nfl_position.eq('T').astype(int)
    q=inputs['player_play'].merge(p[['nfl_id','draft_year']],on='nfl_id',validate='many_to_one').merge(
        g[['game_id','season','season_type']],on='game_id',validate='many_to_one')
    q=q[q.draft_year.isin(years)&q.season.eq(q.draft_year)&q.season_type.eq('REG')].copy()
    good=q.pressure_allowed.notna()&q.lined_up_position.isin(['C','G','T'])&q.team_abbr.eq(q.possession_team)&q.play_nullified_by_penalty.eq('N')
    observed=q[good].copy()
    labels=observed.pressure_allowed.map(lambda z:str(z).lower())
    if not set(labels)<= {'true','false'}:raise ValueError('Unknown charted pressure boolean; no coercion to zero allowed.')
    observed['pressure']=labels.eq('true').astype(int)
    if endpoint=='encountered_at_least_one':observed=observed[observed.pass_rushers_encountered.ge(1)]
    elif endpoint=='pass_results_C_I_IN_S':observed=observed[observed.pass_result.isin(['C','I','IN','S'])]
    elif endpoint=='encountered_and_C_I_IN_S':observed=observed[observed.pass_rushers_encountered.ge(1)&observed.pass_result.isin(['C','I','IN','S'])]
    elif endpoint!='charted_all':raise ValueError('Unrecognized endpoint; no silent substitute.')
    outcome=observed.groupby(['draft_year','nfl_id'],as_index=False).agg(n=('pressure','size'),k=('pressure','sum'))
    outcome['rate']=outcome.k/outcome.n
    allplayers=base.merge(outcome,on=['draft_year','nfl_id'],how='left',validate='one_to_one')
    eligible=allplayers[allplayers.n.ge(50)].copy()
    funnel={str(year):{'Combine_OL_participants':int(c.draft_year.eq(year).sum()),
        'mirror_participants':int(attempts[attempts.draft_year.eq(year)].nfl_id.nunique()),
        'usable_feature_players':int(base.draft_year.eq(year).sum()),'any_rookie_REG_players':int(q[q.draft_year.eq(year)].nfl_id.nunique()),
        'observed_endpoint_players':int((allplayers.draft_year.eq(year)&allplayers.n.notna()).sum()),
        'eligible_players_min50':int(eligible.draft_year.eq(year).sum()),
        'eligible_opportunities':int(eligible.loc[eligible.draft_year.eq(year),'n'].sum()),
        'eligible_pressures':int(eligible.loc[eligible.draft_year.eq(year),'k'].sum()),
        'missing_charted_pressure_rows':int((q.draft_year.eq(year)&q.pressure_allowed.isna()).sum()),
        'observed_zero_rusher_rows':int((observed.draft_year.eq(year)&observed.pass_rushers_encountered.eq(0)).sum())}
        for year in years}
    return {'eligible':eligible,'all_players':allplayers,'funnel':funnel,'attempts':attempts,'events':events,
            'examples':examples,'features':features,'rookie_rows':q,'observed_rows':observed}

def default_specs():
    return {
        'constant_equal_player': {'columns': [], 'family': 'mean'},
        'original_conventional_fixed10': {'columns': BASE, 'family': 'ridge', 'fixed_penalty': 10},
        'original_full_fixed1': {'columns': BASE + FAMILY, 'family': 'ridge', 'fixed_penalty': 1},
        'nested_conventional_ridge': {'columns': BASE, 'family': 'ridge'},
        'nested_full_ridge': {'columns': BASE + FAMILY, 'family': 'ridge'},
        'beta_binomial_constant': {'columns': [], 'family': 'beta_binomial'},
        'beta_binomial_conventional': {'columns': BASE, 'family': 'beta_binomial'},
        'beta_binomial_full': {'columns': BASE + FAMILY, 'family': 'beta_binomial'},
        'binomial_full': {'columns': BASE + FAMILY, 'family': 'binomial'},
    }