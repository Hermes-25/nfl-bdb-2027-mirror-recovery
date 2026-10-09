import csv, io, os, json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import matplotlib
import matplotlib.pyplot as plt

# Root must explicitly authorize and freeze this branch before opening 2025.
FINAL_EVALUATION_ENABLED = True
FINAL_SELECTION_AUTHORIZATION = 'OL_selected_by_root'
FROZEN_BASELINE_ALPHA = 10
FROZEN_ENHANCED_ALPHA = 1
SEED = 73025
GRID = [1.0, 10.0, 100.0]
FIXED_BOOT = 5000
REFIT_BOOT = 250
BASE = ['combine_height','combine_weight','forty','ten_yd_split']
FAMILY = ['normalized_deficit','incoming_speed','lateral_amplitude']
MODEL_COLUMNS = {'conventional': BASE, 'normalized': BASE+FAMILY[:1],
                 'speed_amplitude': BASE+FAMILY[1:], 'full_family': BASE+FAMILY}
REQUIRED = ['players.csv','combine_results.csv','combine_tracking.csv','games.csv','player_play.csv']
WORK = Path.cwd()
VERSIONS = {'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
            'matplotlib':matplotlib.__version__}
print('Runtime:', VERSIONS)
print('Reproduce development first. Reserved evaluation enabled:', FINAL_EVALUATION_ENABLED)

def locate_data():
    explicit=os.environ.get('NFL_DATA_DIR')
    if explicit:
        p=Path(explicit)
        if not all((p/f).is_file() for f in REQUIRED):
            raise FileNotFoundError('NFL_DATA_DIR must contain the five official CSV files.')
        return p
    verified=Path('/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027')
    if all((verified/f).is_file() for f in REQUIRED):return verified
    candidates=[p.parent for p in Path('/kaggle/input').glob('**/players.csv')
                if all((p.parent/f).is_file() for f in REQUIRED)]
    if len(candidates)!=1:raise FileNotFoundError('Set NFL_DATA_DIR explicitly; input directory is ambiguous or absent.')
    return candidates[0]

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

CODE_FINGERPRINT = '837b80cf6446558b03c6af2c210132af262e668c2932b40f5b425e5be5550b9b'
