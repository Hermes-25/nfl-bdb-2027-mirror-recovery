"""Self-contained retrospective coverage, validation and film-support appendix.

Only 2023/24 official records are processed. The original 2025 result is an
immutable archived reference, not recomputed or claimed as a new holdout.
Run with --raw-dir <five-official-CSV-directory> --output-dir <output-directory>.
Outputs only aggregate JSON, figures and a source-code ZIP. No player-row CSVs.
"""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import copy,zipfile,base64,zlib
from scipy.spatial import ConvexHull,QhullError
import csv, io, os, json, hashlib, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import matplotlib
import matplotlib.pyplot as plt

def locate_data():
    explicit = os.environ.get('NFL_DATA_DIR')
    if explicit:
        p = Path(explicit)
        if not all(((p / f).is_file() for f in REQUIRED)):
            raise FileNotFoundError('NFL_DATA_DIR must contain the five official CSV files.')
        return p
    verified = Path('/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027')
    if all(((verified / f).is_file() for f in REQUIRED)):
        return verified
    candidates = [p.parent for p in Path('/kaggle/input').glob('**/players.csv') if all(((p.parent / f).is_file() for f in REQUIRED))]
    if len(candidates) != 1:
        raise FileNotFoundError('Set NFL_DATA_DIR explicitly; input directory is ambiguous or absent.')
    return candidates[0]

def read_selected(path, predicate):
    buf = io.StringIO()
    with path.open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        index = {k: i for i, k in enumerate(header)}
        writer = csv.writer(buf)
        writer.writerow(header)
        for row in reader:
            if len(row) != len(header):
                raise ValueError('Malformed CSV record in ' + path.name)
            if predicate(row, index):
                writer.writerow(row)
    buf.seek(0)
    return pd.read_csv(buf, low_memory=False)

def load_inputs(data_dir, years=(2023, 2024), final_enabled=False):
    years = set((int(y) for y in years))
    if not years <= {2023, 2024, 2025}:
        raise ValueError('Unexpected cohort year.')
    if 2025 in years and (not final_enabled):
        raise PermissionError('2025 loader blocked BEFORE any file is opened.')
    allowed = {str(y) for y in years}
    d = Path(data_dir)
    players = read_selected(d / 'players.csv', lambda r, k: r[k['draft_year']] in allowed)
    results = read_selected(d / 'combine_results.csv', lambda r, k: r[k['draft_year']] in allowed)
    if not players.nfl_id.is_unique or results.duplicated(['draft_year', 'nfl_id']).any():
        raise ValueError('Non-unique player/Combine linkage key.')
    ol = results[results.combine_position.eq('OL')].copy()
    ids = {str(int(x)) for x in ol.nfl_id}
    draft = dict(zip(players.nfl_id.astype(str), players.draft_year.astype(int)))
    games = read_selected(d / 'games.csv', lambda r, k: r[k['season']] in allowed)
    if not games.game_id.is_unique:
        raise ValueError('Duplicate game identifiers.')
    reg = games[games.season_type.eq('REG')]
    gameyear = dict(zip(reg.game_id.astype(str), reg.season.astype(int)))
    tracking = read_selected(d / 'combine_tracking.csv', lambda r, k: r[k['draft_year']] in allowed and r[k['nfl_id']] in ids and (r[k['entity_type']] == 'PLAYER') and (r[k['drill_name']].replace('-', '_') == 'PASS_PRO_MIRROR_DRILL'))

    def rookie_record(r, k):
        game = r[k['game_id']]
        nid = r[k['nfl_id']]
        return game in gameyear and nid in ids and (gameyear[game] == draft.get(nid))
    player_play = read_selected(d / 'player_play.csv', rookie_record)
    assert set(players.draft_year) <= years and set(results.draft_year) <= years
    assert set(tracking.draft_year) <= years and set(games.season) <= years
    for a in years:
        for b in years:
            if a < b:
                assert not set(players[players.draft_year.eq(a)].nfl_id) & set(players[players.draft_year.eq(b)].nfl_id)
    if player_play.duplicated(['game_id', 'play_id', 'nfl_id']).any():
        raise ValueError('Duplicate player-play key.')
    return {'players': players, 'results': results, 'games': games, 'tracking': tracking, 'player_play': player_play, 'audit': {'years': sorted(years), 'accepted_rows': {'players': len(players), 'combine_results': len(results), 'combine_tracking_mirror': len(tracking), 'games': len(games), 'rookie_REG_player_play': len(player_play)}, 'drill_aliases': sorted(tracking.drill_name.unique().tolist()), 'initial_OL_cohort_source': 'combine_results.combine_position == OL', 'outcomes_selected_by_keys_before_typing': True, 'career_file_opened': False}}

def extract(t, window=5, exit_threshold=1.0):
    rows = []
    events = []
    examples = {}
    for key, g in t.groupby(['draft_year', 'event_id', 'nfl_id', 'attempt'], sort=True):
        g = g.sort_values('time').copy()
        tm = pd.to_datetime(g.time)
        sec = (tm - tm.iloc[0]).dt.total_seconds().to_numpy()
        x = g.x.to_numpy()
        y = g.y.to_numpy()
        start = np.median(x[:min(10, len(x))])
        exits = np.flatnonzero((np.abs(x - start) > exit_threshold) & (sec > 0.5))
        stop = int(exits[0]) if len(exits) else len(x)
        quality = 'ok'
        if stop < 15 or np.any(np.abs(np.diff(sec[:stop]) - 0.1) > 0.015):
            quality = 'short_or_non_10hz'
        if np.isnan(x).any() or np.isnan(y).any():
            quality = 'missing_coordinates'
        rev = []
        deficits = []
        if quality == 'ok':
            sy = savgol_filter(y[:stop], window, 2)
            vel = np.gradient(sy, sec[:stop])
            signs = np.where(np.abs(vel) > 0.05, np.sign(vel), 0)
            nz = np.flatnonzero(signs)
            changes = [int(round((i + j) / 2)) for i, j in zip(nz[:-1], nz[1:]) if signs[i] * signs[j] < 0 and j - i <= 10]
            last = -100
            for i in changes:
                if i < 5 or i + 5 >= stop or i - last < 5:
                    continue
                pre = np.median(vel[i - 5:i - 1])
                post = np.median(vel[i + 1:i + 5])
                if abs(pre) < 0.5 or abs(post) < 0.5 or pre * post >= 0:
                    continue
                new = np.sign(post) * vel[i:i + 6]
                deficit = float(trapezoid(np.maximum(0, abs(pre) - new), sec[i:i + 6] - sec[i]))
                rev.append(i)
                deficits.append(deficit)
                last = i
                events.append({'draft_year': key[0], 'event_id': key[1], 'nfl_id': key[2], 'attempt': key[3], 'reversal_time_seconds': float(sec[i]), 'incoming_speed_yards_s': float(abs(pre)), 'recovery_deficit_yards': deficit})
        if len(deficits) < 2 and quality == 'ok':
            quality = 'fewer_than_two_usable_reversals'
        rows.append({'draft_year': key[0], 'event_id': key[1], 'nfl_id': key[2], 'attempt': key[3], 'frames': len(g), 'mirror_frames': stop, 'usable_reversals': len(deficits), 'quality': quality, 'recovery_deficit': float(np.percentile(deficits, 75)) if quality == 'ok' else np.nan, 'x_start': float(start), 'x_range': float(np.ptp(x)), 'y_range': float(np.ptp(y))})
        examples[key] = {'sec': sec, 'x': x, 'y': y, 'stop': stop, 'rev': rev, 'quality': quality}
    return (pd.DataFrame(rows), pd.DataFrame(events), examples)

def extract_mirror(t):
    keys = ['draft_year', 'event_id', 'nfl_id', 'attempt', 'time']
    if t.duplicated(keys).any():
        raise ValueError('Duplicate tracking timestamp key; extraction refused.')
    if t.empty:
        raise ValueError('No mirror trajectories; metric is undefined, not zero.')
    if not t.x.between(0, 120).all() or not t.y.between(0, 53.5).all():
        raise ValueError('Coordinate bounds incompatible with documented field-yard units.')
    return extract(t)

def build_features(t):
    attempts, events, examples = extract_mirror(t)
    keys = ['draft_year', 'event_id', 'nfl_id', 'attempt']
    good = attempts[attempts.quality.eq('ok')]
    if good.empty or events.empty:
        raise ValueError('No usable reversals; metric is undefined.')
    e = events.merge(good[keys], on=keys, validate='many_to_one')
    e['normalized_deficit'] = e.recovery_deficit_yards / e.incoming_speed_yards_s
    a = e.groupby(keys, as_index=False).agg(normalized_deficit=('normalized_deficit', lambda z: np.percentile(z, 75)), incoming_speed=('incoming_speed_yards_s', 'median'))
    task = []
    for _, row in good.iterrows():
        key = (int(row.draft_year), row.event_id, int(row.nfl_id), int(row.attempt))
        ex = examples[key]
        yy = ex['y'][:ex['stop']]
        task.append(dict(zip(keys, key), lateral_amplitude=float(np.percentile(yy, 90) - np.percentile(yy, 10)), phase_seconds=float(ex['sec'][ex['stop'] - 1])))
    z = good.merge(a, on=keys, validate='one_to_one').merge(pd.DataFrame(task), on=keys, validate='one_to_one')
    features = z.groupby(['draft_year', 'nfl_id'], as_index=False).agg(normalized_deficit=('normalized_deficit', 'median'), incoming_speed=('incoming_speed', 'median'), lateral_amplitude=('lateral_amplitude', 'median'), raw_deficit=('recovery_deficit', 'median'), usable_reversals=('usable_reversals', 'sum'), usable_attempts=('attempt', 'size'), phase_seconds=('phase_seconds', 'median'))
    return (features, attempts, events, examples)
import argparse, ast, hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
SEED = 87007
BASE = ['combine_height', 'combine_weight', 'forty', 'ten_yd_split']
MIRROR = ['normalized_deficit', 'incoming_speed', 'lateral_amplitude']
THRESHOLDS = [1, 25, 50, 100, 250]
CANDIDATES = [('constant', None)] + [(f, a) for f in ('combine', 'combine_mirror') for a in (1.0, 10.0, 100.0)]

def build_all_ol(raw: Path, years: list[int], source=None):
    ns = globals()
    inputs = ns['load_inputs'](raw, years, final_enabled=2025 in years)
    features, attempts, events, _ = ns['build_features'](inputs['tracking'])
    c = inputs['results'].query("combine_position == 'OL'").copy()
    p = inputs['players']
    g = inputs['games']
    base = c.merge(p[['draft_year', 'nfl_id', 'nfl_position']], on=['draft_year', 'nfl_id'], how='left', validate='one_to_one')
    base = base.merge(features, on=['draft_year', 'nfl_id'], how='left', validate='one_to_one')
    q = inputs['player_play'].merge(p[['draft_year', 'nfl_id']], on='nfl_id', validate='many_to_one').merge(g[['game_id', 'season', 'season_type']], on='game_id', validate='many_to_one')
    q = q[q.season.eq(q.draft_year) & q.season_type.eq('REG')].copy()
    eligible_rows = q.lined_up_position.isin(['C', 'G', 'T']) & q.team_abbr.eq(q.possession_team) & q.play_nullified_by_penalty.eq('N')
    q = q[eligible_rows].copy()
    observed = q[q.pressure_allowed.notna()].copy()
    labels = observed.pressure_allowed.astype(str).str.lower()
    if not set(labels) <= {'true', 'false'}:
        raise ValueError('Unknown explicit pressure label.')
    observed['pressure'] = labels.eq('true').astype(int)
    outcome = observed.groupby(['draft_year', 'nfl_id'], as_index=False).agg(n=('pressure', 'size'), k=('pressure', 'sum'))
    outcome['rate'] = outcome.k / outcome.n
    base = base.merge(outcome, on=['draft_year', 'nfl_id'], how='left', validate='one_to_one')
    anyreg = inputs['player_play'].groupby('nfl_id').size().rename('rookie_REG_rows')
    base = base.merge(anyreg, on='nfl_id', how='left', validate='one_to_one')
    present = set(attempts.nfl_id)
    base['mirror_present'] = base.nfl_id.isin(present)
    base['mirror_usable'] = base[MIRROR].notna().all(axis=1)
    base['observed_endpoint'] = base.n.notna()
    base['min50'] = base.n.ge(50) & base.mirror_usable
    audit = {'source_sha256': ORIGINAL_SOURCE_SHA256, 'input_audit': inputs['audit'], 'endpoint': 'rookie regular-season, offensive C/G/T, possession team, non-null explicit pressure_allowed, non-nullified', 'missing_pressure_rows_after_football_filter': int(q.pressure_allowed.isna().sum()), 'observed_zero_rusher_rows': int(observed.pass_rushers_encountered.eq(0).sum()), 'unconditional_talent_transport_identified': False}
    return (base, audit)

def loss(y, p):
    return y * (1 - p) ** 2 + (1 - y) * p * p

def preprocess(train, test, cols):
    x = train[cols].to_numpy(float)
    xt = test[cols].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    x = np.where(np.isfinite(x), x, med)
    xt = np.where(np.isfinite(xt), xt, med)
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std > 1e-12, std, 1.0)
    return ((x - mean) / std, (xt - mean) / std)

def predict(train, test, candidate):
    family, alpha = candidate
    y = train.rate.to_numpy(float)
    if family == 'constant':
        return np.full(len(test), y.mean())
    cols = BASE + (MIRROR if family == 'combine_mirror' else [])
    x, xt = preprocess(train, test, cols)
    coef = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    return np.clip(y.mean() + xt @ coef, 0.001, 0.999)

def folds(frame, k, seed, grouped=False):
    strata = frame.draft_year.to_numpy()
    groups = frame.nfl_id.to_numpy()
    if grouped or len(np.unique(groups)) < len(groups):
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(frame)), strata, groups))
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(frame)), strata))

def select_candidate(train, inner_folds, family=None):
    candidates = [c for c in CANDIDATES if family is None or c[0] == family]
    scores = []
    for c in candidates:
        p = np.full(len(train), np.nan)
        for ix, jx in inner_folds:
            p[jx] = predict(train.iloc[ix], train.iloc[jx], c)
        scores.append(float(loss(train.rate.to_numpy(), p).mean()))
    best = min(range(len(candidates)), key=lambda i: (scores[i], 0 if candidates[i][0] == 'constant' else 1 if candidates[i][0] == 'combine' else 2, -(candidates[i][1] or 0)))
    return (candidates[best], dict(((f'{c[0]}:{c[1]}', s) for c, s in zip(candidates, scores))))

def id_folds(frame, records):
    """Map immutable player-ID train/test assignments onto arbitrary row order."""
    ids = frame.nfl_id.to_numpy()
    seen = []
    result = []
    for r in records:
        train_ids = set((int(v) for v in r.get('train_ids', r.get('train_player_ids', []))))
        test_ids = set((int(v) for v in r.get('test_ids', r.get('test_player_ids', []))))
        assert train_ids and test_ids and (not train_ids & test_ids)
        assert train_ids | test_ids == set((int(v) for v in ids)), 'Common fold universe differs from eligible cohort.'
        tr = np.flatnonzero(np.isin(ids, list(train_ids)))
        te = np.flatnonzero(np.isin(ids, list(test_ids)))
        result.append((tr, te))
        seen.extend((int(v) for v in ids[te]))
    assert sorted(seen) == sorted((int(v) for v in ids)), 'Every player must be tested once in common fold map.'
    return result

def map_records(mapping):
    if isinstance(mapping, list):
        return mapping
    for key in ['outer_folds', 'folds', 'outer']:
        if key in mapping:
            return mapping[key]
    raise ValueError('Common fold map must contain outer_folds with player-ID assignments.')

def nested_predictions(frame, seed=SEED, common_folds=None):
    out = pd.DataFrame({'draft_year': frame.draft_year, 'nfl_id': frame.nfl_id, 'rate': frame.rate, 'n': frame.n})
    for name in ['constant', 'combine', 'combine_mirror', 'selected_procedure']:
        out[name] = np.nan
    records = []
    outer_records = map_records(common_folds) if common_folds is not None else None
    outer = id_folds(frame, outer_records) if outer_records is not None else folds(frame, 5, seed)
    for fi, (tr, te) in enumerate(outer):
        train = frame.iloc[tr].reset_index(drop=True)
        test = frame.iloc[te].reset_index(drop=True)
        if outer_records is not None:
            inner_records = outer_records[fi].get('inner_folds', outer_records[fi].get('inner'))
            if inner_records is None:
                raise ValueError('Common outer map must include immutable inner_folds.')
            inner = id_folds(train, inner_records)
        else:
            inner = folds(train, 3, seed + 100 + fi)
        assert not set(train.nfl_id) & set(test.nfl_id), 'Player leaked outer fold.'
        for ii, jj in inner:
            assert not set(train.iloc[ii].nfl_id) & set(train.iloc[jj].nfl_id), 'Player leaked inner fold.'
        rec = {'fold': fi, 'train_ids': [int(x) for x in train.nfl_id], 'test_ids': [int(x) for x in test.nfl_id], 'inner_grid': {}}
        out.loc[out.index[te], 'constant'] = predict(train, test, ('constant', None))
        for family in ['combine', 'combine_mirror']:
            c, grid = select_candidate(train, inner, family)
            out.loc[out.index[te], family] = predict(train, test, c)
            rec[family] = {'chosen': c, 'scores': grid}
        c, grid = select_candidate(train, inner)
        out.loc[out.index[te], 'selected_procedure'] = predict(train, test, c)
        rec['selected_procedure'] = {'chosen': c, 'scores': grid}
        records.append(rec)
    assert out[['constant', 'combine', 'combine_mirror', 'selected_procedure']].notna().all().all()
    return (out, records)

def availability_predict(train, test, columns):
    x, xt = preprocess(train, test, columns)
    y = train.min50.astype(int).to_numpy()
    prevalence = np.clip(y.mean(), 1e-06, 1 - 1e-06)
    if y.min() == y.max():
        return np.full(len(test), prevalence)
    z = np.column_stack([np.ones(len(x)), x])
    zt = np.column_stack([np.ones(len(xt)), xt])
    from scipy.special import expit

    def obj(b):
        linear = z @ b
        val = (np.logaddexp(0, linear) - y * linear).sum() + 5.0 * np.sum(b[1:] ** 2)
        grad = z.T @ (expit(linear) - y)
        grad[1:] += 10.0 * b[1:]
        return (val, grad)
    fitted = minimize(obj, np.r_[np.log(prevalence / (1 - prevalence)), np.zeros(x.shape[1])], jac=True, method='L-BFGS-B')
    if not fitted.success:
        raise RuntimeError('Availability logistic fit failed: ' + fitted.message)
    return expit(zt @ fitted.x)

def coverage_audit(all_players):
    funnel = []
    for year, g in all_players.groupby('draft_year'):
        funnel.append({'draft_year': int(year), 'Combine_OL': len(g), 'mirror_present': int(g.mirror_present.sum()), 'usable_mirror': int(g.mirror_usable.sum()), 'any_rookie_REG': int(g.rookie_REG_rows.notna().sum()), 'observed_endpoint': int(g.observed_endpoint.sum()), 'eligible_min50_and_mirror': int(g.min50.sum())})
    rows = []
    for threshold in THRESHOLDS:
        keep = all_players.mirror_usable & all_players.n.ge(threshold)
        for year, g in all_players.groupby('draft_year'):
            select = g.mirror_usable & g.n.ge(threshold)
            rows.append({'threshold': threshold, 'draft_year': int(year), 'retained': int(select.sum()), 'total': len(g), 'charted_observations': int(g.loc[select, 'n'].sum()), 'pressures': int(g.loc[select, 'k'].sum())})
    smd = []
    selected = all_players.min50
    for c in BASE + MIRROR:
        values = pd.to_numeric(all_players[c], errors='coerce')
        scale = values.std(ddof=0)
        a = values[selected].dropna()
        b = values[~selected].dropna()
        smd.append({'feature': c, 'retained_mean': float(a.mean()), 'excluded_mean': float(b.mean()), 'standardized_mean_difference': float((a.mean() - b.mean()) / scale), 'retained_missing': int(values[selected].isna().sum()), 'excluded_missing': int(values[~selected].isna().sum())})
    pred = np.full(len(all_players), np.nan)
    for tr, te in folds(all_players, 5, SEED):
        pred[te] = availability_predict(all_players.iloc[tr], all_players.iloc[te], BASE + MIRROR)
    y = all_players.min50.astype(int).to_numpy()
    obs = {'purpose': 'Cross-fitted endpoint availability diagnostic, not ability prediction or causal correction.', 'fixed_logistic_L2_penalty': 10.0, 'player_oof_auc': float(roc_auc_score(y, pred)), 'player_oof_average_precision': float(average_precision_score(y, pred)), 'prevalence': float(y.mean()), 'player_oof_brier': float(np.mean((y - pred) ** 2)), 'predicted_min': float(pred.min()), 'predicted_max': float(pred.max()), 'eligible_propensity_range': [float(pred[y == 1].min()), float(pred[y == 1].max())], 'excluded_propensity_range': [float(pred[y == 0].min()), float(pred[y == 0].max())]}
    return (funnel, rows, smd, obs, pred)

def bootstrap_comparison(pred, draws=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = pred.rate.to_numpy()
    diff = loss(y, pred['combine'].to_numpy()) - loss(y, pred['combine_mirror'].to_numpy())
    groups = [np.flatnonzero(pred.draft_year.to_numpy() == yr) for yr in sorted(pred.draft_year.unique())]
    sampled = np.concatenate([rng.choice(ix, size=(draws, len(ix)), replace=True) for ix in groups], axis=1)
    b = diff[sampled].mean(1)
    return {'baseline_minus_mirror_brier': float(diff.mean()), 'fixed_outer_prediction_player_bootstrap_95': np.quantile(b, [0.025, 0.975]).tolist(), 'resampling_unit': 'player within cohort', 'selection_repeated_in_this_interval': False, 'interval_is_not_independent_generalization_guarantee': True}

def retuned_bootstrap(frame, draws):
    rng = np.random.default_rng(SEED + 31)
    records = []
    groups = [np.flatnonzero(frame.draft_year.to_numpy() == yr) for yr in sorted(frame.draft_year.unique())]
    for b in range(draws):
        sampled = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in groups])
        boot = frame.iloc[sampled].reset_index(drop=True)
        try:
            p, r = nested_predictions(boot, seed=SEED + 1000 + b)
        except ValueError as e:
            raise RuntimeError(f'Bootstrap {b} lacked grouped-fold feasibility.') from e
        y = p.rate.to_numpy()
        records.append({'baseline_minus_mirror': float((loss(y, p['combine'].to_numpy()) - loss(y, p['combine_mirror'].to_numpy())).mean()), 'constant_minus_selected': float((loss(y, p['constant'].to_numpy()) - loss(y, p['selected_procedure'].to_numpy())).mean()), 'mirror_selected_folds': sum((q['selected_procedure']['chosen'][0] == 'combine_mirror' for q in r))})
    return records

def cohort_transfer(frame):
    result = []
    for yr in sorted(frame.draft_year.unique()):
        train = frame[frame.draft_year.ne(yr)].reset_index(drop=True)
        test = frame[frame.draft_year.eq(yr)].reset_index(drop=True)
        if len(train) < 10 or len(test) < 5:
            continue
        inner = folds(train, 3, SEED + int(yr))
        rec = {'test_year': int(yr), 'train_years': sorted((int(x) for x in train.draft_year.unique())), 'interpretation': 'Retrospective cross-cohort transfer; backward-facing folds are not forecasting.'}
        y = test.rate.to_numpy()
        for fam in ['constant', 'combine', 'combine_mirror']:
            c, grid = select_candidate(train, inner, fam)
            p = predict(train, test, c)
            rec[fam] = {'brier': float(loss(y, p).mean()), 'selected': c}
        result.append(rec)
    return result

def figures(output, all_players, threshold_rows, smd, pred, availability):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    th = pd.DataFrame(threshold_rows)
    for yr, g in th.groupby('draft_year'):
        ax[0].plot(g.threshold, g.retained / g.total, 'o-', label=str(yr))
    ax[0].set(xlabel='Minimum charted rookie observations', ylabel='Share of Combine OL retained', ylim=(0, 1))
    ax[0].legend(title='Draft cohort')
    d = pd.DataFrame(smd)
    ax[1].barh(d.feature, d.standardized_mean_difference, color='#367786')
    ax[1].axvline(0, color='black', linewidth=0.8)
    ax[1].set(xlabel='Retained − excluded mean / all-prospect SD')
    fig.suptitle('NFL outcome coverage changes the prospect population')
    fig.tight_layout()
    fig.savefig(output / 'coverage_selection.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for flag, label in [(False, 'Excluded'), (True, 'Retained ≥50')]:
        v = availability[all_players.min50.to_numpy() == flag]
        ax[0].hist(v, bins=np.linspace(0, 1, 12), alpha=0.6, label=label)
    ax[0].set(xlabel='Cross-fitted probability of endpoint eligibility', ylabel='Prospects')
    ax[0].legend()
    y = pred.rate.to_numpy()
    values = [loss(y, pred[c].to_numpy()).mean() for c in ['constant', 'combine', 'combine_mirror', 'selected_procedure']]
    ax[1].bar(['Constant', 'Combine', 'Combine +\nmirror', 'Nested selection'], values, color=['#808080', '#367786', '#b86b38', '#745889'])
    ax[1].set(ylabel='Nested outer-fold equal-player Brier loss')
    ax[1].tick_params(axis='x', labelsize=9)
    fig.suptitle('Observability and fully specified selection are distinct questions')
    fig.tight_layout()
    fig.savefig(output / 'nested_validation.png', dpi=180)
    plt.close(fig)
REQUIRED=['players.csv', 'combine_results.csv', 'combine_tracking.csv', 'games.csv', 'player_play.csv']
ORIGINAL_SOURCE_SHA256='9869aa86b0bf6b87a832b984f1efdfb1fac13d9ec6fbe1567fdefa4d6cbb56f6'
COMMON_FOLD_SHA256='4f5ad7a2fef3dbccb0abd23e627546106d90056eace24776f4e673e58a397b34'
COMMON_FOLDS=json.loads(Path(__file__).with_name('common_fold_manifest.json').read_text())
ARCHIVED_REFERENCE=json.loads(Path(__file__).with_name('archived_reference.json').read_text())

FOLD_PLAN_SHA256='28ab0fd1fb3fe896514fbf1aeb9356e218ba0e51d992159f0d1c54ec63945c5e'
FROZEN_SENSITIVITY_PLANS=json.loads(Path(__file__).with_name('sensitivity_plan.json').read_text())
SPLIT_PLANS=FROZEN_SENSITIVITY_PLANS["splits"]
BOOTSTRAP_PLANS=FROZEN_SENSITIVITY_PLANS["bootstrap"]


def retuned_bootstrap(frame,draws):
    if draws>len(BOOTSTRAP_PLANS):raise ValueError('At most 500 frozen bootstrap plans are bundled.')
    rng=np.random.default_rng(SEED+31);records=[]
    groups=[np.flatnonzero(frame.draft_year.to_numpy()==yr) for yr in sorted(frame.draft_year.unique())]
    for b in range(draws):
        sampled=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in groups])
        boot=frame.iloc[sampled].reset_index(drop=True)
        p,r=nested_predictions(boot,common_folds=BOOTSTRAP_PLANS[b])
        y=p.rate.to_numpy()
        records.append({'baseline_minus_mirror':float((loss(y,p['combine'].to_numpy())-loss(y,p['combine_mirror'].to_numpy())).mean()),'constant_minus_selected':float((loss(y,p['constant'].to_numpy())-loss(y,p['selected_procedure'].to_numpy())).mean()),'mirror_selected_folds':sum(q['selected_procedure']['chosen'][0]=='combine_mirror' for q in r)})
    return records

def trim_map(mapping, remove):
    result = copy.deepcopy(mapping)
    for fold in result['outer_folds']:
        for key in ['train_ids', 'test_ids']:
            fold[key] = [x for x in fold[key] if x != remove]
        for inner in fold['inner_folds']:
            for key in ['train_ids', 'test_ids']:
                inner[key] = [x for x in inner[key] if x != remove]
    return result

def summary(pred):
    y = pred.rate.to_numpy()
    scores = {name: float(loss(y, pred[name].to_numpy()).mean()) for name in ['constant', 'combine', 'combine_mirror', 'selected_procedure']}
    scores['combine_minus_mirror'] = scores['combine'] - scores['combine_mirror']
    scores['constant_minus_selected'] = scores['constant'] - scores['selected_procedure']
    return scores
def film_support(frame):
    TASK=['incoming_speed','lateral_amplitude']
    valid=frame.mirror_usable&frame[TASK].notna().all(axis=1)
    ref=frame[valid].reset_index(drop=True)
    x=ref[TASK].to_numpy(float);mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1)
    z=(x-mean)/scale
    rows=[]
    for i,row in ref.iterrows():
     other=np.arange(len(ref))!=i;train=z[other];target=z[i]
     try:
      hull=ConvexHull(train)
      inside=bool(np.all(hull.equations[:,:-1]@target+hull.equations[:,-1]<=1e-10))
     except QhullError:inside=False
     distances=np.sqrt(np.sum((train-target)**2,axis=1));indices=np.flatnonzero(other);closest=indices[np.argsort(distances)[:5]]
     rows.append({'draft_year':int(row.draft_year),'nfl_id':int(row.nfl_id),
                  'feature_only_joint_task_support':inside,
                  'film_comparison_status':'Within measured joint task support; review comparable clips descriptively' if inside else 'Abstain from relative movement ranking: outside measured joint task support',
                  'automatic_pressure_forecast_adjustment':'ABSTAIN: original frozen 2025 test did not confirm improvement',
                  'physiological_repeatability':'UNESTABLISHED: one recorded attempt per player',
                  'median_incoming_speed_yards_s':float(row.incoming_speed),'attempt_excursion_yards':float(row.lateral_amplitude),
                  'normalized_deficit_seconds':float(row.normalized_deficit),'charted_endpoint_eligible_min50':bool(row.min50),
                  'nearest_reference_player_ids':';'.join(str(int(v)) for v in ref.iloc[closest].nfl_id),
                  'nearest_task_distance_standardized':float(distances.min()),
                  'fifth_nearest_task_distance_standardized':float(np.sort(distances)[4])})
    summary={'status':'executed_feature_only_reporting_rule','round':9,'reference_cohorts':[2023,2024],'measured_prospects':len(ref),
             'joint_task_supported':sum(r['feature_only_joint_task_support'] for r in rows),'outside_joint_task_support':sum(not r['feature_only_joint_task_support'] for r in rows),
             'automatic_forecast_adjustments_allowed':0,'covariate_input_sha256':hashlib.sha256(frame[['draft_year','nfl_id']+TASK].to_csv(index=False).encode()).hexdigest(),
             'algorithm':'Leave-the-prospect-out convex hull in incoming-speed × excursion space; nearest five reference IDs and distances for descriptive film comparison. No NFL outcome value enters this rule.',
             'limits':['Geometric support is not a confidence interval, physiological reliability, or equal-opportunity football comparison.','The reference is only the 2023/24 Combine measured OL sample.','Inside-hull comparison does not identify athlete talent or a pressure-risk effect.','No validated numerical threshold for scouting decisions is claimed.'],
             'film_rule':['Require usable trajectory, known turn timestamps and at least two valid reversals.','If incoming speed and excursion fall outside joint measured reference support, show the clip but abstain from relative efficiency ranking.','Within support, compare nearest reference clips at documented task ranges and formulate a coaching question; retain the distances.','Do not translate one recorded attempt into a reliability claim.','Make no automatic pressure forecast or roster adjustment, given the failed original frozen 2025 result and retrospective selection sensitivity.']}
    return summary

def publication_plot(d,output):
    fig,ax=plt.subplots(2,2,figsize=(12.5,9.4));blue='#367786';orange='#b86b38';purple='#745889'
    # A: exact roster coverage, including original reserved cohort.
    rows=d['archive']['cohort_coverage'];x=np.arange(3);w=.34
    ax[0,0].bar(x-w/2,[r['combine_OL'] for r in rows],w,color='#bec5ca',label='Combine OL')
    ax[0,0].bar(x+w/2,[r['eligible_min50'] for r in rows],w,color=blue,label='Usable mirror + ≥50 charted observations')
    for i,r in enumerate(rows):ax[0,0].text(i+w/2,r['eligible_min50']+.8,f"{r['eligible_min50']}/{r['combine_OL']}",ha='center',fontsize=9)
    ax[0,0].set(xticks=x,xticklabels=['2023','2024','2025 original test'],ylabel='Independent prospects',ylim=(0,57),title='A  Only 57 of 121 prospects support the original endpoint')
    ax[0,0].legend(fontsize=8,loc='upper left')
    # B: full fixed feature list, no selective imbalance reporting.
    labels={'combine_height':'Height','combine_weight':'Weight','forty':'40-yard time','ten_yd_split':'10-yard split','normalized_deficit':'Relative recovery cost','incoming_speed':'Incoming lateral speed','lateral_amplitude':'Lateral excursion'}
    s=d['r7']['selection_smd'];y=np.arange(len(s));vals=[r['standardized_mean_difference'] for r in s]
    ax[0,1].barh(y,vals,color=[blue if abs(v)>.4 else '#99adb5' for v in vals]);ax[0,1].axvline(0,color='black',linewidth=.8)
    ax[0,1].set(yticks=y,yticklabels=[labels[r['feature']] for r in s],xlabel='Retained − excluded mean / all-prospect SD',title='B  Selection changes the measured OL population')
    ax[0,1].invert_yaxis()
    # C: all sensitivity directions retained, no favorable split promotion.
    rng=np.random.default_rng(989)
    p=d['r8']['primary']['combine_minus_mirror'];ax[1,0].scatter([p*1e4],[0],marker='D',color='black',s=60,zorder=3)
    for yy,key,col in [(1,'splits',purple),(2,'delete',blue)]:
     v=np.array([r['combine_minus_mirror'] for r in d[key]])
     ax[1,0].scatter(v*1e4,np.full(len(v),yy)+rng.uniform(-.09,.09,len(v)),s=20,alpha=.8,color=col)
    b=np.array([r['baseline_minus_mirror'] for r in d['boot']]);lo,hi=np.quantile(b,[.025,.975]);median=np.median(b)
    ax[1,0].errorbar(median*1e4,3,xerr=1e4*np.array([[median-lo],[hi-median]]),fmt='o',color=orange,capsize=5)
    ax[1,0].axvline(0,color='black',linestyle='--',linewidth=.8)
    ax[1,0].set(yticks=[0,1,2,3],yticklabels=['Shared primary split','20 split repeats','38 prospect deletions','500 nested resamples'],xlabel='Brier improvement (×10⁻⁴); positive favors mirror',ylim=(-.4,3.5),title='C  Mirror evidence depends on resampling')
    ax[1,0].invert_yaxis()
    # D: retain both uncertainty scopes for the original frozen result.
    reserved=d['archive']['signed_comparisons'][1];delta=reserved['baseline_minus_mirror']
    for yy,key,col in [(0,'fixed_prediction95',orange),(1,'refit95',blue)]:
     lo,hi=reserved[key];ax[1,1].errorbar(delta*1e3,yy,xerr=1e3*np.array([[delta-lo],[hi-delta]]),fmt='o',color=col,capsize=6)
    ax[1,1].axvline(0,color='black',linestyle='--',linewidth=.8)
    ax[1,1].set(yticks=[0,1],yticklabels=['Fixed fitted states','Refit train + test players'],xlabel='Brier improvement (×10⁻³); positive favors mirror',ylim=(-.5,1.55),title='D  Original frozen 2025 gain was not confirmed')
    ax[1,1].invert_yaxis()
    fig.suptitle('Coverage and validation limits for mirror-based scouting',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(output/'coverage_selection_stability.png',dpi=200);plt.close(fig)

def run_appendix(raw,output,bootstrap_draws=500,split_repeats=20):
    output.mkdir(parents=True,exist_ok=True)
    all_players,source_audit=build_all_ol(raw,[2023,2024])
    assert len(all_players)==83
    frame=all_players[all_players.min50].sort_values(['draft_year','nfl_id']).reset_index(drop=True)
    assert len(frame)==38 and frame.groupby('draft_year').size().to_dict()=={2023:18,2024:20}
    assert int(frame.n.sum())==16165 and int(frame.k.sum())==1444
    funnel,thresholds,smd,observability,availability=coverage_audit(all_players)
    primary,records=nested_predictions(frame,common_folds=COMMON_FOLDS)
    expected=[0.08281911311702472,0.08280964600983351,0.08285783299123362,0.08287065760981599]
    actual=summary(primary)
    for name,value in zip(['constant','combine','combine_mirror','selected_procedure'],expected):assert abs(actual[name]-value)<1e-12
    result7={'status':'executed_retrospective_analysis','years':[2023,2024],'eligible_players':len(frame),'eligible_observations':int(frame.n.sum()),'eligible_pressures':int(frame.k.sum()),'source_audit':source_audit,'funnel':funnel,'opportunity_thresholds':thresholds,'selection_smd':smd,'availability_diagnostic':observability,'metrics':actual,'comparison':bootstrap_comparison(primary),'common_fold_sha256':COMMON_FOLD_SHA256,'no_new_holdout':True}
    print('R7 reproduced on official data: 83 Combine OL, 38 eligible players; exact shared-fold scores match.',flush=True)
    indexed=primary.set_index('nfl_id');deletions=[]
    for remove in frame.nfl_id:
        reduced=frame[frame.nfl_id.ne(remove)].reset_index(drop=True)
        pred,rec=nested_predictions(reduced,common_folds=trim_map(COMMON_FOLDS,int(remove)))
        aligned=indexed.loc[pred.nfl_id]
        row={'removed_nfl_id':int(remove),**summary(pred)}
        for name in ['combine','combine_mirror','selected_procedure']:row[name+'_max_prediction_shift']=float(np.max(np.abs(pred[name].to_numpy()-aligned[name].to_numpy())))
        deletions.append(row)
    splits=[]
    for i in range(split_repeats):
        seed=SEED+7000+i;pred,rec=nested_predictions(frame,common_folds=SPLIT_PLANS[i])
        splits.append({'seed':seed,**summary(pred)})
    print('All 38 deletion analyses and 20 nested split repeats reproduced.',flush=True)
    boot=retuned_bootstrap(frame,bootstrap_draws)
    result8={'status':'executed_retrospective_robustness','primary':actual,'deletion_count':len(deletions),'deletion_delta_range':[min(r['combine_minus_mirror'] for r in deletions),max(r['combine_minus_mirror'] for r in deletions)],'deletion_mirror_favorable_count':sum(r['combine_minus_mirror']>0 for r in deletions),'max_deleted_player_mirror_prediction_shift':max(r['combine_mirror_max_prediction_shift'] for r in deletions),'split_repeats':len(splits),'split_mirror_favorable_count':sum(r['combine_minus_mirror']>0 for r in splits),'split_delta_mean':float(np.mean([r['combine_minus_mirror'] for r in splits])),'retuned_bootstrap_draws':len(boot),'retuned_bootstrap_95':np.quantile([r['baseline_minus_mirror'] for r in boot],[.025,.975]).tolist(),'retuned_constant_minus_selected_95':np.quantile([r['constant_minus_selected'] for r in boot],[.025,.975]).tolist(),'selected_mirror_bootstrap_frequency':float(sum(r['mirror_selected_folds'] for r in boot)/(5*len(boot)))}
    film=film_support(all_players)
    assert film['measured_prospects']==83 and film['joint_task_supported']==72 and film['outside_joint_task_support']==11
    assert result8['deletion_mirror_favorable_count']==4 and result8['split_mirror_favorable_count']==18
    assert abs(result8['max_deleted_player_mirror_prediction_shift']-.03278123247008892)<1e-12
    if bootstrap_draws==500:
        assert np.allclose(result8['retuned_bootstrap_95'],[-.00019797616882475408,.0002709501818602032],atol=1e-12,rtol=0)
    result={'status':'all_self_contained_appendix_checks_passed','scope':'Only 2023/24 official records; all new work retrospective after original 2025 inspection.','original_2025_archived_not_recomputed':ARCHIVED_REFERENCE,'R7':result7,'R8':result8,'R9_film_support':film,'outputs_exclude_player_row_CSVs':True,'original_frozen_result_changed':False,'new_holdout_claimed':False,'new_predictive_gain_established':False,'common_fold_sha256':COMMON_FOLD_SHA256,'frozen_sensitivity_plan_sha256':FOLD_PLAN_SHA256,'sensitivity_fold_memberships_embedded_to_avoid_sklearn_version_changes':True,'limits':['Specified nested procedure does not capture prior human research choices.','Player resampling excludes shared-team/game dependence.','Geometry support is not physiological repeatability, a causal effect, or a calibrated talent threshold.','No automatic pressure-risk or roster adjustment.']}
    publication_plot({'archive':ARCHIVED_REFERENCE,'r7':result7,'r8':result8,'delete':deletions,'splits':splits,'boot':boot},output)
    # Build ordinary coverage/nested figure only; no per-player predictions are written.
    figures(output,all_players,thresholds,smd,primary,availability)
    result['figure_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.png')}
    (output/'TRANSPORT_APPENDIX_AGGREGATE.json').write_text(json.dumps(result,indent=2))
    with zipfile.ZipFile(output/'TRANSPORT_APPENDIX_SOURCE.zip','w',compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(Path(__file__).resolve(),arcname='Notebook_Appendix.py')
    print(json.dumps({'status':result['status'],'primary':actual,'retuned_bootstrap_95':result8['retuned_bootstrap_95'],'support_outside':film['outside_joint_task_support'],'aggregate':str(output/'TRANSPORT_APPENDIX_AGGREGATE.json')},indent=2),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw-dir',type=Path);parser.add_argument('--output-dir',type=Path,default=Path.cwd()/'transport_appendix_outputs');parser.add_argument('--bootstrap',type=int,default=500)
    args=parser.parse_args();raw=args.raw_dir or locate_data();run_appendix(raw,args.output_dir,args.bootstrap)
