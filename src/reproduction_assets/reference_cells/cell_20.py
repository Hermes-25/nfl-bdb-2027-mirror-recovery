import os, csv, io, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
from scipy.stats import betabinom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=['combine_height','combine_weight','forty','ten_yd_split']
FAMILY=['normalized_deficit','incoming_speed','lateral_amplitude']
required=['players.csv','combine_results.csv','combine_tracking.csv','games.csv','player_play.csv']
explicit=os.environ.get('NFL_DATA_DIR')
if explicit: DATA_DIR=Path(explicit)
else:
    candidates=[p.parent for p in Path('/kaggle/input').glob('**/players.csv') if all((p.parent/f).is_file() for f in required)]
    if len(candidates)!=1: raise FileNotFoundError('Set NFL_DATA_DIR to the official CSV directory.')
    DATA_DIR=candidates[0]
if not all((DATA_DIR/f).is_file() for f in required): raise FileNotFoundError('Official inputs incomplete.')
WORK=Path.cwd()/'hierarchical_results';WORK.mkdir(exist_ok=True)
print('Development cohorts2023/2024 only; original2025 result retained as historical context.')
