def tune_fit(train, test, record, columns):
    """No omitted-player outcome may influence preprocessing or chosen penalty."""
    ids = set(train.player_id)
    trials = []
    for penalty in [.1, 1, 10, 100]:
        pred = pd.Series(index=train.index, dtype=float)
        for inner in record['inner_folds']:
            itr = {str(z) for z in inner['train_ids']} & ids
            ite = {str(z) for z in inner['test_ids']} & ids
            a, b = train[train.player_id.isin(itr)], train[train.player_id.isin(ite)]
            if not len(a) or not len(b):
                raise ValueError('Empty inner split after jackknife restriction.')
            p, _, _ = fit_predict(a, b, columns, penalty, 'beta_binomial')
            pred.loc[b.index] = p
        if pred.isna().any():
            raise ValueError('Inner predictions incomplete.')
        trials.append((metrics(train.y, train.n, pred)['equal_player_brier'], penalty))
    best = min(t[0] for t in trials)
    penalty = max(t[1] for t in trials if t[0] <= best + 1e-12)
    p, model, pre = fit_predict(train, test, columns, penalty, 'beta_binomial')
    return p, model, pre, penalty

def finite_interval_quantiles(lower, upper, alpha=.05):
    """Paper-inspired finite-sample order statistics with tail level alpha=.05.

    Studentization is our adaptation. Only empirical coverage is claimed here;
    neither the paper's iid guarantee nor its nominal target is asserted for NFL.
    """
    m = lower.shape[0]
    lo = int(np.floor(alpha * (m + 1)))
    hi = int(np.ceil((1 - alpha) * (m + 1)))
    lower_bound = np.full(lower.shape[1], -np.inf) if lo < 1 else np.sort(lower, axis=0)[lo - 1]
    upper_bound = np.full(upper.shape[1], np.inf) if hi > m else np.sort(upper, axis=0)[hi - 1]
    return np.clip(lower_bound, 0, 1), np.clip(upper_bound, 0, 1)

def jackknife_plus(frame, folds, columns, label):
    outputs, audit = [], []
    for record in folds['outer_folds']:
        train = frame[frame.player_id.isin({str(z) for z in record['train_ids']})]
        test = frame[frame.player_id.isin({str(z) for z in record['test_ids']})]
        minus, plus, choices = [], [], []
        for ix, omitted in train.iterrows():
            fit = train.drop(index=ix)
            combined = pd.concat([train.loc[[ix]], test], ignore_index=True)
            if columns:
                p, model, _, penalty = tune_fit(fit, combined, record, columns)
            else:
                p, model, _ = fit_predict(fit, combined, [], 0, 'beta_binomial')
                penalty = 0
            # Count noise scale is conditional on recorded/design denominator,
            # never used as a predictor of mean; known opportunity counts affect interval width.
            variance = p * (1 - p) * (combined.n.to_numpy() + model.phi) / (combined.n.to_numpy() * (1 + model.phi))
            scale = np.sqrt(np.maximum(variance, 1e-12))
            score = abs(omitted.y / omitted.n - p[0]) / scale[0]
            minus.append(p[1:] - score * scale[1:]); plus.append(p[1:] + score * scale[1:])
            choices.append({'omitted_player_id': omitted.player_id, 'penalty': penalty,
                            'phi': model.phi, 'calibration_score': float(score),
                            'training_ids': fit.player_id.tolist()})
        lower, upper = finite_interval_quantiles(np.asarray(minus), np.asarray(plus))
        if columns:
            point, model, _, penalty = tune_fit(train, test, record, columns)
        else:
            point, model, _ = fit_predict(train, test, [], 0, 'beta_binomial'); penalty = 0
        rawlo = betabinom.ppf(.05, test.n, point * model.phi, (1 - point) * model.phi) / test.n
        rawhi = betabinom.ppf(.95, test.n, point * model.phi, (1 - point) * model.phi) / test.n
        for (_, row), p, lo, hi, rl, rh in zip(test.iterrows(), point, lower, upper, rawlo, rawhi):
            outputs.append({'player_id': row.player_id, 'cohort': int(row.cohort), 'fold': record['fold'],
                            'model': label, 'y': int(row.y), 'n': int(row.n), 'prediction': float(p),
                            'jackknife_plus_low': float(lo), 'jackknife_plus_high': float(hi),
                            'conditional90_low': float(rl), 'conditional90_high': float(rh)})
        audit.append({'fold': record['fold'], 'model': label, 'outer_penalty': penalty,
                      'jackknife_fits': choices})
        print(f'Completed {label} outer fold{record["fold"]}; {len(train)} nested jackknife omissions', flush=True)
    return pd.DataFrame(outputs), audit

def plot_count_aware(primary, intervals, output):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    wide = primary.pivot(index='player_id', columns='model', values='prediction')
    target = primary[primary.model.eq('constant_equal_player')].set_index('player_id').loc[wide.index]
    rate = (target.y/target.n).to_numpy()
    rng=np.random.default_rng(87007);samples=rng.integers(0,len(wide),size=(5000,len(wide)))
    models=['nested_conventional_ridge','beta_binomial_conventional','beta_binomial_full','nested_full_ridge']
    labels=['Conventional ridge','Conventional beta-binomial','Mirror beta-binomial','Mirror ridge']
    means=[];limits=[]
    for model in models:
        delta=(rate-wide.constant_equal_player.to_numpy())**2-(rate-wide[model].to_numpy())**2
        means.append(delta.mean());limits.append(np.quantile(delta[samples].mean(1),[.025,.975]))
    means=np.array(means);limits=np.array(limits)
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    positions=np.arange(4)
    ax[0].errorbar(means*10000,positions,xerr=np.vstack([(means-limits[:,0])*10000,(limits[:,1]-means)*10000]),
                  fmt='o',color='#176876',capsize=3)
    ax[0].axvline(0,color='#555',linewidth=1)
    ax[0].set_yticks(positions,labels)
    ax[0].set(xlabel='Constant minus model Brier ×10,000',title='A. Point gains remain unresolved')
    ax[0].text(.01,-.21,'95% fixed-OOF player bootstrap; no refitting',transform=ax[0].transAxes,fontsize=8)
    rows=[]
    for label,z in intervals.groupby('model'):
        y=z.y/z.n
        rows.append({'model':label,'raw':((y>=z.conditional90_low)&(y<=z.conditional90_high)).mean(),
                     'resampling':((y>=z.jackknife_plus_low)&(y<=z.jackknife_plus_high)).mean(),
                     'raw_width':(z.conditional90_high-z.conditional90_low).median()*100,
                     'resampling_width':(z.jackknife_plus_high-z.jackknife_plus_low).median()*100})
    rows=pd.DataFrame(rows).set_index('model').loc[['beta_binomial_constant','beta_binomial_full']]
    x=np.arange(2);w=.34
    bars1=ax[1].bar(x-w/2,rows.raw*100,w,color='#BD603B',label='Beta-binomial central90%')
    bars2=ax[1].bar(x+w/2,rows.resampling*100,w,color='#176876',label='Jackknife+ style, α=0.05')
    for bars in [bars1,bars2]:
        for b in bars:
            count=int(round(b.get_height()/100*38));ax[1].text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{count}/38',ha='center',fontsize=9)
    ax[1].set_xticks(x,['Constant prior','Mirror prior']);ax[1].set(ylim=(0,112),ylabel='Empirical player coverage (%)',title='B. Interval assumptions need a test')
    ax[1].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.31),frameon=False)
    for i,(_,r) in enumerate(rows.iterrows()):
        ax[2].plot([r.raw_width,r.resampling_width],[i,i],color='#999',linewidth=2)
        ax[2].scatter(r.raw_width,i,color='#BD603B',s=55);ax[2].scatter(r.resampling_width,i,color='#176876',s=55)
        ax[2].annotate(f'{r.raw_width:.2f}',(r.raw_width,i),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=9)
        ax[2].annotate(f'{r.resampling_width:.2f}',(r.resampling_width,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax[2].set_yticks(x,['Constant prior','Mirror prior']);ax[2].set(xlim=(4,14),ylim=(-.6,1.6),xlabel='Median interval width: pressures per100',title='C. Wider intervals cost precision')
    fig.suptitle('38 rookies,16,165 charted observations | 2023/24 exploratory development',fontsize=14)
    fig.text(.5,-.11,'Different interval constructions; no equal-nominal-level comparison or NFL coverage guarantee.\n'
             'All703 mirror-interval pairs overlap: abstain from Combine-derived ordering. Prior2025 failure remains unchanged.',
             ha='center',fontsize=9)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)