def fit_predict(xx,y,xt,alpha):
    med=np.nanmedian(xx,axis=0);med=np.where(np.isfinite(med),med,0)
    xx=np.where(np.isnan(xx),med,xx);xt=np.where(np.isnan(xt),med,xt)
    mean=xx.mean(0);std=xx.std(0);std=np.where(std>1e-12,std,1)
    xx=(xx-mean)/std;xt=(xt-mean)/std
    ym=y.mean();coef=np.linalg.solve(xx.T@xx+alpha*np.eye(xx.shape[1]),xx.T@(y-ym))
    return np.clip(ym+xt@coef,.001,.999)

def loss(y,p):return y*(1-p)**2+(1-y)*p*p

def logloss(y,p):return -y*np.log(p)-(1-y)*np.log1p(-p)

def cv_select(x,y,ids):
    scores=[];predictions={}
    for alpha in GRID:
        p=np.full(len(y),np.nan)
        for nid in np.unique(ids):
            test=ids==nid;train=~test
            p[test]=fit_predict(x[train],y[train],x[test],alpha)
        assert np.isfinite(p).all()
        scores.append({'alpha':alpha,'player_cv_brier':float(loss(y,p).mean()),'player_cv_log_loss':float(logloss(y,p).mean()),'player_cv_rate_mse':float(((y-p)**2).mean())})
        predictions[str(alpha)]=p.tolist()
    chosen=min(scores,key=lambda z:(z['player_cv_brier'],-z['alpha']))['alpha']
    return chosen,scores,predictions

def scores(y,p,n):
    return {'player_equal_brier':float(loss(y,p).mean()),'player_equal_log_loss':float(logloss(y,p).mean()),
            'player_rate_mae':float(abs(y-p).mean()),'opportunity_weighted_brier':float(np.average(loss(y,p),weights=n)),
            'mean_prediction':float(p.mean()),'observed_mean_rate':float(y.mean())}

def fit_state(train,columns,alpha):
    if len(train)<10:raise ValueError('At least10 eligible training players required.')
    xx=train[columns].to_numpy(float);y=train.rate.to_numpy(float)
    med=np.nanmedian(xx,axis=0);med=np.where(np.isfinite(med),med,0)
    xx=np.where(np.isnan(xx),med,xx);mean=xx.mean(0);std=xx.std(0);std=np.where(std>1e-12,std,1)
    xx=(xx-mean)/std;ym=y.mean()
    coef=np.linalg.solve(xx.T@xx+alpha*np.eye(xx.shape[1]),xx.T@(y-ym))
    return {'feature_columns':columns,'alpha':float(alpha),'median':med.tolist(),'mean':mean.tolist(),
            'std':std.tolist(),'coefficient':coef.tolist(),'intercept':float(ym),'prediction_clip':[.001,.999],
            'fit_player_ids':[int(x) for x in train.nfl_id],'fit_years':sorted(int(x) for x in train.draft_year.unique())}

def apply_state(state,test):
    xx=test[state['feature_columns']].to_numpy(float);med=np.asarray(state['median'])
    xx=np.where(np.isnan(xx),med,xx);xx=(xx-np.asarray(state['mean']))/np.asarray(state['std'])
    return np.clip(state['intercept']+xx@np.asarray(state['coefficient']),*state['prediction_clip'])

def fit_recipe(train,test,baseline_alpha,enhanced_alpha,base_columns=BASE):
    if len(train)<10 or len(test)<10:raise ValueError('At least 10 eligible players per fit/evaluation cohort are required.')
    y=train.rate.to_numpy();yy=test.rate.to_numpy();n=test.n.to_numpy()
    b=fit_predict(train[base_columns].to_numpy(float),y,test[base_columns].to_numpy(float),baseline_alpha)
    f=fit_predict(train[base_columns+FAMILY].to_numpy(float),y,test[base_columns+FAMILY].to_numpy(float),enhanced_alpha)
    return {'constant':scores(yy,np.full(len(test),y.mean()),n),'baseline':scores(yy,b,n),'enhanced':scores(yy,f,n),
            'baseline_predictions':b,'enhanced_predictions':f}

def development_models(cohort,historical_role=False):
    tr=cohort[cohort.draft_year.eq(2023)].reset_index(drop=True);te=cohort[cohort.draft_year.eq(2024)].reset_index(drop=True)
    if len(tr)<10 or len(te)<10:raise ValueError('At least 10 eligible players in each development cohort are required.')
    y=te.rate.to_numpy();yt=tr.rate.to_numpy();n=te.n.to_numpy();ids=tr.nfl_id.to_numpy()
    base=BASE+(['historical_is_tackle'] if historical_role else [])
    columns={'conventional':base,'normalized':base+FAMILY[:1],'speed_amplitude':base+FAMILY[1:],'full_family':base+FAMILY}
    result={'train_players':len(tr),'selection_players':len(te),'train_opportunities':int(tr.n.sum()),'selection_opportunities':int(te.n.sum()),
        'constant':scores(y,np.full(len(te),yt.mean()),n),'models':{},'comparisons':{}}
    pred={};chosen={};arrays={}
    for name,cols in columns.items():
        xx=tr[cols].to_numpy(float);xt=te[cols].to_numpy(float);arrays[name]=(xx,xt)
        alpha,cv,_=cv_select(xx,yt,ids);chosen[name]=alpha;pred[name]=fit_predict(xx,yt,xt,alpha)
        result['models'][name]={'features':cols,'alpha_selected_2023':alpha,'2023_cv_grid':cv,
            '2024_selection_grid':{str(a):scores(y,fit_predict(xx,yt,xt,a),n) for a in GRID},'selected_model_score':scores(y,pred[name],n)}
    # The role-conditioned history is reconstructed, not silently relabeled.
    if historical_role:return result,{'train':tr,'test':te,'predictions':pred,'alphas':chosen}
    rng=np.random.default_rng(SEED);vi=rng.integers(0,len(te),size=(FIXED_BOOT,len(te)))
    rng=np.random.default_rng(SEED+1);bi=rng.integers(0,len(tr),size=(REFIT_BOOT,len(tr)));bj=rng.integers(0,len(te),size=(REFIT_BOOT,len(te)))
    for name in ['normalized','speed_amplitude','full_family']:
        pb=pred['conventional'];pe=pred[name]
        diffs={'brier':loss(y,pb)-loss(y,pe),'log_loss':logloss(y,pb)-logloss(y,pe)}
        boot={k:[] for k in diffs};xb,xbt=arrays['conventional'];xf,xft=arrays[name]
        for ti,j in zip(bi,bj):
            bp=fit_predict(xb[ti],yt[ti],xbt[j],chosen['conventional']);fp=fit_predict(xf[ti],yt[ti],xft[j],chosen[name])
            boot['brier'].append(float((loss(y[j],bp)-loss(y[j],fp)).mean()))
            boot['log_loss'].append(float((logloss(y[j],bp)-logloss(y[j],fp)).mean()))
        influence=[]
        for drop in range(len(tr)):
            keep=np.arange(len(tr))!=drop
            ab,_,_=cv_select(xb[keep],yt[keep],ids[keep]);af,_,_=cv_select(xf[keep],yt[keep],ids[keep])
            bp=fit_predict(xb[keep],yt[keep],xbt,ab);fp=fit_predict(xf[keep],yt[keep],xft,af)
            influence.append({'removed_nfl_id':int(ids[drop]),'baseline_alpha':ab,'enhanced_alpha':af,
                'brier_improvement':float((loss(y,bp)-loss(y,fp)).mean()),'max_prediction_shift':float(abs(fp-pe).max())})
        result['comparisons'][name]={'baseline_score':scores(y,pb,n),'enhanced_score':scores(y,pe,n),
            'uncertainty':{k:{'baseline_minus_enhanced':float(d.mean()),'fixed_model_player_bootstrap95':np.quantile(d[vi].mean(1),[.025,.975]).tolist(),
                             'training_selection_refit_bootstrap95':np.quantile(boot[k],[.025,.975]).tolist()} for k,d in diffs.items()},
            'training_influence_reselecting_alpha':influence,'max_LOO_prediction_shift':max(z['max_prediction_shift'] for z in influence)}
    pp=te[['draft_year','nfl_id','display_name','n','k','rate']].copy()
    for name,val in pred.items():pp['prediction_'+name]=val
    result['baseline_missingness']={str(year):df[BASE].isna().sum().astype(int).to_dict() for year,df in [(2023,tr),(2024,te)]}
    return result,{'train':tr,'test':te,'predictions':pred,'alphas':chosen,'prediction_frame':pp}
