intervals=[];jackknife_audit=[]
for label,columns in [('beta_binomial_constant',[]),('beta_binomial_full',BASE+FAMILY)]:
    predictions,audit=jackknife_plus(frame,FOLDS,columns,label)
    intervals.append(predictions);jackknife_audit.extend(audit)
intervals=pd.concat(intervals,ignore_index=True)
intervals.to_csv(WORK/'jackknife_plus_predictions.csv',index=False)
(WORK/'jackknife_fit_audit.json').write_text(json.dumps(jackknife_audit,indent=2,default=scalar))
coverage={}
for label,rows in intervals.groupby('model'):
    rate=rows.y/rows.n
    coverage[label]={'n':len(rows),'central90_count_coverage':float(((rate>=rows.conditional90_low)&(rate<=rows.conditional90_high)).mean()),
                     'jackknife_style_alpha05_empirical_coverage':float(((rate>=rows.jackknife_plus_low)&(rate<=rows.jackknife_plus_high)).mean()),
                     'median_central90_width':float((rows.conditional90_high-rows.conditional90_low).median()),
                     'median_jackknife_style_alpha05_width':float((rows.jackknife_plus_high-rows.jackknife_plus_low).median())}
(WORK/'interval_coverage.json').write_text(json.dumps(coverage,indent=2))
print(json.dumps(coverage,indent=2))
plot_count_aware(primary,intervals,WORK/'figure_count_aware_loss_coverage.png')
try:
    from IPython.display import display,Image
    display(Image(filename=str(WORK/'figure_count_aware_loss_coverage.png')))
except ImportError:
    print('Saved figure:',WORK/'figure_count_aware_loss_coverage.png')
