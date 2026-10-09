"""A new display of the immutable original 2025 evaluation; no model refit."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=json.loads(a.result.read_text())
    assert r['evaluation_players']==19 and r['charted_opportunities']==7801
    fig,ax=plt.subplots(1,3,figsize=(14.2,4.6),layout='constrained')
    blue='#236c7a';orange='#b8633c';grey='#6c7680'
    names=['Constant','Conventional','+ mirror']
    scores=[r[k]['player_equal_brier'] for k in ['constant','baseline','enhanced']]
    ys=np.arange(3)
    ax[0].scatter(np.array(scores)*1e3,ys,s=70,color=[grey,blue,orange])
    for y,s in zip(ys,scores):ax[0].annotate(f'{s:.6f}',(s*1e3,y),xytext=(7,7),textcoords='offset points',fontsize=10)
    ax[0].axvline(scores[0]*1e3,color=grey,linestyle=':',alpha=.65)
    ax[0].set(yticks=ys,yticklabels=names,xlabel='Equal-player Brier loss ×1,000\nLower is better',title='A  Constant forecast wins',xlim=(83.35,85.2),ylim=(-.6,2.6))
    ax[0].invert_yaxis()
    delta=r['brier_improvement']
    for y,key,col in [(0,'fixed_state_player_bootstrap95',orange),(1,'pooled_training_and_evaluation_refit_bootstrap95',blue)]:
        lo,hi=r[key]
        ax[1].errorbar(delta*1e3,y,xerr=np.array([[delta-lo],[hi-delta]])*1e3,fmt='o',color=col,capsize=5)
    ax[1].axvline(0,color='black',linestyle='--',linewidth=1)
    ax[1].set(yticks=[0,1],yticklabels=['Fixed fitted states','Train + test player refits'],xlabel='Conventional − mirror Brier ×1,000\nPositive favors mirror',title='B  Keep both uncertainty scopes',ylim=(-.6,1.6))
    ax[1].invert_yaxis()
    means=[r[k]['mean_prediction']*100 for k in ['constant','baseline','enhanced']]
    obs=r['baseline']['observed_mean_rate']*100
    ax[2].scatter(means,ys,s=70,color=[grey,blue,orange])
    for y,m in zip(ys,means):ax[2].annotate(f'{m:.2f}',(m,y),xytext=(7,7),textcoords='offset points',fontsize=10)
    ax[2].axvline(obs,color='black',linestyle='--',linewidth=1,label=f'Observed equal-player rate {obs:.2f}')
    ax[2].legend(fontsize=8,loc='lower right',frameon=False)
    ax[2].set(yticks=ys,yticklabels=names,xlabel='Average pressures per100 charted observations',title='C  Mirror forecast is too high on average',ylim=(-.6,2.6),xlim=(8.7,12.25))
    ax[2].invert_yaxis()
    for x in ax:x.spines[['top','right']].set_visible(False);x.grid(axis='x',alpha=.15)
    fig.suptitle('Original frozen 2025 evaluation | 19 rookies, 7,801 charted observations',fontsize=15)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
    plt.close(fig)
    a.output.with_suffix('.provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(a.result.read_bytes()).hexdigest(),'source':a.result.name,'new_fit':False,'new_evaluation':False,'display_only_revision':True},indent=2))

if __name__=='__main__':main()
