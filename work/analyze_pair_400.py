import json, statistics, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def mean(rows, section, key):
    xs=[r[section].get(key) for r in rows]
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else None

def summary(folder):
    d=json.loads((folder/'metrics.json').read_text()); rows=d['updates']; ev=d['evaluations']
    def window(rs):
        n=sum(len(r['rollout']['games']) for r in rs)
        w=sum(r['rollout']['wins'] for r in rs); l=sum(r['rollout']['losses'] for r in rs); draws=n-w-l
        ppo={k:mean(rs,'ppo',k) for k in ['policy_loss','value_loss','total_loss','endpoint_kl','endpoint_entropy','ratio_clip_fraction','gradient_clipped_fraction','pre_clip_gradient_l2','gradient_global_max']}
        vf={}
        for phase in ['monte_carlo_before','monte_carlo_after']:
            # Pool squared/absolute errors by decisions; EV is explicitly an update-average.
            weights=[r['ppo']['value_fit'][phase]['samples'] for r in rs]
            vals=[r['ppo']['value_fit'][phase] for r in rs]
            total=sum(weights)
            mse=sum(v['mse']*n for v,n in zip(vals,weights))/total
            vs=[v['explained_variance'] for v in vals if v['explained_variance'] is not None]
            vf[phase]=dict(mse=mse,rmse=math.sqrt(mse),mae=sum(v['mae']*n for v,n in zip(vals,weights))/total,
                            mean_update_explained_variance=statistics.mean(vs) if vs else None)
        return dict(updates=len(rs),games=n,wins=w,losses=l,draws=draws,score=(w+.5*draws)/n,
                    mean_reward=(w-l+.1*draws)/n,mean_plies=mean(rs,'rollout','mean_plies'),ppo=ppo,value_fit=vf)
    best=max(ev,key=lambda e:e['mean_score']) if ev else None
    reasons={}
    for r in rows:
        for k,v in r['rollout']['reasons'].items():reasons[k]=reasons.get(k,0)+v
    weighted_steps=sum(r['ppo']['optimizer_steps'] for r in rows)
    return dict(variant=folder.name,status=d['status'],completed=rows[-1]['update'],parameter_count=d['parameter_count'],
                settings=d['settings'],initial_eval=ev[0],best_eval=best,final_eval=ev[-1],
                all=window(rows),first50=window(rows[:50]),last50=window(rows[-50:]),
                total_decisions=sum(r['ppo']['transitions'] for r in rows),optimizer_steps=weighted_steps,
                early_stop_updates=sum(r['ppo']['early_stop_kl'] for r in rows),
                weighted_gradient_clip_fraction=sum(r['ppo']['gradient_clipped_fraction']*r['ppo']['optimizer_steps'] for r in rows)/weighted_steps,
                largest_gradient=max(r['ppo']['gradient_global_max'] for r in rows),
                largest_kl=max(r['ppo']['endpoint_kl'] for r in rows),termination_reasons=reasons,
                rollout_seconds=sum(r['rollout_seconds'] for r in rows),
                optimization_seconds=sum(r['optimization_seconds'] for r in rows),wall_seconds=d.get('wall_seconds'),
                final_update=rows[-1],diagnostic_count=len(d['activation_diagnostics']))

if __name__=='__main__':
    out=ROOT/'out/rl/paired-400-analysis.json'
    result={p.name:summary(p) for p in sorted((ROOT/'out/rl').iterdir()) if (p/'metrics.json').exists()}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,d in result.items():
        print(name,d['status'],d['completed'],'initial/best/final',d['initial_eval']['mean_score'],d['best_eval']['mean_score'],d['final_eval']['mean_score'])
        print('first50',d['first50']['score'],'last50',d['last50']['score'],'all',d['all']['score'])
        print('value first/last',d['first50']['value_fit']['monte_carlo_before'],d['last50']['value_fit']['monte_carlo_before'])
        print('health last50',d['last50']['ppo'])
