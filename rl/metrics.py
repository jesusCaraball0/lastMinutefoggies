"""Outcome summaries and persisted PPO learning curves."""
from collections import Counter
from pathlib import Path
import math


def game_summary(games):
    counts=Counter(g['outcome'] for g in games)
    rewards=[g['reward'] for g in games]
    mean=sum(rewards)/len(rewards)
    std=math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return dict(wins=counts['wins'],losses=counts['losses'],draws=counts['draws'],
                score=(counts['wins']+.5*counts['draws'])/len(games),
                mean_reward=mean,reward_std=std,
                reward_standard_error=std/math.sqrt(len(games)-1) if len(games)>1 else None,
                win_rate=counts['wins']/len(games),draw_rate=counts['draws']/len(games),
                loss_rate=counts['losses']/len(games),
                mean_plies=sum(g['plies'] for g in games)/len(games),
                reasons=dict(Counter(g['reason'] for g in games)),game_count=len(games))


def plot_metrics(log,run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=log['updates']
    if not rows:return
    x=[r['update'] for r in rows]
    evaluations=log['evaluations']
    ex=[e['update'] for e in evaluations]

    def draw(filename, panels, columns=2):
        count=len(panels)
        fig,axes=plt.subplots(math.ceil(count/columns),columns,figsize=(6*columns,3.4*math.ceil(count/columns)),squeeze=False)
        for ax,(title,series) in zip(axes.flat,panels):
            for label,xx,yy in series:
                # Missing historical metrics remain gaps, never fabricated zeros.
                ax.plot(xx,[float('nan') if y is None else y for y in yy],label=label)
            for phase in log.get('value_pretraining',[]):
                ax.axvline(phase['after_ppo_update'],color='gray',linestyle='--',alpha=.6,label='Value warmup')
            ax.set(title=title,xlabel='PPO update');ax.grid(alpha=.2)
            if series: ax.legend(fontsize=8)
        for ax in list(axes.flat)[count:]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(run_dir/filename,dpi=150);plt.close(fig)

    def curve(label,section,key):
        return label,x,[r[section].get(key) for r in rows]

    # Historical rewards come from stored game records, preserving their original reward scheme.
    train_rewards=[sum(g['reward'] for g in r['rollout']['games'])/len(r['rollout']['games']) for r in rows]
    eval_rewards=[e.get('mean_reward') for e in evaluations]
    reward_series=[('Training opponents',x,train_rewards),('Frozen evaluation',ex,eval_rewards)]
    score_series=[curve('Training opponents','rollout','score'),('Frozen evaluation',ex,[e['mean_score'] for e in evaluations])]
    draw('rewards.png', [('Mean terminal reward per game',reward_series),('Game score (win=1, draw=0.5)',score_series)])
    draw('losses.png', [(title,[curve(title,'ppo',key)]) for title,key in
                        [('Actor loss','policy_loss'),('Critic loss (half GAE MSE)','value_loss'),('Total PPO loss','total_loss')]])
    value_panels=[]
    for metric,title in [('mae','Value MAE'),('rmse','Value RMSE'),('explained_variance','Value explained variance')]:
        series=[]
        for phase,label in [('before','Fresh rollout, before update'),('after','Same rollout, after update')]:
            series.append((label,x,[r['ppo'].get('value_fit',{}).get('monte_carlo_'+phase,{}).get(metric) for r in rows]))
        # Equal-opponent mean; individual position-weighted measurements remain in JSON.
        series.append(('Frozen evaluation',ex,[
            (sum(v['value_monte_carlo'][metric] for v in e['opponents'].values())/len(e['opponents'])
             if all(v.get('value_monte_carlo',{}).get(metric) is not None for v in e['opponents'].values()) else None)
            for e in evaluations]))
        value_panels.append((title+' vs discounted terminal return',series))
    draw('value_quality.png',value_panels)
    draw('outcomes.png',[('Training outcome rates',[curve(k,'rollout',k+'_rate') for k in ['win','draw','loss']]),
                          ('Training game length',[curve('Mean plies','rollout','mean_plies')])])
    health=[(title,[curve(title,'ppo',key)]) for title,key in [
        ('Endpoint approximate KL','endpoint_kl'),('Masked entropy (nats)','endpoint_entropy'),
        ('PPO ratio clipping fraction','ratio_clip_fraction'),('Gradient clipping fraction','gradient_clipped_fraction'),
        ('Mean pre-clip gradient L2','pre_clip_gradient_l2'),('Maximum pre-clip gradient L2','gradient_global_max')]]
    draw('policy_health.png',health)
    draw('throughput.png',[(title,[curve(title,'rollout',key)]) for title,key in
                           [('Games / rollout second','games_per_second'),('Decisions / rollout second','decisions_per_second'),
                            ('Cumulative training games','cumulative_games'),('Cumulative training decisions','cumulative_decisions')]])
    opponent_names=sorted({name for e in evaluations for name in e['opponents']})
    draw('evaluation_opponents.png',[
        (title,[(Path(name).parent.name if Path(name).suffix=='.pt' else name,ex,[e['opponents'].get(name,{}).get(key) for e in evaluations]) for name in opponent_names])
        for title,key in [('Frozen opponent score','score'),('Frozen opponent mean reward','mean_reward')]])
    draw('training.png',[('Mean reward per game',reward_series),('Game score',score_series),
                         ('Critic loss',[curve('Critic loss','ppo','value_loss')])]+health[:3])
