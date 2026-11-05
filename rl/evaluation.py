"""Separate, color-balanced evaluation against frozen game-contract agents."""
from collections import Counter
import random

import torch

from rl.game import play_game
from rl.policy import PolicyAgent
from rl.rollout import terminal_reward, outcome, discounted_returns
from rl.ppo import value_quality


def evaluate(model, opponents, games_per_opponent=20, temperature=.9, max_plies=256,
             base=180., increment=2., seed=9137, replay_callback=None, gamma=.99):
    if not opponents or games_per_opponent < 2 or games_per_opponent % 2:
        raise ValueError('Evaluation needs opponents and an even game count >= 2')
    python_state=random.getstate()
    device=next(model.parameters()).device
    cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    reports={}
    try:
        # Fixed evaluation seeds and forked RNG make scores more comparable
        # without changing subsequent training trajectories.
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            random.seed(seed)
            for name,factory in opponents:
                counts=Counter(wins=0,losses=0,draws=0,failures=0)
                colors={color:Counter(wins=0,losses=0,draws=0) for color in ['white','black']}
                reasons=Counter()
                rewards, predictions, targets, plies = [], [], [], []
                for game in range(games_per_opponent):
                    side=game%2==0
                    agent=PolicyAgent(model,temperature,record=True)
                    learner=lambda:agent
                    replay=play_game(learner if side else factory,factory if side else learner,
                                     max_plies=max_plies,base=base,increment=increment,
                                     white_name='learner' if side else name,
                                     black_name=name if side else 'learner')
                    if replay.get('failedSide') or replay['result']=='*':
                        raise RuntimeError(f"Evaluation bot failure: {replay['reason']}")
                    reward=terminal_reward(replay['result'],side)
                    result=outcome(replay['result'],side)
                    rewards.append(reward); plies.append(len(replay['frames'])-1)
                    predictions.extend(t.value for t in agent.transitions)
                    targets.extend(discounted_returns(len(agent.transitions),reward,gamma).tolist())
                    counts[result]+=1;colors['white' if side else 'black'][result]+=1
                    reasons[replay['reason']]+=1
                    if replay_callback:
                        replay_callback(name,game,replay)
                reports[name]=dict(counts,score=(counts['wins']+.5*counts['draws'])/games_per_opponent,
                                   games=games_per_opponent,by_color={k:dict(v) for k,v in colors.items()},
                                   reasons=dict(reasons),mean_reward=sum(rewards)/len(rewards),
                                   mean_plies=sum(plies)/len(plies),
                                   value_monte_carlo=value_quality(predictions,targets))
    finally:
        random.setstate(python_state)
    return reports
