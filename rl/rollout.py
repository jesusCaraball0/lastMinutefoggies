"""Complete-game on-policy trajectories, always from one player's perspective."""
from dataclasses import dataclass
import random

import torch

from rl.game import play_game
from rl.policy import PolicyAgent


def outcome(result, side):
    if result == '1/2-1/2':
        return 'draws'
    if result not in ('1-0', '0-1'):
        raise ValueError(f'Unfinished game: {result}')
    return 'wins' if (result == '1-0') == side else 'losses'


def terminal_reward(result, side):
    return {'wins': 1., 'losses': -1., 'draws': .1}[outcome(result, side)]


def discounted_returns(decisions, reward, gamma):
    """Monte Carlo terminal returns, independent of the critic/GAE targets."""
    return reward * gamma ** torch.arange(decisions-1, -1, -1, dtype=torch.float32)


def advantages(values, reward, gamma=.99, gae_lambda=.95):
    """GAE across this player's decisions; opponent plies are not transitions.

    The last state is terminal, so bootstrap is exactly zero. Returns are not
    mixed between colors/games. There are no partial-episode rollout cutoffs.
    """
    values = torch.as_tensor(values, dtype=torch.float32)
    result = torch.zeros_like(values)
    following = 0.
    for i in reversed(range(len(values))):
        next_value = values[i+1] if i+1 < len(values) else 0.
        immediate = reward if i == len(values)-1 else 0.
        delta = immediate + gamma*next_value - values[i]
        following = delta + gamma*gae_lambda*following
        result[i] = following
    return result, result + values


@dataclass
class Batch:
    tokens: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    discounted_returns: torch.Tensor

    def __len__(self):
        return len(self.actions)


class Trajectories:
    """Build one policy's batch without mixing trajectories or player perspectives."""
    def __init__(self, gamma, gae_lambda):
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.transitions, self.advs, self.returns, self.discounted, self.reports = [], [], [], [], []

    def add(self, agent, replay, side, opponent):
        if replay.get('failedSide') or replay['result'] == '*':
            raise RuntimeError(f"Game failed: {replay['reason']} ({replay.get('failedSide')})")
        if not agent.transitions:
            raise RuntimeError('Game yielded no learner decisions')
        reward = terminal_reward(replay['result'], side)
        a, r = advantages([t.value for t in agent.transitions], reward, self.gamma, self.gae_lambda)
        self.transitions.extend(agent.transitions); self.advs.append(a); self.returns.append(r)
        self.discounted.append(discounted_returns(len(a), reward, self.gamma))
        self.reports.append(dict(opponent=opponent, color='white' if side else 'black',
                                 reward=reward, outcome=outcome(replay['result'], side),
                                 result=replay['result'], reason=replay['reason'], plies=len(replay['frames'])-1))

    def finish(self):
        ts = self.transitions
        batch = Batch(torch.stack([t.tokens for t in ts]), torch.stack([t.mask for t in ts]),
                      torch.tensor([t.action for t in ts]), torch.tensor([t.log_probability for t in ts]),
                      torch.tensor([t.value for t in ts]), torch.cat(self.advs), torch.cat(self.returns),
                      torch.cat(self.discounted))
        return batch, self.reports


def collect(model, opponents, games, temperature=.9, gamma=.99, gae_lambda=.95,
            max_plies=256, base=180., increment=2., replay_callback=None):
    if not opponents or games < 2 or games % 2:
        raise ValueError('Use an opponent pool and an even game count >= 2')
    trajectories = Trajectories(gamma, gae_lambda)
    for pair in range(games//2):
        name, factory = random.choice(opponents)
        for side in (True, False):
            learner = PolicyAgent(model, temperature, record=True)
            ours = lambda: learner
            replay = play_game(ours if side else factory, factory if side else ours,
                               base=base, increment=increment, max_plies=max_plies,
                               white_name='learner' if side else name,
                               black_name=name if side else 'learner')
            trajectories.add(learner, replay, side, name)
            if replay_callback: replay_callback(replay)
    return trajectories.finish()


def collect_pair(models, names, games, temperatures=(.9,.9), gamma=.99, gae_lambda=.95,
                 max_plies=256, base=180., increment=2., replay_callback=None):
    """Play once, learn from both sides; callers update neither model until return."""
    if len(models)!=2 or len(names)!=2 or len(temperatures)!=2 or games<2 or games%2:
        raise ValueError('Two models and an even game count >=2 are required')
    trajectories = [Trajectories(gamma, gae_lambda) for _ in models]
    for game in range(games):
        agents = [PolicyAgent(model, temperature, record=True) for model,temperature in zip(models,temperatures)]
        white = game%2
        replay = play_game(lambda:agents[white], lambda:agents[1-white],
                           base=base, increment=increment, max_plies=max_plies,
                           white_name=names[white], black_name=names[1-white])
        for i,agent in enumerate(agents):
            trajectories[i].add(agent, replay, i==white, names[1-i])
        if replay_callback: replay_callback(replay)
    return [t.finish() for t in trajectories]
