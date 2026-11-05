"""Contract parity, PPO mathematics, warm starts, and exact continuation checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rl.storage import variant_name, training_lock, require_completed_pair

import torch

import referee
from rl import train, train_pair, pretrain_value
from rl.evaluation import evaluate
from rl.export import main as export
from rl.game import Command, play_game
from rl.policy import PolicyAgent, load_policy, masked_distribution, initialize_value
from rl.ppo import PPOConfig, evaluate_actions, optimizer_for, update, value_quality
from rl.rollout import advantages, collect, terminal_reward, discounted_returns, collect_pair
from rl.metrics import game_summary
from starters.shared.model import create_policy


class Scripted:
    def reset(self,side,base,increment):
        self.moves=iter(['g1f3','f3g1']*8 if side else ['g8f6','f6g8']*8)
        self.views=[]
    def observe(self,view,ply):self.views.append((view,ply))
    def act(self,remaining):return next(self.moves)
    def close(self):pass


class RLTests(unittest.TestCase):
    def setUp(self):
        previous=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,previous)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def checkpoint(self,**config):
        config=dict(channels=2,hidden_size=8,batch_norm=True,dropout=.1,**config)
        model=create_policy('feedforward',**config).eval()
        path=self.root/'initial.pt'
        torch.save(dict(kind='feedforward',activation='relu',model_config=config,
                        state_dict=model.state_dict()),path)
        return model,path

    def wire_bot(self):
        p=self.root/'scripted.py'
        p.write_text("import sys\nassert input()=='fow'\nside=input().split()[0]\nmoves=iter(['g1f3','f3g1']*8 if side=='w' else ['g8f6','f6g8']*8)\nprint('ready',flush=True)\nfor line in sys.stdin:\n if len(line.strip())!=64:print('move '+next(moves),flush=True)\n")
        return (sys.executable,'-u',str(p))

    def test_same_referee_game_for_python_and_existing_wire_contract(self):
        command=self.wire_bot()
        direct=referee.run_match(command,command,max_plies=20)
        adapted=play_game(Scripted,Scripted,max_plies=20)
        mixed=play_game(Scripted,Command(command),max_plies=20)
        for r in [adapted,mixed]:
            self.assertEqual((r['result'],r['reason']),('1/2-1/2','Threefold repetition'))
            fields=['ply','board','white','black','move']
            self.assertEqual([{k:f[k] for k in fields} for f in r['frames']],
                             [{k:f[k] for k in fields} for f in direct['frames']])

    def test_illegal_agent_response_is_a_referee_forfeit(self):
        class Invalid(Scripted):
            def act(self,remaining):return 'e2e5'
        r=play_game(Invalid,Scripted)
        self.assertEqual(r['result'],'0-1')
        self.assertEqual(r['failedSide'],'white')
        self.assertTrue(r['reason'].startswith('Illegal move'))

    def test_rollout_accepts_existing_executable_bot_without_an_rl_interface(self):
        model,_=self.checkpoint()
        batch,games=collect(model,[('wire-bot',Command(self.wire_bot()))],2,max_plies=4)
        self.assertEqual(len(games),2)
        self.assertGreater(len(batch),0)
        self.assertEqual({g['color'] for g in games},{'white','black'})

    def test_reward_sign_and_terminal_gae(self):
        self.assertEqual(terminal_reward('1-0',True),1)
        self.assertEqual(terminal_reward('1-0',False),-1)
        self.assertEqual(terminal_reward('1/2-1/2',False),.1)
        adv,returns=advantages([.2,.3],1,gamma=.9,gae_lambda=.5)
        self.assertTrue(torch.allclose(adv,torch.tensor([.385,.7])))
        self.assertTrue(torch.allclose(returns,torch.tensor([.585,1.])))
        zero,_=advantages([0.,0.],0)
        self.assertEqual(zero.tolist(),[0.,0.])

    def test_draw_bonus_does_not_turn_draws_into_wins(self):
        model,_=self.checkpoint()
        batch,games=collect(model,[('same',lambda:PolicyAgent(model))],2,max_plies=4)
        self.assertEqual([g['reward'] for g in games],[.1,.1])
        summary=game_summary(games)
        self.assertEqual((summary['wins'],summary['losses'],summary['draws']),(0,0,2))
        self.assertEqual(summary['score'],.5)
        self.assertEqual(summary['mean_reward'],.1)
        report=evaluate(model,[('same',lambda:PolicyAgent(model))],2,max_plies=4)['same']
        self.assertEqual((report['wins'],report['losses'],report['draws']),(0,0,2))
        self.assertEqual(report['score'],.5)
        self.assertEqual(report['mean_reward'],.1)
        self.assertEqual(report['value_monte_carlo']['samples'],4)
        self.assertTrue(torch.allclose(batch.discounted_returns,torch.tensor([.099,.1,.099,.1])))

    def test_value_regression_metrics_and_monte_carlo_targets(self):
        self.assertTrue(torch.allclose(discounted_returns(3,1.,.5),torch.tensor([.25,.5,1.])))
        metrics=value_quality([0.,1.],[1.,-1.])
        self.assertAlmostEqual(metrics['mse'],2.5)
        self.assertAlmostEqual(metrics['mae'],1.5)
        self.assertAlmostEqual(metrics['rmse'],2.5**.5)
        self.assertAlmostEqual(metrics['explained_variance'],-1.25)
        self.assertIsNone(value_quality([0.,1.],[1.,1.])['explained_variance'])

    def test_masks_and_zero_probability_for_unavailable_actions(self):
        mask=torch.tensor([[True,False,True]])
        distribution=masked_distribution(torch.tensor([[1.,100.,2.]]),mask,.9)
        self.assertEqual(float(distribution.probs[0,1]),0)
        self.assertTrue(torch.isfinite(distribution.entropy()).all())
        with self.assertRaises(ValueError):masked_distribution(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool),1)

    def test_ppo_preserves_modes_and_reuses_exact_rollout_probabilities(self):
        model,path=self.checkpoint()
        opponents=[('frozen',lambda:PolicyAgent(model))]
        batch,games=collect(model,opponents,2,max_plies=8)
        self.assertEqual(len(games),2)
        self.assertEqual({g['color'] for g in games},{'white','black'})
        logp,_,_=evaluate_actions(model,batch.tokens,batch.masks,batch.actions,.9)
        self.assertTrue(torch.allclose(logp.detach(),batch.old_log_probabilities,atol=2e-5))
        self.assertTrue(batch.masks.gather(1,batch.actions[:,None]).all())
        buffers={k:v.clone() for k,v in model.named_buffers()}
        weights=model.policy.weight.detach().clone()
        config=PPOConfig(epochs=2,minibatch_size=4,target_kl=.1)
        stats=update(model,optimizer_for(model,config),batch,config)
        self.assertGreater(stats['optimizer_steps'],0)
        self.assertFalse(torch.equal(weights,model.policy.weight))
        self.assertFalse(model.training)
        for k,v in model.named_buffers():self.assertTrue(torch.equal(v,buffers[k]),k)
        self.assertGreaterEqual(stats['endpoint_kl'],0)

    def test_loading_opponent_does_not_change_training_rng(self):
        _,path=self.checkpoint();state=torch.get_rng_state().clone()
        load_policy(path)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_evaluation_isolates_rng_and_uses_frozen_model(self):
        model,_=self.checkpoint();random.seed(8);torch.manual_seed(8)
        python_state=random.getstate();state=torch.get_rng_state().clone()
        weights={k:v.clone() for k,v in model.state_dict().items()}
        result=evaluate(model,[('same',lambda:PolicyAgent(model))],2,max_plies=6)
        self.assertEqual(result['same']['games'],2)
        self.assertEqual(random.getstate(),python_state)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]),k)

    def test_recurrent_opponent_supported_but_recurrent_actor_rejected(self):
        model=create_policy('recurrent',channels=2,hidden_size=8)
        with self.assertRaises(ValueError):PolicyAgent(model,record=True)
        r=play_game(lambda:PolicyAgent(model),Scripted,max_plies=4)
        self.assertNotIn('failedSide',r)

    def test_end_to_end_resume_extension_matches_uninterrupted_training(self):
        original,path=self.checkpoint()
        original_bytes=path.read_bytes()
        base=['--init-checkpoint',str(path),'--opponent',str(path),'--eval-opponent',str(path),
              '--games-per-update','2','--max-plies','8','--eval-every','1','--eval-games','2',
              '--epochs','2','--minibatch-size','4','--device','cpu','--threads','1',
              '--self-play','--snapshot-every','1','--checkpoint-every','1','--temperature','0.7']
        continuous=self.root/'continuous';extended=self.root/'extended'
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(base+['--updates','2','--run-dir',str(continuous)])
            train.main(base+['--updates','1','--run-dir',str(extended)])
            train.main(['--resume',str(extended/'policy.pt'),'--updates','2'])
        a=torch.load(continuous/'policy.pt',weights_only=True)
        b=torch.load(extended/'policy.pt',weights_only=True)
        for k,v in a['state_dict'].items():self.assertTrue(torch.equal(v,b['state_dict'][k]),k)
        self.assertTrue(torch.equal(a['torch_rng_state'],b['torch_rng_state']))
        self.assertEqual(a['python_rng_state'],b['python_rng_state'])
        self.assertEqual(a['training_log']['updates'][-1]['ppo']['endpoint_kl'],
                         b['training_log']['updates'][-1]['ppo']['endpoint_kl'])
        log=b['training_log']
        self.assertEqual(log['reward_scheme']['draw'],.1)
        self.assertEqual(log['updates'][-1]['rollout']['cumulative_games'],4)
        self.assertIn('monte_carlo_before',log['updates'][-1]['ppo']['value_fit'])
        for name in ['rewards','losses','value_quality','outcomes','policy_health','throughput','evaluation_opponents','training']:
            self.assertTrue((extended/(name+'.png')).is_file(),name)
        # A historical draw-zero critic objective cannot silently resume as draw +0.1.
        legacy=self.root/'legacy.pt'
        legacy_saved=dict(b);legacy_saved['training_log']=dict(log)
        legacy_saved['training_log'].pop('reward_scheme')
        torch.save(legacy_saved,legacy)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            train.parse_args(['--resume',str(legacy),'--updates','3'])
        self.assertEqual(path.read_bytes(),original_bytes)
        self.assertEqual(list(extended.glob('*.pt')),[extended/'policy.pt'])
        self.assertEqual(b['best_policy']['update'],0)  # Equal all-draw scores retain the original best.
        with contextlib.redirect_stdout(io.StringIO()):
            train.main(['--resume',str(extended/'policy.pt'),'--updates','3','--gae-lambda','1'])
        changed=torch.load(extended/'policy.pt',weights_only=True)
        self.assertEqual(changed['settings']['gae_lambda'],1.)
        self.assertEqual(changed['training_log']['recipe_changes'],[
            dict(after_update=2,parameter='gae_lambda',previous=.95,value=1.)])
        self.assertEqual(changed['training_log']['updates'][-1]['ppo']['gae_lambda'],1.)
        self.assertEqual(changed['training_log']['updates'][:2],b['training_log']['updates'])
        adv,targets=advantages([.2,-.1,.4],1.,gamma=.99,gae_lambda=1.)
        self.assertTrue(torch.allclose(targets,torch.tensor([.99**2,.99,1.])))
        self.assertEqual(len(b['opponent_history']),2)
        self.assertTrue(torch.equal(b['opponent_history'][0]['state_dict']['policy.weight'],
                                    a['opponent_history'][0]['state_dict']['policy.weight']))
        self.assertFalse(torch.equal(b['best_policy']['state_dict']['value.weight'],b['state_dict']['value.weight']))
        self.assertEqual(b['update'],2)
        self.assertEqual(b['inference_temperature'],.7)
        # PPO weights remain ordinary game-contract checkpoint weights.
        loaded,_=load_policy(extended/'policy.pt')
        with torch.no_grad():
            self.assertTrue(torch.isfinite(loaded(torch.zeros(1,14,8,8))[0]).all())
        bundle=export(['--checkpoint',str(extended/'policy.pt'),'--output',str(self.root/'ppo-bot')])
        env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
        result=subprocess.run(['bash',str(bundle/'build.sh')],env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        exported=torch.load(bundle/'policy.pt',weights_only=True)
        self.assertEqual(exported['update'],b['best_policy']['update'])
        self.assertTrue(torch.equal(exported['state_dict']['policy.weight'],b['best_policy']['state_dict']['policy.weight']))
        self.assertNotIn('optimizer_state_dict',exported)
        # Shared arena entrypoint honors PPO sampling metadata.
        code="from bot import load_policy; assert load_policy().temperature==0.7"
        subprocess.run([sys.executable,'-c',code],cwd=bundle,env=env,check=True,timeout=30)

    def test_canonical_variant_continues_automatically_and_writer_lock(self):
        _,path=self.checkpoint()
        base=['--init-checkpoint',str(path),'--opponent',str(path),'--eval-every','0',
              '--games-per-update','2','--max-plies','4','--epochs','1','--device','cpu','--threads','1']
        with patch('rl.train.RL_ROOT',self.root/'rl'),patch('rl.train.capture',return_value={}),patch('rl.train.plot_metrics'),contextlib.redirect_stdout(io.StringIO()):
            folder=train.main(base+['--updates','1'])
            train.main(['--init-checkpoint',str(path),'--updates','2'])
        saved=torch.load(folder/'policy.pt',weights_only=True)
        self.assertEqual(saved['update'],2)
        self.assertEqual(len(saved['training_log']['updates']),2)
        self.assertEqual(len(list((self.root/'rl').iterdir())),1)
        self.assertEqual(list(folder.glob('*.pt')),[folder/'policy.pt'])
        same=dict(saved);same['model_config']=dict(saved['model_config'],residual_blocks=0)
        self.assertEqual(variant_name(same),variant_name(saved))
        different=dict(saved);different['model_config']=dict(saved['model_config'],residual_blocks=3)
        self.assertNotEqual(variant_name(different),variant_name(saved))
        with training_lock(folder),self.assertRaises(RuntimeError):
            with training_lock(folder):pass

    def test_new_best_is_saved_and_survives_later_worse_updates(self):
        _,path=self.checkpoint()
        folder=self.root/'best-test'
        evaluations=[{'frozen':dict(score=score,mean_reward=0.)} for score in [.25,.75,.5]]
        base=['--init-checkpoint',str(path),'--run-dir',str(folder),'--opponent',str(path),
              '--eval-opponent',str(path),'--updates','2','--games-per-update','2',
              '--max-plies','4','--epochs','1','--eval-every','1','--eval-games','2',
              '--device','cpu','--threads','1']
        with patch('rl.train.evaluate',side_effect=evaluations),patch('rl.train.capture',return_value={}),patch('rl.train.plot_metrics'),contextlib.redirect_stdout(io.StringIO()):
            train.main(base)
        saved=torch.load(folder/'policy.pt',weights_only=True)
        self.assertEqual(saved['update'],2)
        self.assertEqual(saved['best_policy']['update'],1)
        self.assertEqual(saved['best_score'],.75)
        best,_=load_policy(folder/'policy.pt',best=True)
        self.assertTrue(torch.equal(best.policy.weight,saved['best_policy']['state_dict']['policy.weight']))
        self.assertFalse(torch.equal(best.policy.weight,saved['state_dict']['policy.weight']))

    def test_paired_games_record_both_sides_without_changing_weights(self):
        first=create_policy('feedforward',channels=2,hidden_size=8).eval()
        second=create_policy('feedforward',channels=3,hidden_size=9,residual_blocks=1).eval()
        before=[{k:v.clone() for k,v in model.state_dict().items()} for model in [first,second]]
        pairs=collect_pair([first,second],['cnn','residual'],2,max_plies=4)
        for model,previous,(batch,games) in zip([first,second],before,pairs):
            self.assertEqual(len(batch),4)
            self.assertEqual({g['color'] for g in games},{'white','black'})
            self.assertEqual([g['reward'] for g in games],[.1,.1])
            logp,_,_=evaluate_actions(model,batch.tokens,batch.masks,batch.actions,.9)
            self.assertTrue(torch.allclose(logp.detach(),batch.old_log_probabilities,atol=2e-5))
            for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,previous[k]),k)
        # Check perspective signs independently of stochastic game outcomes.
        from rl.rollout import Trajectories
        from rl.policy import Transition
        class Agent:pass
        agents=[Agent(),Agent()]
        for agent in agents:
            agent.transitions=[Transition(torch.zeros(64,dtype=torch.uint8),torch.ones(4096,dtype=torch.bool),0,0.,0.)]
        white,black=Trajectories(.99,.95),Trajectories(.99,.95)
        result=dict(result='1-0',reason='King capturable',frames=[{},{}])
        white.add(agents[0],result,True,'black');black.add(agents[1],result,False,'white')
        self.assertEqual(white.finish()[0].returns.tolist(),[1.])
        self.assertEqual(black.finish()[0].returns.tolist(),[-1.])

    def test_paired_training_updates_both_and_resumes_exactly(self):
        sources=[]
        for name,config in [('cnn',dict(channels=2,hidden_size=8)),
                            ('residual',dict(channels=3,hidden_size=9,residual_blocks=1))]:
            model=create_policy('feedforward',**config).eval()
            path=self.root/(name+'.pt')
            torch.save(dict(kind='feedforward',activation='relu',model_config=config,state_dict=model.state_dict()),path)
            sources.append(path)
        originals=[path.read_bytes() for path in sources]
        base=['--first',str(sources[0]),'--second',str(sources[1]),'--games-per-update','2',
              '--max-plies','4','--epochs','1','--eval-every','0','--device','cpu','--threads','1']
        with patch('rl.train.capture',return_value={}),patch('rl.train.plot_metrics'),contextlib.redirect_stdout(io.StringIO()):
            continuous=train_pair.main(base+['--run-root',str(self.root/'continuous-pair'),'--updates','2'])
            resumed=train_pair.main(base+['--run-root',str(self.root/'resumed-pair'),'--updates','1'])
            reversed_base=['--first',str(sources[1]),'--second',str(sources[0])]+base[4:]
            train_pair.main(reversed_base+['--run-root',str(self.root/'resumed-pair'),'--updates','2'])
        for source,original,a_folder,b_folder in zip(sources,originals,continuous,resumed):
            self.assertEqual(source.read_bytes(),original)
            a=torch.load(a_folder/'policy.pt',weights_only=True)
            b=torch.load(b_folder/'policy.pt',weights_only=True)
            old=torch.load(source,weights_only=True)
            self.assertEqual(b['update'],2)
            self.assertEqual(b['training_log']['training_mode'],'paired')
            self.assertEqual(b['opponent_history'],[])
            self.assertEqual(list(b_folder.glob('*.pt')),[b_folder/'policy.pt'])
            self.assertFalse(torch.equal(old['state_dict']['policy.weight'],b['state_dict']['policy.weight']))
            for k,v in a['state_dict'].items():self.assertTrue(torch.equal(v,b['state_dict'][k]),k)
            self.assertTrue(torch.equal(a['torch_rng_state'],b['torch_rng_state']))
            self.assertEqual(a['python_rng_state'],b['python_rng_state'])
            for row in b['training_log']['updates']:
                self.assertEqual(row['rollout']['draws'],2)
                self.assertEqual(row['rollout']['mean_reward'],.1)
                self.assertGreater(row['ppo']['optimizer_steps'],0)

    def test_paired_commit_recovers_interruption_between_model_saves(self):
        folders=[self.root/'a',self.root/'b']
        for folder in folders:folder.mkdir()
        pending={folder/'policy.pt':dict(update=1,state_dict={'weight':torch.tensor([i])})
                 for i,folder in enumerate(folders)}
        original_save=train_pair.save_checkpoint
        def interrupted(saved,path):
            if path==folders[1]/'policy.pt':raise OSError('Simulated interruption')
            original_save(saved,path)
        with patch('rl.train_pair.save_checkpoint',side_effect=interrupted),self.assertRaises(OSError):
            train_pair.commit(pending,folders)
        self.assertTrue((folders[0]/'.paired-commit.tmp').exists())
        with self.assertRaises(RuntimeError):require_completed_pair(folders[1])
        train_pair.recover(folders)
        for i,folder in enumerate(folders):
            saved=torch.load(folder/'policy.pt',weights_only=True)
            self.assertEqual(saved['update'],1)
            self.assertEqual(int(saved['state_dict']['weight'][0]),i)
        self.assertFalse((folders[0]/'.paired-commit.tmp').exists())

    def test_historical_value_targets_use_own_decision_discount_and_fog_only(self):
        replay=play_game(Scripted,Scripted,max_plies=4)
        replay['rulesVersion']='fow-v1'
        replay['result']='1-0';replay['reason']='King capturable'
        for frame in replay['frames']:frame['board']='hidden board must never be read'
        examples,why=pretrain_value.value_examples(replay,.9)
        self.assertIsNone(why)
        self.assertEqual([r for t,r in examples],[.9,-.9,1.,-1.])
        replay['result']='1/2-1/2';replay['reason']='Move limit reached'
        examples,_=pretrain_value.value_examples(replay,.9)
        self.assertTrue(torch.allclose(torch.tensor([r for t,r in examples]),torch.tensor([.09,.09,.1,.1])))
        replay['failedSide']='white'
        self.assertEqual(pretrain_value.value_examples(replay)[0],[])

    def test_value_warmup_changes_only_heads_and_preserves_continuation(self):
        sources=[]
        for name,c in [('cnn',2),('residual',3)]:
            config=dict(channels=c,hidden_size=8,batch_norm=True,dropout=.1,residual_blocks=0 if c==2 else 1)
            model=create_policy('feedforward',**config).eval();path=self.root/(name+'.pt')
            torch.save(dict(kind='feedforward',activation='relu',model_config=config,state_dict=model.state_dict()),path);sources.append(path)
        base=['--first',str(sources[0]),'--second',str(sources[1]),'--run-root',str(self.root/'pair'),
              '--games-per-update','2','--max-plies','4','--epochs','1','--eval-every','0','--device','cpu','--threads','1']
        with patch('rl.train.capture',return_value={}),patch('rl.train.plot_metrics'),contextlib.redirect_stdout(io.StringIO()):
            folders=train_pair.main(base+['--updates','1'])
        before=[torch.load(f/'policy.pt',weights_only=True) for f in folders]
        data=self.root/'data';data.mkdir();paths=[]
        for i in range(4):
            replay=play_game(Scripted,Scripted,max_plies=4);replay['rulesVersion']='fow-v1'
            p=data/(str(i)+'.json');p.write_text(json.dumps(replay));paths.append(str(p))
        split=self.root/'split.json';split.write_text(json.dumps(dict(train=paths[:2],validation=paths[2:])))
        cli=['--checkpoint',str(folders[0]/'policy.pt'),'--checkpoint',str(folders[1]/'policy.pt'),
             '--split',str(split),'--data-dir',str(data),'--batch-size','2','--threads','1','--after-update','1']
        with patch('rl.pretrain_value.plot_report'),contextlib.redirect_stdout(io.StringIO()):
            pretrain_value.main(cli)
            pretrain_value.main(cli)  # Does not repeat the epoch.
        for folder,old in zip(folders,before):
            new=torch.load(folder/'policy.pt',weights_only=True)
            self.assertEqual(new['update'],1)
            self.assertEqual(len(new['training_log']['value_pretraining']),1)
            self.assertFalse(torch.equal(old['state_dict']['value.weight'],new['state_dict']['value.weight']))
            for k,v in old['state_dict'].items():
                if not k.startswith('value.'):self.assertTrue(torch.equal(v,new['state_dict'][k]),k)
            self.assertTrue(torch.equal(old['torch_rng_state'],new['torch_rng_state']))
            actor_ids=old['optimizer_state_dict']['param_groups'][0]['params']
            for pid in actor_ids:
                for k,v in old['optimizer_state_dict']['state'].get(pid,{}).items():
                    if isinstance(v,torch.Tensor):self.assertTrue(torch.equal(v,new['optimizer_state_dict']['state'][pid][k]))
            for pid in old['optimizer_state_dict']['param_groups'][1]['params']:
                self.assertNotIn(pid,new['optimizer_state_dict']['state'])
        with patch('rl.train.capture',return_value={}),patch('rl.train.plot_metrics'),contextlib.redirect_stdout(io.StringIO()):
            train_pair.main(base+['--updates','2'])
        for folder in folders:
            new=torch.load(folder/'policy.pt',weights_only=True)
            self.assertEqual(new['update'],2)
            self.assertEqual(len(new['training_log']['value_pretraining']),1)

    def test_initial_critic_reset_leaves_warm_started_actor_unchanged(self):
        original,path=self.checkpoint()
        model,_=load_policy(path)
        tokens=torch.zeros(2,14,8,8)
        with torch.no_grad():before=model(tokens)[0].clone()
        initialize_value(model,{'kind':'feedforward'})
        with torch.no_grad():after,values,_=model(tokens)
        self.assertTrue(torch.equal(before,after))
        self.assertTrue((values==0).all())
        with torch.no_grad():model.value.bias.fill_(.25)
        initialize_value(model,{'algorithm':'ppo'})
        self.assertEqual(float(model.value.bias.detach()),.25)


if __name__=='__main__':unittest.main()
