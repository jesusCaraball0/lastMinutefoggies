"""Boosted feature provenance,20-own-turn indexing,protocol/PPO compatibility."""
import contextlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import chess,numpy as np,torch
from referee import FowBoard,make_frame
from starters.shared.boosted_features import BoostedHistory,move_code,boosted_planes,CHANNELS,TOKEN_COUNT,RECORD_SIZE,TURNS
from starters.shared.model import create_policy
from starters.shared.moves import ObservationTracker
from training.boosted_dataset import BoostedReplayDataset,boosted_rows
from training.datasets import observation_planes
from training.probes import PolicyProbes
from training import train
from rl.policy import PolicyAgent,load_policy
from rl.game import play_game,Command
from rl.rollout import collect
from rl.ppo import PPOConfig,optimizer_for,update
from rl.storage import variant_name
from starters.shared.bundle import bundle_bot

class BoostedTests(unittest.TestCase):
 def setUp(self):
  old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def replay(self,moves=('e2e4','d7d5','e4d5','d8d5','g1f3','g8f6')):
  b=FowBoard();frames=[make_frame(b,[180,180])]
  for move in moves:b.push_uci(move);frames.append(make_frame(b,[180,180],move))
  return dict(result='1-0',frames=frames)
 def dataset(self,replays):
  paths=[]
  for i,r in enumerate(replays):
   p=self.root/f'{i}.json';p.write_text(json.dumps(r));paths.append(p)
  return BoostedReplayDataset(paths,self.root/'cache'),paths
 def test_move_encoding_unique_rays_knights_promotions(self):
  for color,move in [(True,'e2e4'),(False,'g8f6'),(True,'e1g1'),(True,'a7a8q'),(True,'a7a8n'),(False,'h2h1b')]:
   c=move_code(chess.Move.from_uci(move),color);self.assertTrue(0<=c<73)
  self.assertNotEqual(move_code(chess.Move.from_uci('a7a8q'),True),move_code(chess.Move.from_uci('a7a8n'),True))
 def test_action_target_excluded_captures_and_ages(self):
  r=self.replay();rows=list(boosted_rows(r))
  self.assertEqual(rows[0][0][64:67],[64,73,64])
  self.assertEqual(rows[2][0][64],chess.E2)
  self.assertEqual(rows[2][0][65],move_code(chess.Move.from_uci('e2e4'),True))
  self.assertEqual(rows[4][0][67],chess.D5)
  self.assertEqual(rows[4][0][68],chess.D5)
  changed=json.loads(json.dumps(r))
  for f in changed['frames']:f['board']='NEVER READ HIDDEN BOARD'
  # A later legal own target change cannot change earlier feature rows.
  changed['frames'][-1]['white']='?'*64
  for before,after in zip(rows,list(boosted_rows(changed))):self.assertEqual(before,after)
  h=BoostedHistory(True);v='?'*64;v='r'+v[1:];h.observe(v,0);h.observe('?'*64,1);h.observe('?'*64,2)
  x=boosted_planes(torch.tensor([h.encoded_tokens(0)],dtype=torch.uint8))
  self.assertEqual(x.shape,(1,CHANNELS,8,8))
  offset=20*90+2
  self.assertEqual(float(x[0,offset+9,0,0]),1.)
  self.assertAlmostEqual(float(x[0,offset+14,0,0]),2/254,places=6)
  self.assertEqual(float(x[0,offset+15,0,0]),1.)
  self.assertEqual(float(x[0,offset+16,0,0]),0.)
  self.assertEqual(float(x[0,offset+15,0,1]),0.)
 def test_cache_stacks_stay_within_game_and_player(self):
  r=self.replay();ds,paths=self.dataset([r,r]);self.assertEqual(ds.positions,12)
  self.assertEqual(ds.rows.shape[1],203)
  a=ds[2][0][0];self.assertEqual(a.shape,(TOKEN_COUNT,))
  self.assertEqual(a[2+RECORD_SIZE+69],1)
  self.assertEqual(a[2+2*RECORD_SIZE+72],0)
  self.assertEqual(ds[6][0][0][2+RECORD_SIZE+72],0)
  ds2=BoostedReplayDataset(paths,self.root/'cache');self.assertEqual(ds.directory,ds2.directory)
  self.assertTrue(np.array_equal(ds[4][0],ds2[4][0]))
 def test_twenty_own_turns_and_observed_en_passant(self):
  r=self.replay(['g1f3','g8f6','f3g1','f6g8']*12)
  ds,_=self.dataset([r]);packed=ds[46][0][0]
  records=packed[2:2+TURNS*RECORD_SIZE].reshape(TURNS,RECORD_SIZE)
  self.assertEqual(int(records[:,72].sum()),20)
  self.assertTrue((records[:,69]==1).all())
  ep=self.replay(['e2e4','a7a6','e4e5','d7d5','e5d6','g8f6','g1f3'])
  rows=list(boosted_rows(ep));self.assertEqual(rows[6][0][67],chess.D5)
 def test_live_features_match_imitation_and_probe_inputs(self):
  model=create_policy('feedforward',channels=2,hidden_size=8,boosted=True).eval()
  agents=[PolicyAgent(model,record=True),PolicyAgent(model,record=True)]
  r=play_game(lambda:agents[0],lambda:agents[1],max_plies=8)
  self.assertFalse(r.get('failedSide'))
  ds,paths=self.dataset([r]);counts=[0,0]
  for i in range(len(ds)):
   side=i%2;self.assertEqual(ds[i][0][0].tolist(),agents[side].transitions[counts[side]].tokens.tolist());counts[side]+=1
  probes=PolicyProbes(paths,ds.positions,boosted=True)
  for side,trajectory in enumerate(probes.trajectories):
   for i in trajectory['masks']:
    self.assertTrue(torch.equal(trajectory['tokens'][i],torch.from_numpy(ds[i][0][0])))
  value=probes.evaluate(model,'cpu');self.assertEqual(value['positions'],len(ds))
  agents[0].reset(True,180,2);self.assertFalse(agents[0].boosted_history.records)
 def test_tiny_training_resume_ppo_and_export(self):
  data=self.root/'data';data.mkdir()
  for i in range(3):(data/f'{i}.json').write_text(json.dumps(self.replay()))
  run=self.root/'run'
  with patch.dict(os.environ,{'FOW_BOOSTED_CACHE_DIR':str(self.root/'cache')}),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   train.main(['--data-dir',str(data),'--run-dir',str(run),'--boosted','--channels','2','--hidden-size','8','--epochs','1','--batch-size','2','--schedule','none','--device','cpu','--threads','1','--probe-max-positions','4'])
   train.main(['--resume',str(run/'last_policy.pt'),'--device','cpu','--threads','1'])
  model,s=load_policy(run/'best_policy.pt');self.assertTrue(model.boosted)
  self.assertEqual(model.input_channels,1819);self.assertTrue(variant_name(s).startswith('boosted-cnn-'))
  self.assertGreater(float(model.encoder[0].weight[:,14:].detach().abs().sum()),0.)
  batch,_=collect(model,[('same',lambda:PolicyAgent(model))],2,max_plies=4)
  cfg=PPOConfig(epochs=1,minibatch_size=2);result=update(model,optimizer_for(model,cfg),batch,cfg);self.assertGreater(result['optimizer_steps'],0)
  bundled=bundle_bot(Path(__file__).resolve().parent/'starters/boosted_cnn',self.root/'bundle',run/'best_policy.pt')
  for white in [True,False]:
   bot=Command((sys.executable,'-u',str(bundled/'bot.py')));agent=lambda:PolicyAgent(model)
   r=play_game(bot if white else agent,agent if white else bot,max_plies=6);self.assertFalse(r.get('failedSide'),r['reason'])
 def test_old_variant_ids_preserved(self):
  for p in (Path(__file__).resolve().parent/'out/rl').glob('*/policy.pt'):
   s=torch.load(p,map_location='cpu',weights_only=True);self.assertEqual(variant_name(s),p.parent.name)

if __name__=='__main__':unittest.main()
