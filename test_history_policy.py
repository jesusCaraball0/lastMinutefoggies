"""History alignment, isolation, protocol parity, PPO and checkpoint compatibility."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
import torch
from referee import FowBoard,make_frame
from starters.shared.model import create_policy,encode
from starters.shared.observation_history import ObservationHistory
from training.datasets import TOKEN,history_examples,observation_planes,ReplayDataset
from training.probes import PolicyProbes
from rl.policy import PolicyAgent,load_policy
from rl.game import play_game,Command
from rl.rollout import collect
from rl.ppo import PPOConfig,optimizer_for,update
from rl.storage import variant_name
from starters.shared.bundle import bundle_bot
from training import train
import sys

class HistoryTests(unittest.TestCase):
 def setUp(self):
  before=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,before)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def replay(self):
  b=FowBoard();frames=[make_frame(b,[180,180])]
  for move in ['e2e4','e7e5','g1f3','b8c6']:
   b.push_uci(move);frames.append(make_frame(b,[180,180],move))
  return dict(result='1-0',frames=frames)
 def test_order_padding_and_future_hidden_board_isolation(self):
  r=self.replay();xs=list(history_examples(r,4))
  self.assertEqual(xs[0][0],[TOKEN[c] for c in r['frames'][0]['white']+'?'*192])
  self.assertEqual(xs[1][0],[TOKEN[c] for c in r['frames'][1]['black']+r['frames'][0]['black']+'?'*128])
  self.assertEqual(xs[2][0],[TOKEN[c] for c in ''.join(r['frames'][j]['white'] for j in [2,1,0])+'?'*64])
  changed=json.loads(json.dumps(r));changed['frames'][3]['white']='?'*64
  for frame in changed['frames']:frame['board']='PRIVATE TRUE BOARD MUST NOT BE USED'
  self.assertEqual(list(history_examples(changed,4))[:3],xs[:3])
  h=ObservationHistory(4);h.observe(r['frames'][0]['white']);self.assertEqual(h.encoded_view(),r['frames'][0]['white']+'?'*192)
 def test_encoding_matches_standalone_and_old_single_frame(self):
  view=self.replay()['frames'][0]['white'];flat=view+'?'*64+view[::-1]+'-'*64
  tokens=torch.tensor([[TOKEN[c] for c in flat]])
  self.assertTrue(torch.equal(observation_planes(tokens,'cpu')[0],encode(flat)))
  self.assertTrue(torch.equal(observation_planes(tokens[:,:64],'cpu')[0],encode(view)))
  self.assertEqual(observation_planes(tokens[:,None],'cpu').shape,(1,1,56,8,8))
 def test_actual_game_recording_matches_imitation_history_and_reset(self):
  model=create_policy('feedforward',channels=2,hidden_size=8,history_frames=4)
  agents=[PolicyAgent(model,record=True),PolicyAgent(model,record=True)]
  replay=play_game(lambda:agents[0],lambda:agents[1],max_plies=8)
  self.assertFalse(replay.get('failedSide'))
  counters=[0,0]
  for frame,(tokens,action) in zip(replay['frames'],history_examples(replay,4)):
   side=frame['ply']%2;actual=agents[side].transitions[counters[side]];counters[side]+=1
   self.assertEqual(actual.tokens.tolist(),tokens);self.assertEqual(actual.action,action)
  agents[0].reset(True,180,2);self.assertEqual(agents[0].history.encoded_view(),'?'*256)
 def test_matched_initial_function_and_history_gradients(self):
  torch.manual_seed(77);current=create_policy('feedforward',channels=2,hidden_size=8).eval()
  torch.manual_seed(77);history=create_policy('feedforward',channels=2,hidden_size=8,history_frames=4).eval()
  x=torch.randn(3,56,8,8)
  a=current(x[:,:14])[0];b=history(x)[0]
  self.assertTrue(torch.allclose(a,b,atol=1e-6))
  b.sum().backward()
  self.assertGreater(float(history.encoder[0].weight.grad[:,14:].norm()),0)
 def test_history_training_probes_ppo_wire_and_legacy_variant_ids(self):
  data=self.root/'data';data.mkdir()
  for i in range(4):(data/f'{i}.json').write_text(json.dumps(self.replay()))
  run=self.root/'imitation'
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   train.main(['--data-dir',str(data),'--run-dir',str(run),'--history-frames','4','--epochs','1','--channels','2','--hidden-size','8','--batch-size','2','--schedule','none','--probe-max-positions','4','--threads','1','--device','cpu'])
  model,s=load_policy(run/'best_policy.pt');self.assertEqual(model.history_frames,4)
  probe=PolicyProbes(list(data.glob('*.json')),4,history_frames=4).evaluate(model,'cpu');self.assertGreater(probe['positions'],0)
  batch,_=collect(model,[('same',lambda:PolicyAgent(model))],2,max_plies=6)
  self.assertEqual(batch.tokens.shape[-1],256)
  cfg=PPOConfig(epochs=1,minibatch_size=4);result=update(model,optimizer_for(model,cfg),batch,cfg);self.assertGreater(result['optimizer_steps'],0)
  self.assertTrue(variant_name(s).startswith('history4-'))
  bundle=bundle_bot(Path(__file__).resolve().parent/'starters/neural',self.root/'bot',run/'best_policy.pt')
  wire=Command((sys.executable,'-u',str(bundle/'bot.py')))
  replay=play_game(wire,lambda:PolicyAgent(model),max_plies=4)
  self.assertFalse(replay.get('failedSide'))
  for p in (Path(__file__).resolve().parent/'out/rl').glob('*/policy.pt'):
   saved=torch.load(p,map_location='cpu',weights_only=True)
   self.assertEqual(variant_name(saved),p.parent.name)

if __name__=='__main__':unittest.main()
