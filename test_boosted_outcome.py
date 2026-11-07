"""Joint pretraining must learn values without leaking outcome labels into inputs."""
import contextlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from referee import FowBoard,make_frame
from starters.shared.model import create_policy
from training.outcomes import BoostedOutcomeDataset,outcome_label
from training.boosted_dataset import BoostedReplayDataset
from training import train
from rl.policy import load_policy,initialize_value,PolicyAgent
from rl.storage import variant_name
from rl.rollout import collect
from rl.ppo import update,optimizer_for,PPOConfig
from starters.shared.bundle import bundle_bot
from rl.game import play_game,Command

class OutcomeTests(unittest.TestCase):
 def setUp(self):
  old=torch.get_num_threads();torch.set_num_threads(1);self.addCleanup(torch.set_num_threads,old)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def replay(self,result='1-0'):
  b=FowBoard();frames=[make_frame(b,[180,180])]
  for move in ['e2e4','d7d5','e4d5','d8d5','g1f3','g8f6']:
   b.push_uci(move);frames.append(make_frame(b,[180,180],move))
  return dict(result=result,frames=frames,rulesVersion='fow-v1',reason='King capturable')
 def test_perspective_draw_failure_and_no_label_leak(self):
  paths=[]
  for i,result in enumerate(['1-0','0-1','1/2-1/2']):
   p=self.root/f'{i}.json';p.write_text(json.dumps(self.replay(result)));paths.append(p)
  ds=BoostedOutcomeDataset(paths,self.root/'cache');base=BoostedReplayDataset(paths,self.root/'cache')
  self.assertEqual(ds.directory,base.directory);self.assertEqual(ds.outcome_positions,18)
  self.assertEqual([float(ds[i][1]['outcome'][0]) for i in range(4)],[1,-1,1,-1])
  self.assertEqual(float(ds[6][1]['outcome'][0]),-1)
  self.assertAlmostEqual(float(ds[12][1]['outcome'][0]),.1)
  # Identical observations/actions and different final results have identical inputs.
  for i in range(6):
   self.assertTrue(np.array_equal(ds[i][0],ds[i+6][0]));self.assertTrue(np.array_equal(ds[i][0],base[i][0]))
  bad=self.replay();bad['failedSide']='black';self.assertEqual(outcome_label(bad,True),(0.,False))
  bad.pop('failedSide');bad['reason']='Timeout';self.assertFalse(outcome_label(bad,True)[1])
  p=self.root/'bad.json';p.write_text(json.dumps(bad));masked=BoostedOutcomeDataset([p],self.root/'cache')
  self.assertEqual(masked.positions,6);self.assertEqual(masked.outcome_positions,0)
 def test_identical_initialization_old_ids_and_value_preservation(self):
  config=dict(channels=32,hidden_size=128,batch_norm=True,residual_blocks=3,dropout=.1,boosted=True)
  torch.manual_seed(7980);a=create_policy('feedforward',**config)
  torch.manual_seed(7980);b=create_policy('feedforward',**config,outcome_supervision=True)
  self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
  self.assertEqual(sum(p.numel() for p in b.parameters()),1370625)
  self.assertEqual(variant_name(dict(kind='feedforward',model_config=config)),'boosted-cnn-c32-h128-f4004a979d')
  for c,expected in [(dict(channels=64,hidden_size=256,batch_norm=True),'cnn-c64-h256-8ebd1cf653'),(dict(channels=32,hidden_size=128,batch_norm=True,residual_blocks=3,dropout=.1),'residual-c32-h128-518d814dce')]:
   self.assertEqual(variant_name(dict(kind='feedforward',model_config=c)),expected)
  self.assertTrue(variant_name(dict(kind='feedforward',model_config=dict(config,outcome_supervision=True))).startswith('boosted-outcome-cnn-'))
  old=b.value.weight.detach().clone();initialize_value(b,{'value_pretrained':True});self.assertTrue(torch.equal(old,b.value.weight))
 def test_outcome_gradient_reaches_encoder_without_policy_gradient(self):
  m=create_policy('feedforward',channels=2,hidden_size=8,residual_blocks=1,boosted=True,outcome_supervision=True)
  logits,value,_=m(torch.randn(2,1819,8,8))
  loss=.5*(value[:,0]-torch.tensor([1.,-1.])).square().mean();loss.backward()
  self.assertGreater(float(m.encoder[0].weight.grad.abs().sum()),0.)
  self.assertIsNone(m.policy.weight.grad)
 def test_joint_value_gradients_resume_ppo_and_bundle(self):
  data=self.root/'data';data.mkdir()
  for i in range(3):(data/f'{i}.json').write_text(json.dumps(self.replay('1-0' if i%2==0 else '0-1')))
  run=self.root/'run'
  args=['--data-dir',str(data),'--run-dir',str(run),'--boosted-outcome','--channels','2','--hidden-size','8','--residual-blocks','1','--batch-norm','--epochs','2','--batch-size','2','--schedule','none','--device','cpu','--threads','1','--probe-max-positions','4','--selection-metric','legal_accuracy']
  original_save=train.save_checkpoint
  class StopAfterEpoch(Exception):pass
  def stop_after(checkpoint,path):
   original_save(checkpoint,path)
   if path.name=='last_policy.pt' and checkpoint['epoch']==1:raise StopAfterEpoch()
  with patch.dict(os.environ,{'FOW_BOOSTED_CACHE_DIR':str(self.root/'cache')}),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   with patch.object(train,'save_checkpoint',side_effect=stop_after),self.assertRaises(StopAfterEpoch):train.main(args)
   first=torch.load(run/'last_policy.pt',weights_only=True);self.assertEqual(first['epoch'],1)
   self.assertGreater(first['metrics']['train']['gradient_l2']['value']['pre_clip']['mean'],0)
   train.main(['--resume',str(run/'last_policy.pt'),'--device','cpu','--threads','1'])
  m,s=load_policy(run/'final_policy.pt');self.assertTrue(s['value_pretrained']);self.assertTrue(m.outcome_supervision)
  self.assertEqual(len(s['training_log']['epochs']),2)
  self.assertTrue((run/'outcome_training.png').exists())
  self.assertEqual(s['metrics']['validation']['outcome']['positions'],6)
  self.assertGreater(s['metrics']['validation']['total_loss'],s['metrics']['validation']['loss'])
  original_value=m.value.weight.detach().clone();initialize_value(m,s);self.assertTrue(torch.equal(original_value,m.value.weight))
  batch,_=collect(m,[('same',lambda:PolicyAgent(m))],2,max_plies=4)
  report=update(m,optimizer_for(m,PPOConfig(epochs=1)),batch,PPOConfig(epochs=1));self.assertGreater(report['optimizer_steps'],0)
  bundle=self.root/'bundle';bundle_bot(Path('starters/boosted_outcome_cnn'),bundle,run/'best_policy.pt')
  replay=play_game(Command((sys.executable,'-u',str(bundle/'bot.py'))),lambda:PolicyAgent(m),max_plies=4)
  self.assertFalse(replay.get('failedSide'),replay['reason'])

if __name__=='__main__':unittest.main()
