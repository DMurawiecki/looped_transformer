import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from tasks import get_task_sampler
from main_utils import gen_dataloader

def evaluate_model(model, args, curriculum, device, test_size=30):
  task_sampler = get_task_sampler(
                      task_name=args['training']['task_name'],
                      batch_size=args['training']['batch_size'],
                      n_points=curriculum.n_points,
                      n_dims=args['model']['n_dims'],
                      n_dims_truncated=curriculum.n_dims_truncated,
                      device=device)

  test_loader = gen_dataloader(task_sampler,
                              test_size,
                              args['training']['batch_size'])
  test_losses = []

  model.eval()

  with torch.no_grad():
    for batch in tqdm(test_loader):
      xs, ys = batch['x'].to(device), batch['y'].to(device)
      if args['model']['family'] in ['gpt2']:
        output = model(xs, ys)  # [B,]
      elif args['model']['family'] in ['gpt2_loop']:
        n_loops = curriculum.n_loops  # K
        y_pred_list = model(xs, ys, 0, n_loops)
        output = y_pred_list[-1]  # [B, n]
        point_wise_loss = (output - ys).square().mean(dim=0)
      else:
        raise NotImplementedError
      point_wise_loss = (output - ys).square().mean(dim=0)
      loss = point_wise_loss.mean().detach().cpu()
      test_losses.append(loss)
  return (sum(test_losses)/len(test_losses)).item()
  # plt.plot(test_losses, color = 'salmon')
  # print('mean test loss: ', (sum(test_losses)/len(test_losses)).item())


def save_model(model, path):
    training_state = {
                    "model_state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    # "train_step": i,
                }
    torch.save(training_state, path)
    print('your model was saved')
