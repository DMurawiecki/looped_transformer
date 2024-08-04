import torch
from tasks import get_task_sampler
from tqdm import tqdm
from train import calculate_gradient_norm
from eval_and_save import evaluate_model
from main_utils import gen_dataloader
from train_implementation import train_step
import wandb

def train_model_wandb(starting_step, ending_step, args, model, ctx, add_inputs_embeds, optimizer, curriculum, scaler, device, run_name):
  torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
  torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
  dtype = 'float16'  # 'bfloat16', 'float32'
  ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]  
  if ctx:
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype, cache_enabled=False)

  model.train()
  losses_train = []
  losses_val = []
  run = wandb.init(
    project= run_name,
    reinit=True
    config={
        "learning_rate": args['training']['learning_rate'],
        "epochs": ending_step - starting_step
    },
)

  task_sampler = get_task_sampler(
                      task_name=args['training']['task_name'],
                      batch_size=args['training']['batch_size'],
                      n_points=curriculum.n_points,
                      n_dims=args['model']['n_dims'],
                      n_dims_truncated=curriculum.n_dims_truncated,
                      device=device)
  test_size = 128
  test_loader = gen_dataloader(task_sampler,
                              test_size,
                              args['training']['batch_size'])
  for i in tqdm(range(starting_step, ending_step)):
    task_sampler = get_task_sampler(
                    task_name=args['training']['task_name'],
                    batch_size=args['training']['batch_size'],
                    n_points=curriculum.n_points,
                    n_dims=args['model']['n_dims'],
                    n_dims_truncated=curriculum.n_dims_truncated,
                    device=device)

    real_task = task_sampler()
    xs, ys = real_task.xs.float(), real_task.ys.float()


    loss, output, total_norm, grad_norm_dict = train_step(args= args,
                                                          curriculum= curriculum,
                                                          model= model,
                                                          xs = xs,
                                                          ys = ys,
                                                          optimizer=optimizer,
                                                          ctx= ctx,
                                                          scaler= scaler,
                                                          add_inputs_embeds= add_inputs_embeds)
    losses_train.append(loss)
    val_loss = evaluate_model(model= model, args= args, curriculum=curriculum, device=device, test_loader=test_loader)
    losses_val.append(val_loss)
    wandb.log({"training loss": loss, 
               "validation loss": val_loss},
               step = i)
  run.finish()

  return losses_train, losses_val
