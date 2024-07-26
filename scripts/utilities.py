import torch
import yaml

def set_optim_scaler_curriculum_by_args(args_path):
  with open(args_path, 'r') as config_file:
    args = yaml.safe_load(config_file)

  optimizer = torch.optim.Adam(model.parameters(), lr=args['training']['learning_rate'], weight_decay=args['training']['weight_decay'])
  scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
  curriculum = Curriculum(args['training']['curriculum'])
  return optimizer, scaler, curriculum
