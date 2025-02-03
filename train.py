from src.model import DenoisingModel
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import sys
import torch
from utils import get_dataloaders

def main(args):
    train_dl, val_dl = get_dataloaders(args)
    
    net = DenoisingModel(input_channels=args.input_channels, fixed_sigma=args.fixed_sigma, sigma=args.noise_rate)
    
    tags = [args.batch_size, args.noise_rate, args.epochs, args.dataset]
    if args.fp64:
        tags.append('fp64')
    
    logger = WandbLogger(project='denoisy', 
                         name=f'{args.dataset}-B{args.batch_size}-N{args.noise_rate}-E{args.epochs}',
                         entity='elte-ai4covid')
    
    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      log_every_n_steps=len(train_dl)//10)
    trainer.fit(net, train_dl, val_dl)
    
if __name__ == '__main__':
    from utils import get_parser
    parser = get_parser()
    
    args = parser.parse_args()
    
    if args.fp64:
        torch.set_default_dtype(torch.float64)
    
    sys.exit(main(args))