from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

import logging
from core import dataloader
from core.clip_logger import log_setting
from core.utils import set_seed, get_train_args
from core.augmentation import train_aug
from core.model import load_model, get_optimizer
from core import train


def finetune(args):
    # initial setting
    set_seed(args.seed) # seed
    log_setting(args.save_log_dir) # logger
    writer = SummaryWriter(args.save_log_dir) # tensorboard
    weights_path = Path(args.save_log_dir) # save path
    weights_path.mkdir(exist_ok=True)
    model, preprocess = load_model(args.model_name, args.device) # load model
    params, optimizer, scheduler = get_optimizer(model) # optimizer
    loss_img = torch.nn.CrossEntropyLoss() # img loss
    loss_txt = torch.nn.CrossEntropyLoss() # txt loss

    train_dataloader, test_dataloader = dataloader.get_subclass_loader(
        train_path=args.train_path,
        test_path=args.test_path,
        train_aug=train_aug(),
        test_aug=preprocess,
        train_batch_size=args.batch_size,
        test_batch_size=1
    )

    ## logging
    logging.info(f"MODEL_NAME : {args.model_name}")
    logging.info(f"BATCH_SIZE : {args.batch_size}")
    logging.info(f"SEED : {args.seed}")
    logging.info(f"NUM_EPOCHS : {args.num_epochs}")
    logging.info(f"TRAIN_PATH : {args.train_path}")
    logging.info(f"TEST_PATH : {args.test_path}")
    logging.info(f"OPTIMIZER : {optimizer}")
    logging.info(f"LOSS : {loss_img}")
    logging.info(f"TRAIN_AUG : {train_aug()}")
    logging.info(f"TEST_AUG : {preprocess}")
    logging.info(F"MODEL : {model}")

    train.train_class(
        args=args,
        model=model,
        weights_path=weights_path,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        use_albumentations=True,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_img=loss_img,
        loss_txt=loss_txt,
        writer=writer,
        device=device
    )

if __name__ == '__main__':
    parser = get_train_args()
    args = parser.parse_args()
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    args.device = device
    finetune(args)
