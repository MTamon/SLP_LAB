"""For Simple-Model Learning process"""

from tqdm import tqdm

import torch
from torch.optim import AdamW

from utils import get_args, add_datetime_path
from logger_gen import set_logger
from hme_dataset import HmeDataset
from hme_dataloader import HmeDataloader
from hme_trainer import HmeTrainer
from simple import SimpleModel

args = get_args()
logger = set_logger("TRAIN", args.log)

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = HmeDataset(**vars(args))
dataloader = HmeDataloader(dataset, **vars(args))
model = SimpleModel(**vars(args), device=device)

optimizer = AdamW(
    params=model.parameters(),
    lr=args.lr,
    betas=args.betas,
    eps=args.eps,
    weight_decay=args.weight_decay,
)

trainer = HmeTrainer(
    net=model,
    optimizer=optimizer,
    dataloader=dataloader,
    scaler=scaler,
    logger=logger,
    **vars(args),
)

train_length = len(trainer.train_valid_loader["train"])
valid_length = len(trainer.train_valid_loader["valid"])

train_fpath = add_datetime_path(args.train_result_path)
valid_fpath = add_datetime_path(args.valid_result_path)
t_f = open(train_fpath, mode="w", encoding="utf-8")
t_f.write(f"loss, acc, {train_length}\n")
t_f.close()
v_f = open(valid_fpath, mode="w", encoding="utf-8")
v_f.write(f"loss, acc, {valid_length}\n")
v_f.close()

for current_epoch in range(args.epoch):
    epoch_loss = []
    epoch_acc = []
    logger.info(" Epoch >>> %s / %s", (current_epoch + 1), args.epoch)

    for mode in ["train", "valid"]:
        trainer.set_mode(mode)

        with tqdm(trainer, desc=mode) as prog:
            for loss, acc in prog:

                if mode == "train":
                    with open(train_fpath, mode="a", encoding="utf-8") as f:
                        f.write(f"{loss},{acc}\n")
                else:
                    with open(valid_fpath, mode="a", encoding="utf-8") as f:
                        f.write(f"{loss},{acc}\n")

                _loss = round(float(loss), 2)
                _acc = round(float(acc), 2)
                prog.postfix = f"L:{_loss}, A:{_acc}"

    epo_inf = f"0{current_epoch}" if current_epoch < 10 else str(current_epoch)
    path = ".".join(args.model_save_path.split(".")[:-1]) + f"E{epo_inf}.pth"
    trainer.save_model(path)
