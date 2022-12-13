"""For Simple-Model Learning process"""

from tqdm import tqdm

import torch
from torch.optim import AdamW

from utils import get_args, add_datetime_path, fix_seed
from logger_gen import set_logger
from hme_dataset import HmeDataset
from hme_dataloader import HmeDataloader
from hme_trainer import HmeTrainer
from simple import SimpleModel

SEED = 42
fix_seed(SEED)

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


def process(_mode):
    phase_loss = []
    phase_acc = []

    with tqdm(trainer, desc=_mode) as prog:
        for loss, acc in prog:

            if _mode == "train":
                with open(train_fpath, mode="a", encoding="utf-8") as f:
                    f.write(f"{loss},{acc}\n")
            else:
                with open(valid_fpath, mode="a", encoding="utf-8") as f:
                    f.write(f"{loss},{acc}\n")

            _loss = round(float(loss.to("cpu")), 2)
            _acc = round(float(acc.to("cpu")), 2)
            prog.postfix = f"L:{_loss}, A:{_acc}"

            phase_loss.append(float(loss))
            phase_acc.append(float(acc))

    phase_loss = sum(phase_loss) / len(phase_loss)
    phase_acc = sum(phase_acc) / len(phase_acc)

    return phase_loss, phase_acc


for current_epoch in range(args.epoch):
    logger.info(" Epoch >>> %s / %s", (current_epoch + 1), args.epoch)

    for mode in ["train", "valid"]:
        trainer.set_mode(mode)

        _loss, _acc = process(mode)
        logger.info(" Result |[ Loss : %s, Acc : %s ]|", _loss, _acc)

    epo_inf = f"0{current_epoch}" if current_epoch < 10 else str(current_epoch)
    path = ".".join(args.model_save_path.split(".")[:-1]) + f"E{epo_inf}.pth"
    trainer.save_model(path)
