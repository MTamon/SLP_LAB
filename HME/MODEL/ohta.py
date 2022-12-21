"""For Simple-Model Learning process"""

from tqdm import tqdm
import pickle

import torch
from torch.optim import AdamW

from src import add_datetime_path, fix_seed
from src import set_logger
from ohta_dataset import OhtaDataset
from ohta_dataloader import OhtaDataloader
from ohta_trainer import OhtaTrainer
from model.ohta.alpha import build_alpha
from model.ohta.alphaS import build_alphaS
from model.ohta.utils import get_args

SEED = 42
fix_seed(SEED)

args = get_args()
logger = set_logger("TRAIN", args.log)

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = OhtaDataset(**vars(args))
dataloader = OhtaDataloader(dataset, **vars(args))

if args.use_model == "alpha":
    model = build_alpha(**vars(args)).to(device=device)
elif args.use_model == "alphaS":
    model = build_alphaS(**vars(args)).to(device=device)
else:
    raise ValueError(f"Invalid model {args.use_model}")

optimizer = AdamW(
    params=model.parameters(),
    lr=args.lr,
    betas=args.betas,
    eps=args.eps,
    weight_decay=args.weight_decay,
)

trainer = OhtaTrainer(
    net=model,
    optimizer=optimizer,
    dataloader=dataloader,
    scaler=scaler,
    logger=logger,
    **vars(args),
)
with open(add_datetime_path("./loader/dataloader.dl"), "wb") as p:
    pickle.dump(trainer.train_valid_loader, p)

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
fpath = {"train": train_fpath, "valid": valid_fpath}


def process(_mode):
    phase_loss = []
    phase_acc = []

    text = ""

    trainer.set_mode(_mode)

    with tqdm(trainer, desc=_mode) as prog:
        for i, (loss, acc) in enumerate(prog):

            if i % 100 == 0:
                with open(fpath[_mode], mode="a", encoding="utf-8") as f:
                    f.write(text)
                    text = ""
            else:
                text += f"{loss},{acc}\n"

            prog.postfix = f"L:{round(loss, 2)}, A:{round(acc, 2)}"

            phase_loss.append(loss)
            phase_acc.append(acc)

    phase_loss = sum(phase_loss) / len(phase_loss)
    phase_acc = sum(phase_acc) / len(phase_acc)

    return phase_loss, phase_acc


if not args.skip_first_valid:
    logger.info(" Init Valid-Mode >>> ")
    _loss, _acc = process("valid")
    logger.info(" Result |[ Loss : %s, Acc : %s ]|", round(_loss, 2), round(_acc, 2))

for current_epoch in range(args.epoch):
    logger.info(" Epoch >>> %s / %s", (current_epoch + 1), args.epoch)

    for mode in ["train", "valid"]:
        _loss, _acc = process(mode)
        logger.info(" Result |[ Loss : %s, Acc : %s ]|", _loss, _acc)

    epo_inf = f"0{current_epoch}" if current_epoch < 10 else str(current_epoch)
    path = ".".join(args.model_save_path.split(".")[:-1]) + f"E{epo_inf}.pth"
    trainer.save_model(path)
