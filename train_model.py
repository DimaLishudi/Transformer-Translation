from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm.auto import trange, tqdm

from data import TranslationDataset
from decoding import translate, get_attn_mask
from model import TranslationModel

import wandb

def train_epoch(
    model: TranslationModel,
    train_dataloader,
    CELoss,
    optimizer,
    scheduler,
    device,
    src_tokenizer,
    tgt_tokenizer,
    logger=None,
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    model.to(device)

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    total_loss = 0
    total_size = 0

    for batch in tqdm(train_dataloader):
        #getting data 
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        bs, tgt_len = tgt.shape

        # building masks
        tgt_attn_mask = get_attn_mask(tgt_len-1).to(device)
        src_pad_mask = (src == src_pad_id).to(device)
        tgt_pad_mask = (tgt[:,:-1] == tgt_pad_id).to(device)
        # print(tgt.shape, src.shape, tgt_attn_mask.shape, src_pad_mask.shape, tgt_pad_mask.shape)

        # forward, give target except ["EOS"]
        out = model(tgt[:,:-1], src, tgt_attn_mask, src_pad_mask, tgt_pad_mask)
        # compare to target except ["BOS"]
        out = out.reshape(bs * (tgt_len-1), tgt_vocab_size)
        loss = CELoss(out, tgt[:,1:].reshape(-1))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        total_loss += loss.cpu().item() * bs
        total_size += bs
        if logger is not None:
            log = {
                "loss" : loss.cpu().item(),
                "lr"   : scheduler.get_last_lr()[0]
            }
            logger.log(log)

    return total_loss / total_size




@torch.inference_mode()
def evaluate(
    model: TranslationModel,
    val_dataloader,
    CELoss,
    device,
    src_tokenizer,
    tgt_tokenizer,
    logger=None
    ):
    # compute the loss over the entire validation subset
    model.eval()
    model.to(device)

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    total_loss = 0
    total_size = 0

    for batch in tqdm(val_dataloader):
        #getting data 
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        bs, tgt_len = tgt.shape

        # building masks
        tgt_attn_mask = get_attn_mask(tgt_len-1).to(device)
        src_pad_mask = (src == src_pad_id).to(device)
        tgt_pad_mask = (tgt[:,:-1] == tgt_pad_id).to(device)
        # print(tgt.shape, src.shape, tgt_attn_mask.shape, src_pad_mask.shape, tgt_pad_mask.shape)

        # forward, give target except ["EOS"]
        out = model(tgt[:,:-1], src, tgt_attn_mask, src_pad_mask, tgt_pad_mask)
        # compare to target except ["BOS"]
        out = out.reshape(bs * (tgt_len-1), tgt_vocab_size)
        loss = CELoss(out, tgt[:,1:].reshape(-1))

        # loss calc
        total_loss += loss.cpu().item() * bs
        total_size += bs

    return total_loss / total_size


def train_model(data_dir, tokenizer_path, num_epochs, enable_wandb):
    config = {
        "batch_size" : 16,
        "lr" : 3e-4,
        "max_len" : 128,  # might be enough at first
        "num_encoder_layers" : 1,
        "num_decoder_layers" : 1,
        "emb_size" : 256,
        "dim_feedforward" : 1024,
        "n_head" : 8,
        "dropout_prob" : 0.1,
    }

    src_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_de.json"))
    tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / "tokenizer_en.json"))

    train_dataset = TranslationDataset(
        data_dir / "train.de.txt",
        data_dir / "train.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=config["max_len"],
    )
    val_dataset = TranslationDataset(
        data_dir / "val.de.txt",
        data_dir / "val.en.txt",
        src_tokenizer,
        tgt_tokenizer,
        max_len=config["max_len"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    src_pad_id = tgt_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")

    model = TranslationModel(
        config["num_encoder_layers"],
        config["num_decoder_layers"],
        config["emb_size"],
        config["dim_feedforward"],
        config["n_head"],
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size(),
        config["dropout_prob"],
        src_pad_id,
        tgt_pad_id,
        config["max_len"]
    )


    if enable_wandb:
        wandb.init(config=config, project="DL-LHW2")
        logger = wandb
    else:
        logger = None

    print("Total no. of model parameters:",
        "pytorch_total_params =", sum(p.numel() for p in model.parameters())
    )
    model.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        config["batch_size"],
        collate_fn = train_dataset.collate_translation_data,
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        config["batch_size"],
        collate_fn = val_dataset.collate_translation_data,
    )
    print(len(val_dataloader))

    # standard for transformers Adam optimizer + OneCycle scheduler
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config["lr"],
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1
    )
    CELoss = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad_id)

    min_val_loss = float("inf")

    for epoch in trange(1, num_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, CELoss, optimizer, scheduler, device, src_tokenizer, tgt_tokenizer, logger)
        val_loss = evaluate(model, val_dataloader, CELoss, device, src_tokenizer, tgt_tokenizer, logger)
        if logger is not None:
            logger.log({
                "epoch" : epoch,
                "val_loss" : val_loss
            })

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer' : optimizer,
                    'scheduler' : scheduler
                }, "checkpoint_best.pth")
            min_val_loss = val_loss

        torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer' : optimizer,
                'scheduler' : scheduler
            }, "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth"))
    return model


def translate_test_set(model: TranslationModel, data_dir, tokenizer_path):
    model.eval()

    greedy_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_greedy.txt", "w+"
    ) as output_file:
        # translate with greedy search
        pass

    beam_translations = []
    with open(data_dir / "test.de.txt") as input_file, open(
        "answers_beam.txt", "w+"
    ) as output_file:
        # translate with beam search
        pass

    with open(data_dir / "test.en.txt") as input_file:
        references = [line.strip() for line in input_file]

    bleu = BLEU()
    bleu_greedy = bleu.corpus_score(greedy_translations, [references]).score

    # we're recreating the object, as it might cache some stats
    bleu = BLEU()
    bleu_beam = bleu.corpus_score(beam_translations, [references]).score

    print(f"BLEU with greedy search: {bleu_greedy}, with beam search: {bleu_beam}")
    # maybe log to wandb/comet/neptune as well


if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )
    data_group.add_argument(
        "--tokenizer-path", type=Path, help="Path to the trained tokenizer files"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--enable-wandb", type=bool, default=False, help="log to wandb", action=BooleanOptionalAction
    )

    args = parser.parse_args()

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs, args.enable_wandb)
    translate_test_set(model, args.data_dir, args.tokenizer_path)
