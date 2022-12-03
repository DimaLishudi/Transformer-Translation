from argparse import ArgumentParser
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU
from tokenizers import Tokenizer
from tqdm import trange

from data import TranslationDataset
from decoding import translate
from model import TranslationModel


def train_epoch(
    model: TranslationModel,
    train_dataloader,
    optimizer,
    scheduler,
    CELoss,
    device,
):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    model.to(device)

    total_loss = 0
    total_size = 0

    for batch in train_dataloader():
        batch.to(device) # TODO: fix
        bs = 0 # TODO: fix
        out = model(batch) # TODO: fix
        loss = CELoss(batch["target"], out) # TODO: fix
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.cpu().item() * bs
        total_size += bs

    return total_loss / total_size




@torch.inference_mode()
def evaluate(model: TranslationModel, val_dataloader, device):
    # compute the loss over the entire validation subset
    model.eval()
    pass


def train_model(data_dir, tokenizer_path, num_epochs):
    config = {
        "batch_size" : 32,
        "lr" : 3e-4,
        "max_len" : 128,  # might be enough at first
        "num_encoder_layers" : 3,
        "num_decoder_layers" : 3,
        "emb_size" : 256,
        "dim_feedforward" : 256,
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

    src_pad_id = tgt_tokenizer.convert_tokens_to_ids("[PAD]")
    tgt_pad_id = tgt_tokenizer.convert_tokens_to_ids("[PAD]")

    model = TranslationModel(
        config["num_encoder_layers"],
        config["num_decoder_layers"],
        config["emb_size"],
        config["dim_feedforward"],
        config["n_head"],
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size,
        config["dropout_prob"],
        src_pad_id,
        tgt_pad_id,
        config["max_len"]
    )
    print("Total no. of model parameters:",
        pytorch_total_params = sum(p.numel() for p in model.parameters())
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

    # standard for transformers Adam optimizer + OneCycle scheduler
    optimizer = torch.optim.Adam(model.parameters(), config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config["lr"],
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1
    )
    CELoss = torch.nn.CrossEntropyLoss(ignore_idx=tgt_pad_id)

    min_val_loss = float("inf")

    for epoch in trange(1, num_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, CELoss, optimizer, scheduler, device)
        val_loss = evaluate(model, val_dataloader, CELoss, device)

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print("New best loss! Saving checkpoint")
            torch.save({
                    'model_state_dict' : model.state_dict(),
                    'optimizer' : opt
                }, "checkpoint_best.pth")
            min_val_loss = val_loss

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

    args = parser.parse_args()

    model = train_model(args.data_dir, args.tokenizer_path, args.num_epochs)
    translate_test_set(model, args.data_dir, args.tokenizer_path)
