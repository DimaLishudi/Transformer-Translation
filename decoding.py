import torch
from sacremoses import MosesDetokenizer, MosesPunctNormalizer
from tokenizers import Tokenizer

from model import TranslationModel

# it"s a surprise tool that will help you later
detok = MosesDetokenizer(lang="en")
mpn = MosesPunctNormalizer()


def get_attn_mask(L):
    inf_mask = torch.triu(torch.ones(L, L, dtype=bool), diagonal=1)
    res = torch.zeros(L, L).masked_fill(inf_mask, -float("inf"))
    return res


def _greedy_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with greedy search.
    The decoding procedure terminates once either max_len steps have passed
    or the "end of sequence" token has been reached for all sentences in the batch.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :return: a (batch, time) tensor with predictions
    """
    src.to(device)
    model.to(device)
    model.eval()

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    bs = src.shape[0]

    # indices of sequences in batch, which are not fully translated yet
    not_finished_inds = torch.ones(bs, dtype=torch.bool).to(device)

    src_mask = (src == src_pad_id)

    encoded_src = model.encode(src, src_mask)

    res_tensor = torch.ones(bs, max_len, dtype=torch.long) * pad_id
    res_tensor[:,0] = bos_id * torch.ones(bs)
    # res_tensor[:,-1] = eos_id * torch.ones(bs)

    res_tensor = res_tensor.to(device)

    for i in range(1, max_len):
        tgt_mask = get_attn_mask(i).to(device)
        new_tokens = model.decode_last(
                encoded_src[not_finished_inds],
                res_tensor[not_finished_inds,:i],
                tgt_mask,
                src_mask[not_finished_inds]
        ).argmax(dim=-1)

        # update indices of not finished

        end_mask = new_tokens != eos_id
        # print(end_mask)
        res_tensor[not_finished_inds,i] = new_tokens
        not_finished_inds[not_finished_inds.clone()] = end_mask
        # print(not_finished_inds)
        # print('-'*50)

        if sum(not_finished_inds).item() == 0:
            # print(i)
            break 
    
    return res_tensor


def _beam_search_decode(
    model: TranslationModel,
    src: torch.Tensor,
    max_len: int,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    beam_size: int,
) -> torch.Tensor:
    """
    Given a batch of source sequences, predict its translations with beam search.
    The decoding procedure terminates once max_len steps have passed.
    :param model: the model to use for translation
    :param src: a (batch, time) tensor of source sentence tokens
    :param max_len: the maximum length of predictions
    :param tgt_tokenizer: target language tokenizer
    :param device: device that the model runs on
    :param beam_size: the number of hypotheses
    :return: a (batch, time) tensor with predictions
    """
    src.to(device)
    model.to(device)
    model.eval()

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    bos_id = tgt_tokenizer.token_to_id("[BOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    bs = src.shape[0]
    # indices of sequences in batch, which are not fully translated yet
    not_finished_inds = torch.ones(bs, beam_size, dtype=torch.bool)

    src_mask = (src == src_pad_id)

    encoded_src = model.encode(src, src_mask)

    res_tensor = torch.ones(bs*beam_size, max_len, dtype=torch.long) * pad_id
    res_tensor[:,0] = bos_id * torch.ones(bs)
    res_tensor[:,-1] = eos_id * torch.ones(bs)
    beam_probs = torch.zeros(bs, beam_size)  # log probabilities of beams

    # repeat source encoding beam_size times
    encoded_src = encoded_src.unsqueeze(dim=1).expand(res_tensor.shape).view(bs*beam_size, -1)

    res_tensor = res_tensor.to(device)
    beam_probs = beam_probs.to(device)

    for i in range(1, max_len):
        tgt_mask = get_attn_mask(i).to(device)

        # logits of model
        model_out = model.decode_last(encoded_src[not_finished_inds], res_tensor[not_finished_inds,:i], tgt_mask)

        # logits, tokens: torch.Tensor not_finished_inds.sum() x beam_size (proposed continuations)
        logits, tokens = model_out.topk(k=beam_size, dim=-1)
        probs = beam_probs[not_finished_inds] + torch.log_softmax(logits)
        
        not_finished_proposals = res_tensor[not_finished_inds, :i] + tokens.view(1, -1)

        # add finished sequences (for each add beam_size-1 zero probability sequencies for padding)
        proposals = torch.ones(bs*beam_size, beam_size, i+1, dtype=torch.long) * eos_id
        proposals[not_finished_inds]


        # probs = probs.view(bs, beam_size, beam_size)
        # tokens = tokens.view(bs, beam_size, beam_size)

        # each beam has beam_size possible continuations
        proposals = res_tensor[not_finished_inds,:i]

        # update indices of not finished

        end_mask = new_tokens != eos_id
        res_tensor[not_finished_inds,i] = new_tokens
        not_finished_inds[not_finished_inds.clone()] = end_mask.cpu()

        if sum(not_finished_inds).item() == 0:
            break 
    
    return res_tensor


@torch.inference_mode()
def translate(
    model: torch.nn.Module,
    src_sentences: list[str],
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    translation_mode: str,
    device: torch.device,
) -> list[str]:
    """
    Given a list of sentences, generate their translations.
    :param model: the model to use for translation
    :param src_sentences: untokenized source sentences
    :param src_tokenizer: source language tokenizer
    :param tgt_tokenizer: target language tokenizer
    :param translation_mode: either "greedy", "beam" or anything more advanced
    :param device: device that the model runs on
    """

    # encoding and padding ====================================================

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    src_tensor_list = []
    for sentence in src_sentences:
        src_tensor_list.append(torch.asarray(src_tokenizer.encode(sentence).ids))

    src_tensor = torch.nn.utils.rnn.pad_sequence(
        sequences=src_tensor_list,
        batch_first=True,
        padding_value=src_pad_id
    ).long().to(device)

    max_len = src_tensor.shape[1] + 15

    # running translation =====================================================

    if translation_mode  == "greedy":
        res = _greedy_decode(model, src_tensor, max_len, src_tokenizer, tgt_tokenizer, device)
    elif translation_mode == "beam":
        res = _beam_search_decode(model, src_tensor, max_len, tgt_tokenizer, device)
    else:
        raise NotImplementedError()
    
    # decoding translation ====================================================
    
    decoded_res = tgt_tokenizer.decode_batch(res.tolist())
    # print(*decoded_res, sep='\n')

    normalized_res = []
    for line in decoded_res:
        normalized_res.append(detok.detokenize(line.split()).replace(" '", "'").replace("' ", "'"))

    return normalized_res