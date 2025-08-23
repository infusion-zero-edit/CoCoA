import argparse
import logging
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import math
import datasets
import re
import math
import torch
import torch.nn.functional as F
from torch.distributed import is_initialized, all_reduce
import transformers
import accelerate
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    BitsAndBytesConfig,
)

import numpy as np
from termcolor import colored
import json
from accelerate import InitProcessGroupKwargs
import datetime
import torch.nn.functional as F
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
# Disable all log output for the transformers module.
transformers_logger = logging.getLogger("transformers")
transformers_logger.handlers = []
transformers_logger.propagate = False
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

hf_token = os.getenv('HF_TOKEN', None)


def filter_logits_top_p(logits, top_p, negative_multiplier=False):
    assert len(logits.size()) == 3

    # get top-p indices
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    if negative_multiplier:
        filtered_logits = logits.masked_fill(valid_indices == 0, 1000)
    else:
        filtered_logits = logits.masked_fill(valid_indices == 0, -1000)
    return filtered_logits


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3

    # 1) Prevent exact zeros in softmax (avoids log(0) in downstream ops)
    eps = torch.finfo(logits.dtype).tiny
    probs = torch.nn.functional.softmax(logits, dim=-1)
    probs = torch.clamp(probs, min=eps, max=1.0)

    # 2) Compute nucleus mask
    sorted_probs, indices   = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs           = torch.cumsum(sorted_probs, dim=-1)
    nucleus                 = cum_sum_probs < top_p
    nucleus                 = torch.cat(
                                [nucleus.new_ones(nucleus.shape[:-1] + (1,)), 
                                 nucleus[..., :-1]],
                                dim=-1
                              )
    # scatter back to original positions
    valid_indices = torch.zeros_like(nucleus)
    valid_indices.scatter_(2, indices, nucleus)

    # 3) Fallback: if for any sample *no* token is selected, force top‑1 in
    sum_valid = valid_indices.sum(dim=-1, keepdim=True)  # [B,1,1]
    no_valid  = sum_valid == 0
    if no_valid.any():
        # pick the highest‑prob token (indices[...,0]) as a fallback
        top1     = indices[..., :1]  # [B,1,1]
        fallback = torch.nn.functional.one_hot(top1.squeeze(-1),
                                               logits.size(-1)).bool().unsqueeze(1)
        valid_indices = torch.where(no_valid, fallback, valid_indices)

    # 4) Mask and *sanitize* the logits
    filtered_logits = logits.masked_fill(valid_indices == 0, 
                                         torch.finfo(logits.dtype).min)
    filtered_logits = torch.nan_to_num(
        filtered_logits,
        nan=torch.finfo(filtered_logits.dtype).min,
        posinf=torch.finfo(filtered_logits.dtype).max,
        neginf=torch.finfo(filtered_logits.dtype).min
    )

    # 5) Sample
    m        = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(
               selected, logits.size(2)
           ) - one_hot_value

def decode(args, batch_input_ids, dec_depth, model, tokenizer):
    import torch
    import torch.nn.functional as F

    batch_size = args.per_device_eval_batch_size
    assert batch_input_ids.size(1) == args.context_size, "Context size mismatch"
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        raise ValueError("context cannot be none")
    history_decode_ids = None
    past_key_values = None

    if args.model_category == 'seq2seq':
        model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
            batch_input_ids[:, :args.context_size].clone(), dict(), None
        )
        history_decode_ids = model._prepare_decoder_input_ids_for_generation(
            batch_input_ids.size(0),
            model_kwargs=model_kwargs,
            device=batch_input_ids.device,
        )
    else:
        model_kwargs = None

    # Precompute context frequency for answer tokens
    B, context_len = unit_context_input_ids.shape
    freq_list = []
    for b in range(B):
        f = torch.bincount(unit_context_input_ids[b], minlength=args.vocab_size).float() / context_len
        freq_list.append(f)
    # Shape: [B, 1, vocab_size]
    context_freq = torch.stack(freq_list, dim=0).unsqueeze(1).to(batch_input_ids.device)
    
    # --- Initialize history counts & length tracker ---
    hist_counts = torch.zeros([B, args.vocab_size], device=batch_input_ids.device)
    gen_len = 0
    
    for _i in range(dec_depth):
        if args.model_category == 'causal':
            model_inputs = model.prepare_inputs_for_generation(unit_context_input_ids, past_key_values=past_key_values)
            outputs = model(**model_inputs, output_hidden_states=False)
        elif args.model_category == 'seq2seq':
            model_inputs = model.prepare_inputs_for_generation(history_decode_ids, **model_kwargs)
            outputs = model(**model_inputs, output_hidden_states=False)
        else:
            raise ValueError("model category not supported")

        score = outputs.logits[:, -1:, :].clone().contiguous()

        # Dual streams: with-context (score1) and without-context (score2)
        if args.assigned_weight > 0:
            score1 = score.clone().to(args.accelerator.device)
            score2 = torch.zeros(score.shape).to(args.accelerator.device)
            score = filter_logits_top_p(score, top_p=args.filter_top_p)
        else:
            score1 = torch.zeros(score.shape).to(args.accelerator.device)
            score2 = score.clone().to(args.accelerator.device)
            score = filter_logits_top_p(score, top_p=args.filter_top_p_prior, negative_multiplier=True)

        torch.distributed.all_reduce(score1, group=args.gathering_group)
        torch.distributed.all_reduce(score2, group=args.gathering_group)

        # 3) To probabilities & logs
        p_ctx     = F.softmax(score1, dim=-1)
        p_pri     = F.softmax(score2, dim=-1)
        log_p_ctx = F.log_softmax(score1, dim=-1)
        log_p_pri = F.log_softmax(score2, dim=-1)

        # 4) Entropy gap ΔH
        H_ctx   = -torch.sum(p_ctx * log_p_ctx, dim=-1, keepdim=True)
        H_pri   = -torch.sum(p_pri * log_p_pri, dim=-1, keepdim=True)
        Delta_H = H_pri - H_ctx  # [B,1,1]

        # 5) Renyi divergence D2 for extra signal
        inner      = torch.pow(p_ctx, -1.0)
        D2         = 1.0 - torch.sum(p_pri * inner, dim=-1, keepdim=True)
        D2_norm    = torch.clamp(D2 / (1.0 + D2), 0.0, 1.0)

        # 6) Global blend α from ΔH + D2
        gamma        = getattr(args, 'dsab_gamma', 1.0)
        raw_a        = torch.sigmoid(gamma * Delta_H + gamma * D2_norm)
        global_alpha = torch.clamp(raw_a, 0.1, 0.9)
        global_alpha = 0.5
        # 7) Token-wise contrast Δ and robust z
        Delta_med    = torch.median(log_p_ctx - log_p_pri, dim=-1, keepdim=True)[0]
        Delta_mad    = torch.median(torch.abs((log_p_ctx - log_p_pri) - Delta_med), dim=-1, keepdim=True)[0] + 1e-8
        Delta        = log_p_ctx - log_p_pri
        z            = torch.clamp((Delta - Delta_med) / Delta_mad, -5.0, 5.0)
        local_gate   = torch.sigmoid(getattr(args, 'dsab_local_gamma', 1.0) * z)

        # 8) Optional can comment: Adaptive τ from ΔH + Rényi boost
        tau0         = getattr(args, 'dsab_tau0', 1.0)
        kappa        = getattr(args, 'dsab_kappa', 1.0)
        tau_t_base   = tau0 * torch.exp(-kappa * Delta_H)
        inner_R      = torch.sqrt(p_ctx * p_pri)
        D_R          = -2.0 * torch.log(torch.sum(inner_R, dim=-1, keepdim=True) + 1e-12)
        D_R_norm     = torch.clamp(D_R / (1.0 + D_R), 0.0, 1.0)
        kappa_R      = getattr(args, 'dsab_kappa_R', 1.0)
        tau_t        = tau_t_base * (1.0 + kappa_R * D_R_norm)
        tau_t        = torch.clamp(tau_t, 0.1, 5.0)

        # 9) Context freq penalty
        freq_pen     = torch.clamp(1 - getattr(args, 'dsab_beta_freq', 0.5) * context_freq, 0.5, 1.0)

        # 10) History frequency penalty
        hist_freq    = (hist_counts / (gen_len + 1e-8)).unsqueeze(1)
        hist_pen     = torch.clamp(1 - getattr(args, 'dsab_beta_hist', 0.5) * hist_freq, 0.5, 1.0)
        # 11) Mix logits
        S_mix = (
            global_alpha * log_p_ctx
            + (1 - global_alpha) * log_p_pri + 1*Delta
            # + tau_t * local_gate * Delta * freq_pen
        )
        S_mix = S_mix + torch.log(hist_pen)

        # ==== Entropy Gap Δ-booster ====
        H2       = 1.0 - torch.sum(F.softmax(S_mix, dim=-1) ** 2, dim=-1, keepdim=True)
        H2_norm  = torch.clamp(H2 / (1.0 + H2), 0.0, 1.0)
        kappa_H2 = getattr(args, 'dsab_kappa_H2', 0.5)
        S_mix    = S_mix + kappa_H2 * H2_norm * Delta

        # ==== **Margin-z-score peak amplifier [NEW]** ====
        # even when Δ≈0, this boosts whichever logits are already slightly higher
        S_med    = torch.median(S_mix, dim=-1, keepdim=True)[0]
        S_mad    = torch.median(torch.abs(S_mix - S_med), dim=-1, keepdim=True)[0] + 1e-8
        S_z      = torch.clamp((S_mix - S_med) / S_mad, -100.0, 100.0)
        lambda_pm = getattr(args, 'dsab_lambda_pm', 100)
        S_mix    = S_mix + lambda_pm * S_z

        # ==== Optional: Safeguards (uniform & L2 & floor) ====
        V          = args.vocab_size
        eps_t      = torch.clamp(
            getattr(args, 'dsab_eps0', 0.05) * torch.exp(-getattr(args, 'dsab_kappa_s', 1.0) * Delta_H),
            0.0, getattr(args, 'dsab_eps_max', 0.2)
        )
        lambda2    = getattr(args, 'dsab_lambda2', 0.1)
        p_mix      = F.softmax(S_mix, dim=-1)
        p_u        = torch.full_like(p_mix, 1.0 / V)
        p_blend    = (1 - eps_t) * p_mix + eps_t * p_u
        denom      = 1.0 + 2.0 * lambda2
        S_final    = torch.log(p_blend + 1e-12) / denom
        S_floor    = torch.log(torch.full_like(S_final, getattr(args, 'dsab_kappa_min', 1e-4)))
        S_reg      = torch.max(S_final, S_floor)

        # ==== Minimum-Length EOS lock ====
        min_len    = getattr(args, 'dsab_min_len', 5)
        if gen_len < min_len:
            eos = model.generation_config.eos_token_id
            if isinstance(eos, list):
                for e in eos:
                    S_reg[..., 0, e] = -1e9
            else:
                S_reg[..., 0, eos] = -1e9
                
        # Distributed reduction.
        torch.distributed.all_reduce(S_reg, group=args.gathering_group)

        projected_logits = logits_sampling_projection(S_reg, top_p=args.projection_top_p, one_hot_value=args.one_hot_value)
        if not args.accelerator.is_main_process:
            projected_logits = torch.zeros_like(projected_logits)
        torch.distributed.all_reduce(projected_logits, group=args.gathering_group)

        simplex = F.softmax(projected_logits, dim=-1)
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)

        if args.model_category == 'causal':
            unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

        if args.model_category == 'causal':
            past_key_values = outputs.past_key_values
        elif args.model_category == 'seq2seq':
            model_kwargs["past_key_values"] = outputs.past_key_values

        # Stop condition: exit on EOS token.
        assert real_token_ids_list.size(0) == 1 and real_token_ids_list.size(1) == 1
        if isinstance(model.generation_config.eos_token_id, list):
            if real_token_ids_list[0, -1].item() in model.generation_config.eos_token_id:
                break
        elif real_token_ids_list[0, -1].item() == model.generation_config.eos_token_id:
            break

    init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'),
                                               skip_special_tokens=True)
    logger.info(f"sampled: {colored(str(sampled_sequences), 'red')}")
    return history_decode_ids, init_context_input_ids, None, sampled_sequences, context_sequences, None

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument("--init_blank_language_model", action="store_true", help="Whether or not to use a completely blank LM.")
    parser.add_argument(
        "--file_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--train_mode", type=str, default="", help="",
    )
    parser.add_argument(
        "--decode_truncate_len", type=int, default=50, help="",
    ) # how many to cut from right
    parser.add_argument(
        "--decode_depth", type=int, default=2, help="",
    )
    parser.add_argument(
        "--projection_top_p", type=float, default=0.2, help="",
    )
    parser.add_argument(
        "--filter_top_p", type=float, default=1.0, help="",
    )
    parser.add_argument(
        "--filter_top_p_prior", type=float, default=1.0, help="",
    )
    parser.add_argument("--big_model_inference", type=str, default="no")
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--int4", type=str, default="no", help="If ture, will use int4 quantization.")
    parser.add_argument("--local_dir", type=str, default="models/", help="Local directory for model weights.")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=259200))])
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        # set_seed(args.seed)
        accelerate.utils.set_seed(args.seed, device_specific=True) # differ slightly for each device

    if accelerator.is_main_process:
        pass
        # if args.output_dir is not None:
        #     os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.train_mode == "decode":
        if len(args.model_name_or_path.split('|')) > 0:
            main_model_name = args.model_name_or_path.split('|')[0]
            fallback_model_name = args.model_name_or_path.split('|')[1]
            args.model_name_or_path = main_model_name
            args.orig_model_name_or_path = fallback_model_name
        else:
            args.orig_model_name_or_path = args.model_name_or_path
    else:
        raise ValueError("training should be in a separate file (irrelevant in context-aware decoding)")

    # Han: assign ensemble models
    args.file_mode = args.file_mode.split('|')
    assert args.file_mode[0] == "fin"
    assert os.path.exists(args.file_mode[1])
    fin_path = args.file_mode[1]
    fin_data = []
    with open(fin_path, 'r', encoding='utf-8') as f:
        for line in f:
            proc_line = line.strip()
            if proc_line:
                fin_data.append(json.loads(proc_line))
    rank2model = dict()
    for _fd in fin_data:
        if _fd['assigned_process'] in rank2model: # sanity check
            assert ' '.join(rank2model[_fd['assigned_process']]) == ' '.join(_fd['assigned_model'].split('|'))
        else:
            rank2model[_fd['assigned_process']] = _fd['assigned_model'].split('|') 

    # Han: add gathering group
    default_backend = torch.distributed.get_backend(torch.distributed.distributed_c10d._get_default_group())
    args.gathering_group = torch.distributed.new_group(ranks=list(sorted(rank2model.keys())), backend=default_backend)

    if accelerator.process_index not in rank2model.keys(): # Han: exit if not in the ensemble
        return
    args.model_name_or_path = rank2model[accelerator.process_index][0]

    print(args.model_name_or_path)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        if 'llama' in args.model_name_or_path.lower():
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(args.model_name_or_path, token=hf_token,)
        elif 'mistral' in args.model_name_or_path.lower():
            config = AutoConfig.from_pretrained(args.model_name_or_path, token=hf_token,)
        else:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if 'neox' in args.model_name_or_path.lower(): # Han: gpt-neox doesn't have a slow tokenizer, use GPTNeoXTokenizerFast
        from transformers import GPTNeoXTokenizerFast
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.model_name_or_path)
    elif 'llama' in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.local_dir, token=hf_token,)
    elif 'mistral' in args.model_name_or_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.local_dir, token=hf_token,)
    else:
        assert args.use_slow_tokenizer == True 
        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

    if args.init_blank_language_model:
        raise ValueError("disabled")
        model = AutoModelForMaskedLM.from_config(config)
    elif args.model_name_or_path:
        if 't5' in args.model_name_or_path.lower() or 'tk' in args.model_name_or_path.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=False,
                torch_dtype=torch.float16,
            )
            args.model_category = 'seq2seq'
            model = model.to(accelerator.device)
        else:
            if 'llama' in args.model_name_or_path.lower(): # llama special case
                from transformers import LlamaForCausalLM
                if args.big_model_inference == 'no':
                    if args.int4 == 'yes':
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                        model = LlamaForCausalLM.from_pretrained(
                            args.model_name_or_path,
                            torch_dtype=torch.float16,
                            quantization_config=bnb_config,
                            cache_dir=args.local_dir,
                            token=hf_token,
                        )
                    else:
                        model = LlamaForCausalLM.from_pretrained(
                            args.model_name_or_path,
                            torch_dtype=torch.float16,
                            cache_dir=args.local_dir,
                            token=hf_token,
                        )
                    model = model.to(accelerator.device)
                else:
                    # Han: we assume 8 GPUs
                    if accelerator.process_index == 0:
                        # local_devices = [0, 2, 4, 6]
                        local_devices = [i*2 for i in range(int(args.num_gpus / 2))]
                    elif accelerator.process_index == 1:
                        # local_devices = [1, 3, 5, 7]
                        local_devices = [i*2+1 for i in range(int(args.num_gpus / 2))]
                    else:
                        raise ValueError("check accelerator.process_index")
                    # this is architecture specific
                    my_device_map = {'model.embed_tokens': local_devices[0],
                                    'lm_head': local_devices[0],
                                    'model.norm': local_devices[0]}
                    for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                        for layer_idx in layer_idx_list:
                            my_device_map[f'model.layers.{layer_idx}'] = local_devices[_device_i]
                    if args.int4 == 'yes':
                        # bnb_config = BitsAndBytesConfig(
                        #     load_in_4bit=True,
                        #     bnb_4bit_quant_type="nf4",
                        #     bnb_4bit_compute_dtype=torch.float16,
                        #     bnb_4bit_use_double_quant=True,
                        # )
                        bnb_config = BitsAndBytesConfig(
                           load_in_8bit=True
                        )
                        model = LlamaForCausalLM.from_pretrained(
                            args.model_name_or_path,
                            device_map=my_device_map,
                            torch_dtype=torch.float16,
                            quantization_config=bnb_config,
                            cache_dir=args.local_dir,
                            token=hf_token,
                        )
                    else:
                        model = LlamaForCausalLM.from_pretrained(
                            args.model_name_or_path,
                            device_map=my_device_map,
                            torch_dtype=torch.float16,
                            cache_dir=args.local_dir,
                            token=hf_token,
                        )
            elif 'mistral' in args.model_name_or_path.lower():
                model = AutoModelForCausalLM.from_pretrained(
                            args.model_name_or_path,
                            torch_dtype=torch.float16,
                            cache_dir=args.local_dir,
                            token=hf_token,
                        )
                model = model.to(accelerator.device)
            elif args.big_model_inference == 'no':
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    torch_dtype=torch.float16, 
                )
                model = model.to(accelerator.device)
            elif args.big_model_inference == 'yes' and 'opt' in args.model_name_or_path.lower():
                if accelerator.process_index == 0:
                    local_devices = [i*2 for i in range(int(args.num_gpus / 2))]
                elif accelerator.process_index == 1:
                    local_devices = [i*2+1 for i in range(int(args.num_gpus / 2))]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'model.decoder.embed_tokens': local_devices[0],
                                'lm_head': local_devices[0],
                                'model.decoder.embed_positions': local_devices[0],
                                'model.decoder.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'model.decoder.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neox' in args.model_name_or_path.lower():
                if accelerator.process_index == 0:
                    local_devices = [i*2 for i in range(int(args.num_gpus / 2))]
                elif accelerator.process_index == 1:
                    local_devices = [i*2+1 for i in range(int(args.num_gpus / 2))]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'gpt_neox.embed_in': local_devices[0],
                                'embed_out': local_devices[0],
                                'gpt_neox.final_layer_norm': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'gpt_neox.layers.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            elif args.big_model_inference == 'yes' and 'neo' in args.model_name_or_path.lower():
                if accelerator.process_index == 0:
                    local_devices = [i*2 for i in range(int(args.num_gpus / 2))]
                elif accelerator.process_index == 1:
                    local_devices = [i*2+1 for i in range(int(args.num_gpus / 2))]
                else:
                    raise ValueError("check accelerator.process_index")
                # this is architecture specific
                my_device_map = {'transformer.wte': local_devices[0],
                                'lm_head': local_devices[0],
                                'transformer.wpe': local_devices[0],
                                'transformer.drop': local_devices[0],
                                'transformer.ln_f': local_devices[0]}
                for _device_i, layer_idx_list in enumerate(np.array_split(np.arange(config.num_hidden_layers), len(local_devices))):
                    for layer_idx in layer_idx_list:
                        my_device_map[f'transformer.h.{layer_idx}'] = local_devices[_device_i]
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    ignore_mismatched_sizes=False,
                    device_map=my_device_map,
                    torch_dtype=torch.float16,
                )
            else:
                raise ValueError("check args.big_model_inference")

            args.model_category = 'causal'
        model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward) # referred to https://github.com/huggingface/accelerate/blob/38fd30e764ea87ef86e7d69fcba559c3605925b1/src/accelerate/accelerator.py#L1138
        model.forward = accelerate.utils.convert_outputs_to_fp32(model.forward)
    else:
        raise ValueError("specify --init_blank_language_model")

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    logger.info(f"model size: {sum(p.numel() for p in model.parameters())}")
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)
    one_hot_value = 5.0 # unused
    # Load the external critic model (Sentence Transformer) for re-ranking.
    # critic_model = SentenceTransformer('all-MiniLM-L6-v2', device=accelerator.device)
    ##########################################

    # change the output file name later
    out_json_fn = f"{fin_path}_cagd.output_topp{args.projection_top_p}_genlen{args.decode_depth}.jsonl"
    if accelerator.is_main_process:
        with open(out_json_fn, 'w') as f:
            f.write('placeholder, program not finished ...\n')

    args.tokenizer = tokenizer

    if args.train_mode == "decode":
        model.eval()

        args.one_hot_value = one_hot_value
        args.vocab_size = vocab_size
        args.hidden_size = hidden_size
        args.accelerator = accelerator

        export_list = []
        args.orig_decode_truncate_len = args.decode_truncate_len
        with torch.no_grad():
            for _fd in fin_data: # only support batch size 1 for now since the context size can be different across lines
                if _fd['assigned_process'] != args.accelerator.process_index: # remember to unblock barriers before this line
                    continue
                args.assigned_weight = _fd.get('assigned_weight', 1.0)

                ctx_field_name = 'context_string'
                assert ctx_field_name in _fd
                assert args.per_device_eval_batch_size == 1

                input_ids = torch.LongTensor(tokenizer.encode(_fd[ctx_field_name], add_special_tokens=True)).unsqueeze(0).to(args.accelerator.device)
                args.context_size = input_ids.size(1)
                args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size # Han: this compensates for the unknown input context size

                

                if 'filter_p' in _fd: # token filtering
                    args.filter_top_p = _fd['filter_p']
                if 'filter_p_prior' in _fd:
                    args.filter_top_p_prior = _fd['filter_p_prior']

                if args.decode_truncate_len < 0:
                    continue # skipping very long examples
                logger.info(f"idx: {_fd['input_index']}")

                repeat_sample = 1 # change here manually if necessary
                for _r in range(repeat_sample):
                    history_decode_ids, _, _, sampled_sequences, _, _ = \
                        decode(args, input_ids, args.decode_depth, model, tokenizer)
                    if _r == 0: # first sample
                        # export to jsonl
                        for _i in range(args.per_device_eval_batch_size):
                            export_dict = dict()
                            export_dict['tokens'] = [history_decode_ids.tolist()[_i]]
                            export_dict['string'] = [sampled_sequences[_i]]
                            export_dict['assigned_process'] = _fd['assigned_process']
                            export_dict['assigned_model'] = args.model_name_or_path
                            export_dict['output_index'] = len(export_list)
                            export_dict['input_index'] = _fd['input_index']
                            export_list.append(export_dict)
                    else:
                        for _i in range(args.per_device_eval_batch_size):
                            export_list[-(args.per_device_eval_batch_size - _i)]['tokens'].append(history_decode_ids.tolist()[_i])
                            export_list[-(args.per_device_eval_batch_size - _i)]['string'].append(sampled_sequences[_i])

        if accelerator.is_main_process:
            if os.path.exists(out_json_fn):
                os.remove(out_json_fn)
                logger.info(f"Cleaning existing {out_json_fn}")
            with open(out_json_fn, mode="w") as f_out: # use mode 'a' if several processes are writing to the same file
                for export in export_list:
                    f_out.write(json.dumps(export))
                    f_out.write("\n")


if __name__ == "__main__":
    main()