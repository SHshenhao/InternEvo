# Copyright (c) InternLM. All rights reserved.
import os
import re

import torch

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import partition_uniform
from internlm.utils.logger import get_logger
from internlm.utils.storage_manager import get_fns, llm_load
from internlm.utils.utils import ModelType
from transformers import AutoModelForCausalLM

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def load_llama_pretrained_weights(folder, model):
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = []
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(os.path.join(folder, fn))

    if len(model_fns) == 0:
        model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".pth") or fn.endswith(".pt")]

    if len(model_fns) == 0:
        raise FileNotFoundError(f"No checkpoint file found in {folder}")

    model_fns.sort()

    old_tp = len(model_fns)
    cur_tp = gpc.get_world_size(ParallelMode.TENSOR)
    # If the two tp are inconsistent, you need to consider the merge before splitting
    if old_tp != cur_tp:
        raise RuntimeError(
            f"Your current tp is `{cur_tp}`, but the tp in folder:`{folder}` is `{old_tp}`, use `` to convert first"
        )

    states = llm_load(model_fns[gpc.get_local_rank(ParallelMode.TENSOR)], map_location="cpu")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        for name in list(states.keys()):
            if f".{i}." in name:
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys:
        current_states["tok_embeddings.weight"] = states["tok_embeddings.weight"]
        assert model.first_layer == 0, f"Expect model.NaiveAMPModel to be 0, but got {model.first_layer}"
    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["norm.weight"]
        current_states["output.weight"] = states["output.weight"]
    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )

    del states
    del current_states
    internlm_accelerator.empty_cache()


def load_hf_llama_pretrained_weights(folder, model):
    """NOTE: when loading huggingface's llama pretrained weights, you should set `adapt_hf=True` in your config."""
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".bin") and fn.startswith("pytorch_model")]
    model_fns.sort()

    states = {}

    for model_fn in model_fns:
        states.update(llm_load(model_fn, map_location="cpu"))

    deep_split = getattr(model, "deep_split", False)
    if deep_split:
        print("using deep split when loading pretrained weights!")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        if gpc.config.model_type == ModelType.LLAMA2.name:
            if deep_split:
                layer_ids = i // 2
            else:
                layer_ids = i

            if not deep_split or (i + 2) % 2 == 0:
                states[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.q_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.k_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.v_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.o_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.input_layernorm.weight"
                )

            if not deep_split or (i + 2) % 2 == 1:
                states[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.gate_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.up_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.down_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]

                states[f"layers.{i}.ffn_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.post_attention_layernorm.weight"
                )

            if f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq" in states:
                states.pop(f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq")

        for name in list(states.keys()):
            if name.startswith(f"layers.{i}"):
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys or "tok_embeddings.word_embeddings.weight" in model_state_keys:
        if gpc.config.model.get("embed_split_hidden", True):
            current_states["tok_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        else:
            current_states["tok_embeddings.word_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        assert model.first_layer == 0, f"Expect model.first_layer to be 0, but got {model.first_layer}"

    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["model.norm.weight"]
        current_states["output.weight"] = torch.chunk(
            states["lm_head.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=0
        )[gpc.get_local_rank(ParallelMode.TENSOR)]

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )
    internlm_accelerator.empty_cache()


def load_internlm_with_dynamic_parallel_size(folder, model):

    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = []
    for fn in fns:
        # filter with `_t` is for avoiding conflict with model_config.py
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    old_tp, old_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        old_tp = max(old_tp, int(tp[2:]) + 1)
        old_pp = max(old_pp, int(pp[2:]) + 1)

    assert old_tp > 0 and old_pp > 0, f"ckpt with tp:{old_tp} and pp:{old_pp} is illegal"

    tp = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    assert old_tp % tp == 0 or tp % old_tp == 0, (
        f"Expected TP size in loaded checkpoint to be fit with TP size in current config, but got {old_tp} in "
        f"checkpoint and {tp} in current config"
    )

    correspond_tps = []

    if old_tp <= tp:
        correspond_tps.append(tp_rank // (tp // old_tp))
        ratio = tp // old_tp
        rank = tp_rank % ratio
    else:
        for i in range(old_tp // tp):
            correspond_tps.append(tp_rank * (old_tp // tp) + i)
        rank = 0
        ratio = 1

    current_states = {}

    pp = gpc.get_world_size(ParallelMode.PIPELINE)

    assert gpc.config.model.num_chunks == 1, "May cause future collisions, ignore this if necessary"

    old_pp_partition = partition_uniform(gpc.config.model.num_layers, old_pp, 1)

    for idx, parts in enumerate(old_pp_partition):
        start, end = parts[0]
        if model.last_layer <= start or model.first_layer >= end:
            continue

        tmp_states = {}

        for correspond_tp in correspond_tps:
            model_name = f"model_tp{correspond_tp}_pp{idx}.pt"
            states = llm_load(os.path.join(folder, model_name), map_location="cpu")
            for i in range(start, end):
                if i >= model.last_layer:
                    break
                if i < model.first_layer:
                    continue
                for name in list(states.keys()):
                    if f".{i-start}." in name:
                        to_name = name.replace(f".{i-start}.", f".{i-model.first_layer}.")
                        if "norm" in name:
                            tmp_states[to_name] = [states.pop(name)]
                        elif any(x in name for x in ("out_proj", "w2")):
                            if "bias" not in name:
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=-1)[rank])
                            else:
                                tmp_states[to_name] = [states.pop(name)]
                        elif any(x in name for x in ("w1", "w3")):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                        elif any(x in name for x in ("Wqkv",)):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            _wqkv = states.pop(name).chunk(3, dim=0)
                            _wq_splits = _wqkv[0].chunk(ratio, dim=0)
                            _wk_splits = _wqkv[1].chunk(ratio, dim=0)
                            _wv_splits = _wqkv[2].chunk(ratio, dim=0)
                            new_wqkv = torch.concat([_wq_splits[rank], _wk_splits[rank], _wv_splits[rank]], dim=0)
                            tmp_states[to_name].append(new_wqkv)
                        else:
                            raise KeyError(f"Unknown key {name}.")

            if "embedding.weight" in states and model.first_layer == 0:
                tmp_states["embedding.weight"] = tmp_states.get("embedding.weight", [])
                tmp_states["embedding.weight"].append(states["embedding.weight"].chunk(ratio, dim=1)[rank])
            if "head.weight" in states and model.last_layer == gpc.config.model.num_layers:
                tmp_states["norm.weight"] = [states["norm.weight"]]
                tmp_states["head.weight"] = tmp_states.get("head.weight", [])
                tmp_states["head.weight"].append(states["head.weight"].chunk(ratio, dim=0)[rank])

            states = {}

        for name in list(tmp_states.keys()):
            data = tmp_states.pop(name)
            if len(data) == 1:
                current_states[name] = data[0]
            else:
                current_states[name] = torch.concat(
                    data, dim=1 if name == "embedding.weight" or any(x in name for x in ("out_proj", "w2")) else 0
                )

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )


def load_hf_model_pretrained_weights(folder, model):
    """NOTE: when loading huggingface's model pretrained weights, you should set `adapt_hf=True` in your config."""
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    pretrained_model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
    model.load_state_dict(pretrained_model.state_dict(), strict=False)

    if gpc.is_rank_for_log():
        logger.info("Pretrained weights loaded successfully")


from safetensors import safe_open


def load_pp_hf_ckpt(path, prefix: str = None, suffix: str = None):
    assert path is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {path}")
    fns = get_fns(path)
    model_fns, ckpt_type = [], None
    if prefix or suffix:
        prefix = "" if prefix is None else prefix
        suffix = "" if suffix is None else suffix
        for fn in fns:
            if fn.endswith(suffix) and fn.startswith(prefix):
                if not ckpt_type:
                    if fn.endswith(".safetensors"):
                        ckpt_type = "safetensors"
                    else:
                        ckpt_type = "torch"
                model_fns.append(os.path.join(path, fn))
    else:
        for fn in fns:
            if not ckpt_type:
                if fn.endswith(".safetensors") and fn.startswith("model"):
                    ckpt_type = "safetensors"
                elif fn.endswith(".bin") and fn.startswith("pytorch_model"):
                    ckpt_type = "torch"
            if (ckpt_type == "safetensors" and fn.endswith(".safetensors")) or (
                ckpt_type == "torch" and fn.endswith(".bin")
            ):
                model_fns.append(os.path.join(path, fn))
    model_fns.sort()
    states = {}

    for model_fn in model_fns:
        tensors = {}
        with safe_open(model_fn, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        states.update(tensors)
    return ckpt_type == "safetensors" if ckpt_type else False, states


def get_mapping(key, mappings):
    match = []
    for mapping in mappings:
        if isinstance(mapping, tuple) and len(mapping) and re.search(mapping[0], key):
            match.append(mapping)
    # search the ordinal number of the layer
    layer_pattern = re.search(r"\.\b(\d+)\.", key)
    if layer_pattern:
        layer = int(layer_pattern.group(1))
    else:
        layer = None
    return layer, match, "bias" in key


def replace_between_dots(text, old, new):
    pattern = re.compile(r"(?<=\.)" + re.escape(old) + r"(?=\.)")
    return pattern.sub(new, text)


def get_local_splited_weight(
    states,
    mappings,
    pp_layer_range,
    tp_world_size,
    tp_local_rank,
):
    def find_interval_index(number, intervals):
        for chunk_id, (start, end) in enumerate(intervals):
            if start <= number <= end:
                return chunk_id
        return -1

    current_states = {}
    for k, v in states.items():
        # match the pattern in ckpt module name
        layer, matches, bias = get_mapping(k, mappings)
        if matches:
            # import pdb; pdb.set_trace()
            if layer and (chunk_id := find_interval_index(layer, pp_layer_range)) == -1:  # [(0, 14), (14, 28)]
                continue

            for mapping in matches:
                ckpt_name, model_name, chunk_dim, local_rank = mapping
                if local_rank:
                    # replace the pattern in ckpt module name into model module name
                    key = re.sub(ckpt_name, model_name, k).replace("model.", "")
                    if layer:
                        key = replace_between_dots(key, str(layer), str(layer - pp_layer_range[chunk_id][0]))
                    # don't chunk dim 1 row tensor (row vector)
                    if tp_world_size > 1 and (chunk_dim == 0 or (chunk_dim == 1 and v.dim() > 1)):
                        value = torch.chunk(v, tp_world_size, dim=chunk_dim)[tp_local_rank]
                    else:
                        value = v
                    if not bias or tp_local_rank == 0 or chunk_dim != 1:
                        current_states[key] = value
        else:
            print("unknown key: ", k)
    return current_states


def obtain_spliting_parameters():
    num_layers = gpc.config.model.num_layers
    num_chunks = gpc.config.model.num_chunks
    if gpc.is_initialized(ParallelMode.TENSOR):
        tp_world_size = gpc.get_world_size(ParallelMode.TENSOR)
        tp_local_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    else:
        tp_world_size = 1
        tp_local_rank = 0
    if gpc.is_initialized(ParallelMode.PIPELINE):
        pp_world_size = gpc.get_world_size(ParallelMode.PIPELINE)
        pp_local_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    else:
        pp_world_size = 1
        pp_local_rank = 0
    # 0 represents num_chunks=1, currently only support num_chunks=1.
    assert num_chunks == 1, "May cause future collisions, ignore this if necessary"
    pp_layer_range = partition_uniform(num_layers, pp_world_size, num_chunks)
    return (
        num_layers,
        num_chunks,
        tp_world_size,
        tp_local_rank,
        pp_world_size,
        pp_local_rank,
        pp_layer_range[pp_local_rank],
    )


def load_qwen_2_pretrained_weights_dynamic(folder, model, **kwargs):  # pylint: disable=W0613
    # assert gpc.config.model_type == "QWEN", 'Please use model_type="QWEN" to load qwen huggingface checkpoint.'
    # is_st, states = load_ckpt(folder, "gemma", ".ckpt")  # torch checkpoint
    _, states = load_pp_hf_ckpt(folder)  # huggingface checkpoint

    if gpc.config.model.num_layers >= 80 and gpc.is_rank_for_log():
        logger.warning(
            "you are loading a very large huggingface model, it may lead to out of CPU memory，\
You can try to manually let the rank with tp<4 sleep here for 2 minutes."
        )
        import time

        if gpc.get_global_rank() % 8 < 4:
            time.sleep(120)

    model_state_dict = {}
    for key, value in states.items():
        if "transformer.h" in key:
            model_state_dict[key.replace("transformer.h", "layers")] = value
        elif "transformer." in key:
            model_state_dict[key.replace("transformer.", "")] = value
        elif "post_attention_layernorm." in key:
            model_state_dict[key.replace("post_attention_layernorm.", "ln2.")] = value
        elif "input_layernorm." in key:
            model_state_dict[key.replace("input_layernorm.", "ln1.")] = value
        else:
            model_state_dict[key] = value
    del states

    (
        num_layers,
        num_chunks,  # pylint: disable=W0612
        tp_world_size,
        tp_local_rank,
        pp_world_size,
        pp_local_rank,
        pp_layer_range,
    ) = obtain_spliting_parameters()
    first = pp_local_rank == 0
    last = pp_local_rank + 1 == pp_world_size

    mappings = [
        ("self_attn.q_proj", "attention.wq", 0, True),
        ("self_attn.k_proj", "attention.wk", 0, True),
        ("self_attn.v_proj", "attention.wv", 0, True),
        ("self_attn.o_proj", "attention.wo", 1, True),
        ("mlp.up_proj", "feed_forward.w3", 0, True),
        ("mlp.gate_proj", "feed_forward.w1", 0, True),
        ("mlp.down_proj", "feed_forward.w2", 1, True),
        ("ln1", "attention_norm", -1, True),
        ("ln2", "ffn_norm", -1, True),
        ("norm", "norm", -1, last),
        ("embed_tokens", "tok_embeddings", 1, first),
        ("lm_head", "output", 0, last),
    ]

    block = model.layers[0]
    num_kv_heads = block.attention.num_kv_heads
    num_attn_heads = block.attention.num_heads
    head_dim = block.attention.head_dim

    current_states = get_local_splited_weight(model_state_dict, mappings, pp_layer_range, tp_world_size, tp_local_rank)

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{tp_local_rank}, pp:{pp_local_rank}"
        )


LOAD_FUNC_DICT = {
    "llama": load_llama_pretrained_weights,
    "hf_llama": load_hf_llama_pretrained_weights,
    "internlm_test": load_internlm_with_dynamic_parallel_size,
    "hf_model": load_hf_model_pretrained_weights,
    "hf_qwen2": load_qwen_2_pretrained_weights_dynamic,
}
