import os
import pandas as pd
import transformers
import datasets
import torch
import inspect
import einops
import copy
import torch.nn as nn
from typing import *
import optuna
import yaml
import sys
import numpy as np

from torch.utils.data import DataLoader, SequentialSampler
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from peft import get_peft_model, LoraConfig, AutoPeftModel
from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
from transformers import TrainingArguments as TrainingArgumentsBase, TrainerState, TrainerControl
from transformers import IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames, default_logdir
from transformers.utils.generic import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import get_last_checkpoint, has_length
from transformers.activations import ACT2FN
from transformers import Trainer as TrainerBase
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    PreTrainedModel,
    AutoModel,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    default_data_collator,
)

# logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

# parser args
@dataclass
class TrainingArguments(TrainingArgumentsBase):
    # general
    output_dir: str = field(default='output/0')
    tuner_results_dir: str = field(default='./tuner_results/')
    run_results_dir: str = field(default='./run_results/')
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    do_predict: bool = field(default=True)
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be set at the beginning of training."
        }
    )
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='eval_spearmanr')
    include_inputs_for_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the inputs will be passed to the `compute_metrics` function."
        }
    )  # for extra info
    report_to: Union[None, str, List[str]] = field(
        default='tensorboard', metadata={"help": "The list of integrations to report the results and logs to."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    # greater_is_better: Optional[bool] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Whether the `metric_for_best_model` should be maximized or not."
    #             "- `True` if doesn't end in `'loss'`."
    #             "- `False` if ends in `'loss'`."
    #         )
    #     }
    # )
    # debug: Union[str, List[DebugOption]] = field(
    #     default="underflow_overflow",
    #     metadata={
    #         "help": (
    #             "Whether or not to enable debug mode. Current options: "
    #             "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
    #             "`tpu_metrics_debug` (print debug metrics on TPU)."
    #         )
    #     }
    # )

    # dataset
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    num_train_epochs: float = field(default=3)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    dataloader_prefetch_factor: Optional[int] = field(
        default=1,
        metadata={
           "help": "Number of batches loaded in advance by each worker."
        }
    )
    dataloader_num_workers: int = field(
        default=2,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    # learning rate
    optim: Union[OptimizerNames, str] = field(default='adamw_torch', metadata={"help": "The optimizer to use."}, )
    learning_rate: float = field(default=3e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for AdamW if we apply some."})
    lr_scheduler_type: Union[SchedulerType, str] = field(default="linear")
    warmup_ratio: float = field(
        default=0.1,
        metadata={
           "help": "Ratio of total training steps used for a linear warmup from 0 to `learning_rate`."
        }
    )

    # gradient
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(default=1)  # if>1, real save/eval_steps = it * ori_steps

    # eval and save
    eval_strategy: Union[IntervalStrategy, str] = field(default='steps')
    save_strategy: Union[IntervalStrategy, str] = field(default="steps")
    eval_steps: Optional[float] = field(
        default=500,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: Optional[int] = field(default=1)

    # log/progress bar
    log_level: Optional[str] = field(default="info")
    disable_tqdm: Optional[bool] = field(default=False)
    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    # callback
    early_stopping_patience: int = field(
        default=10,
        metadata={
            "help": "Will stop the training when the eval metric stops improving after N evaluations."
        }
    )

    # resuming training
    ignore_data_skip: bool = field(
        default=True,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    restore_callback_states_from_checkpoint: bool = field(
        default=True,
        metadata={
            "help": "Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to the `Trainer` if they exist in the checkpoint."
        }
    )
    
    # tuner setup
    tuner: bool = field(
        default=False,
        metadata={
            'help': "Use optuna tuner or not."
            "default hyperparameter search setup must add tuner_xxx"
        }
    )
    n_trials: int = field(
        default=5,
        metadata={
            'help': "Number of trials to run."
        }
    )

    # run
    run_times: int = field(default=5, metadata={'help': "Run times."})

    # resume
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=512)
    padding: str = field(default='longest', metadata={"help": "The padding strategy to use.(longest/max_length)"})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
        },
    )
    train_file: Optional[str] = field(
        default="data/csts_train.csv",
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default="data/csts_validation.csv",
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default="data/csts_test.csv",
        metadata={"help": "A csv or a json file containing the test data."},
    )

@dataclass
class TokenizerAndModelArguments:
    model_name_or_path: str = field(
        default="princeton-nlp/sup-simcse-roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    cl_temp: float = field(default=1.5, metadata={"help": "Temperature for contrastive loss."})
    pool_type: str = field(
        default='cls',
        metadata={
            'help': "Pool type: Options"
            "1) cls: use [CLS] token."
            "2) avg: mean pooling."
            "3) avg_top2: mean pooling of last 2 layers."
            "4) avg_first_last: average of first and last layers."
        }
    )
    train_with_lora: bool = field(default=True)
    freeze_encoder: Optional[bool] = field(
        default=True, metadata={"help": "Freeze encoder weights."}
    )
    transform: Optional[bool] = field(
        default=True,
        metadata={"help": "Use a linear transformation on the encoder output"},
    )
    # hyperparameter space
    tuner_cl_temp: dict = field(
        default_factory=lambda: {'type':'float', 'low':0.5, 'high':1.5, 'step':0.1},
    )
    tuner_pool_type: dict = field(
        default_factory=lambda: {'type':'categorical', 'choices':['cls', 'avg', 'avg_top2', 'avg_first_last']},
    )

# dataset
def str_if_contain_in_str_list(one_str:str, str_list:Iterable[str], mode:str='contain'):
    """
    check if one_str in str_list
    :param one_str:
    :param str_list:
    :param mode:
    :return:
    """
    for one_str_in_list in str_list:
        if mode=='contain':
            if one_str_in_list in one_str:
                return True
        elif mode=='contained':
            if one_str in one_str_in_list:
                return True
        else:
            raise ValueError(f'mode {mode} not recognized')
    return False

def listdict_map_dictlist(listdict:Optional[List[Dict[str, Any]]]=None,
                          dictlist:Optional[Dict[str, List[Any]]]=None):
    """
    listdict: [{"a":1, "b":2}, {"a":3, "b":4}]
    dictlist: {"a":[1,3], "b":[2,4]}
    :param listdict:
    :param dictlist:
    :return:
    """
    result = None
    if listdict is not None:
        result = {}
        for one_dict in listdict:
            for key, value in one_dict.items():
                if key not in result.keys():
                    result[key] = []
                result[key].append(value)
    elif dictlist is not None:
        result = []
        for key, value_list in dictlist.items():
            for idx, value in enumerate(value_list):
                if idx >= len(result):
                    result.append({})
                result[idx][key] = value
    return result

@dataclass
class DataCollatorWithPaddingForMultiRenameInputs:
    group_feature_names: List[List[str]]
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    model_input_names: tuple[str] = ("input_ids", "token_type_ids", "attention_mask")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_input_keys = features[0].keys()
        features: Dict[str, List[Any]] = listdict_map_dictlist(features,None)
        other_keys_not_in_group_feature_names = []
        for input_key in all_input_keys:
            is_in_group = False
            for one_group_keys in self.group_feature_names:
                if input_key in one_group_keys:
                    is_in_group = True
                    break
            if not is_in_group:
                if str_if_contain_in_str_list(input_key, self.model_input_names, mode='contain'):
                    pass # although not in group_feature_names, we really need to pad them
                else:
                    other_keys_not_in_group_feature_names.append(input_key)

        batch = {}
        for other_key in other_keys_not_in_group_feature_names:
            batch[other_key] = features[other_key]
        for gid, one_group_keys in enumerate(self.group_feature_names):
            #  first, find group_key contain input_ids
            one_group_input_ids = []
            for group_key in one_group_keys:
                if self.model_input_names[0] in group_key:
                    one_group_input_ids.append(group_key)
            count_input_ids = len(one_group_input_ids)

            # second, expand complete group keys
            complete_one_group_keys = one_group_input_ids.copy()
            for input_ids_var in one_group_input_ids:
                for stand_name in self.model_input_names[1:]:
                    complete_one_group_keys.append(input_ids_var.replace(self.model_input_names[0], stand_name))

            # third, intersection really input
            complete_one_group_keys = set(complete_one_group_keys).intersection(all_input_keys)
            complete_one_group_keys = sorted(list(complete_one_group_keys))

            # forth, construct stand_name_list map original_name_list
            original_name_list = complete_one_group_keys.copy()
            stand_name_list = []
            for original_name in original_name_list:
                for stand_name in self.model_input_names:
                    if stand_name in original_name:
                        stand_name_list.append(stand_name)
                        break
            original_to_stand_map = dict(zip(original_name_list, stand_name_list))

            # fifth, concat multiInput
            original_name_features: Dict[str, List[Any]] = {}
            for original_name in original_name_list:
                original_name_features[original_name] = features[original_name]
            group_features: Dict[str, List[Any]] = {key:[] for key in set(stand_name_list)}
            for original_name in original_name_list:
                group_features[original_to_stand_map[original_name]].extend(original_name_features[original_name])

            # sixth, pad
            group_batch = self.tokenizer.pad(group_features,
                                             padding=self.padding,
                                             max_length=self.max_length,
                                             pad_to_multiple_of=self.pad_to_multiple_of,
                                             return_tensors=self.return_tensors)

            # chunk
            #if count_input_ids > 1:
            per_chunk_size = group_batch[self.model_input_names[0]].shape[0] // count_input_ids
            for stand_name in group_batch.keys():
                corr_original_name_list = []
                for original_name in original_name_list:
                    if stand_name in original_name:
                        corr_original_name_list.append(original_name)
                corr_original_name_features_list = []
                for idx in range(count_input_ids): #2  0/1
                    corr_original_name_features_list.append(group_batch[stand_name][idx*per_chunk_size:(idx+1)*per_chunk_size])
                batch.update(dict(zip(corr_original_name_list, corr_original_name_features_list)))
        return BatchEncoding(batch, tensor_type=self.return_tensors)

def unbatch(examples):
    out = {}
    for k,v in examples.items():
        tv = []
        for iv in v:
            tv = tv + iv
        out[k] = tv
    return out

def scale_to_range(labels: List, scale: tuple):
    min_, max_ = scale
    return list(map(lambda x: (x - min_) / (max_ - min_), labels))

def preprocess_func(examples, tokenizer: PreTrainedTokenizerBase,
                    sentence1_key: str, sentence2_key: str, condition_key: str,
                    similarity_key: str, scale: tuple,):
    sent1_args = (examples[sentence1_key], examples[condition_key])
    sent2_args = (examples[sentence2_key], examples[condition_key])
    sent1_res = tokenizer(*sent1_args, truncation=True)
    sent2_res = tokenizer(*sent2_args, truncation=True)
    for idx in [2, ]:
        for key in sent2_res.keys():
            sent1_res[key + '_' + str(idx)] = sent2_res[key]
    sent1_res['labels'] = scale_to_range(examples[similarity_key], scale)
    return sent1_res

def add_prefix(examples, prefix: str, columns: Optional[Union[str, List[str]]] = None):
    if isinstance(columns, str):
        examples[prefix + '_' + columns] = examples[columns]
        return examples
    elif columns is None or columns == 'all':
        columns = list(examples.keys())
    for key in columns:
        examples[prefix + '_' + key] = examples[key]
    return examples

def concat_pos_and_neg_datasets(datasets: Dataset):
    pos = datasets.shard(2, 0).map(add_prefix, batched=True, remove_columns=datasets.column_names,
                                   fn_kwargs={'prefix': 'pos', 'columns': None})
    neg = datasets.shard(2, 1).map(add_prefix, batched=True, remove_columns=datasets.column_names,
                                   fn_kwargs={'prefix': 'neg', 'columns': None})
    new_datasets = concatenate_datasets([pos, neg], axis=1)
    return new_datasets

def load_datasets_func(data_args:DataTrainingArguments,
                       training_args:TrainingArguments,
                       tokenizer:PreTrainedTokenizerBase):
    data_files = {}
    if training_args.do_train:
        data_files['train'] = data_args.train_file
    if training_args.do_eval:
        data_files['validation'] = data_args.validation_file
    if training_args.do_predict:
        data_files['test'] = data_args.test_file

    #datasets.disable_caching()
    datasets.disable_progress_bars()
    if data_args.validation_file.endswith('.csv'):
        data_style = 'csv'
    elif data_args.validation_file.endswith('.json'):
        data_style = 'json'
    else:
        raise ValueError('data style must be csv or json')
    raw_datasets = load_dataset(data_style, data_files=data_files)

    label_unique = raw_datasets.unique('label')
    all_labels = set(label_unique['train'] + label_unique['validation'])
    scale = (min(all_labels), max(all_labels))
    with training_args.main_process_first(desc='dataset map pre-processing'):
        sort_datasets = raw_datasets.sort(['sentence1', 'label'], reverse=[False, True])
        if training_args.do_train:
            train_dataset = sort_datasets.pop('train')
            batch_train_dataset = train_dataset.batch(2).shuffle()
            unbatch_train_dataset = batch_train_dataset.map(unbatch, batched=True)
            sort_datasets['train'] = unbatch_train_dataset
        trans_datasets = sort_datasets.map(
            preprocess_func, 
            batched=True,
            remove_columns=raw_datasets['validation'].column_names,
            fn_kwargs={'tokenizer': tokenizer,
                       'sentence1_key': 'sentence1',
                       'sentence2_key': 'sentence2',
                       'condition_key': 'condition',
                       'similarity_key': 'label',
                       'scale': scale,}
        )

        # select max samples
        select_trans_datasets = {}
        if training_args.do_train:
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(trans_datasets['train']), data_args.max_train_samples)
                select_trans_datasets['train'] = trans_datasets['train'].select(range(max_train_samples))
            else:
                select_trans_datasets['train'] = trans_datasets['train']
        if training_args.do_eval:
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(trans_datasets['validation']), data_args.max_eval_samples)
                select_trans_datasets['validation'] = trans_datasets['validation'].select(range(max_eval_samples))
            else:
                select_trans_datasets['validation'] = trans_datasets['validation']
        if training_args.do_predict:
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(trans_datasets['test']), data_args.max_predict_samples)
                select_trans_datasets['test'] = trans_datasets['test'].select(range(max_predict_samples))
            else:
                select_trans_datasets['test'] = trans_datasets['test']
        select_trans_datasets = DatasetDict(select_trans_datasets)
    return select_trans_datasets

# model
def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None

def filter_inputs_for_func(inputs, func:Optional[Callable]=None, param_names:Optional[List[str]]=None):
    """
    filter inputs for func
    :param inputs:
    :param func:
    :return:
    """
    if func is not None:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
    assert param_names is not None, 'param_names or func must be provided!'
    filtered_inputs = {}
    for key in inputs.keys():
        if key in param_names:
            filtered_inputs[key] = inputs[key]
    return filtered_inputs

class Pool(nn.Module):
    def __init__(self, pool_type:str):
        super().__init__()
        self.pool_type = pool_type
        assert self.pool_type in ['cls', 'avg', 'avg_top2',
                                  'avg_first_last'], 'unsupported pool type %s' % self.pool_type

    @staticmethod
    def mask_mean(tensor: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        :param tensor: [batch_size, seq_len, hidden_size]
        :param mask: [batch_size, seq_len]
        :return: [batch_size, hidden_size]
        """
        masked_tensor = tensor * mask.unsqueeze(-1)
        sum_tensor = masked_tensor.sum(dim=1)
        masked_sum = mask.sum(dim=1, keepdim=True)
        return sum_tensor / masked_sum

    def forward(self, outputs, attention_mask):
        if self.pool_type == 'cls':
            return outputs.last_hidden_state[:, 0]
        elif self.pool_type == 'avg':
            return self.mask_mean(outputs.last_hidden_state, attention_mask)
        elif self.pool_type == 'avg_first_last':
            first_hidden = outputs.hidden_states[0]
            last_hidden = outputs.hidden_states[-1]
            return self.mask_mean((first_hidden+last_hidden)/2.0, attention_mask)
        elif self.pool_type == 'avg_top2':
            second_hidden = outputs.hidden_states[-2]
            last_hidden = outputs.hidden_states[-1]
            return self.mask_mean((second_hidden+last_hidden)/2.0, attention_mask)
        else:
            raise NotImplementedError

class Similarity(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class InfoNCE(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.sim_fct = Similarity(temp)

    def forward(self, pos1, pos2, neg1, neg2):
        p_sim = self.sim_fct(pos1, pos2)
        n_sim = self.sim_fct(neg1, neg2)

        cos_sim_labels = torch.zeros(int(pos1.shape[0])).long().to(pos1.device)
        cos_sim = torch.stack([p_sim, n_sim], dim=1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, cos_sim_labels)
        return loss

class BiEncoderForClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(
            config.model_name_or_path,
            config=config,
            #device_map="auto",
            torch_dtype="auto",
        )

        if config.train_with_lora:
            lora_config = LoraConfig(
                target_modules=['dense'],
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()

        self.pool_layer = Pool(config.pool_type)
        if config.pool_type in ['avg_top2','avg_first_last']:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False

        dropout_rate = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        if config.transform:
            self.transform = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(config.hidden_size, config.hidden_size),
                ACT2FN[config.hidden_act],
            )

        self.loss_fct = InfoNCE
        self.loss_fct_kwargs = {'temp': config.cl_temp}


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            input_ids_2=None,
            attention_mask_2=None,
            token_type_ids_2=None,
            labels=None,
    ):
        batch_size = input_ids.shape[0]
        # sent1 + sent2
        backbone_inputs: Dict = dict()
        backbone_inputs['input_ids'] = concat_features(input_ids, input_ids_2)
        backbone_inputs['attention_mask'] = concat_features(attention_mask, attention_mask_2)
        backbone_inputs['token_type_ids'] = concat_features(token_type_ids, token_type_ids_2)
        if self.config.train_with_lora:
            outputs = self.backbone(**backbone_inputs,
                                    output_hidden_states=self.output_hidden_states)
        else:
            with torch.set_grad_enabled(not self.config.freeze_encoder):
                outputs = self.backbone(**backbone_inputs,
                                        output_hidden_states=self.output_hidden_states)
        outputs = self.pool_layer(outputs, backbone_inputs['attention_mask'])
        if self.config.transform:
            outputs = self.transform(outputs)

        sent1_out, sent2_out = torch.split(outputs, batch_size, dim=0)
        # logits and loss
        loss = None
        pos1, neg1 = einops.rearrange(sent1_out, '(bs t) d -> t bs d', t=2)
        pos2, neg2 = einops.rearrange(sent2_out, '(bs t) d -> t bs d', t=2)
        loss = self.loss_fct(**self.loss_fct_kwargs)(pos1, pos2, neg1, neg2)

        logits = torch.cosine_similarity(sent1_out, sent2_out, dim=-1)
        if labels is not None:
            loss += torch.nn.MSELoss()(logits, labels)

        return {'loss': loss, 'logits': logits}

# tuner
class PruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.trial.Trial, monitor:str, **kwargs):
        super().__init__(**kwargs)
        self.trial = trial
        self.monitor = monitor

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if (state.is_hyper_param_search
            and str_if_contain_in_str_list('eval_', logs.keys(), mode='contained')):
            self.trial.report(logs[self.monitor], state.global_step)

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if self.trial.should_prune():
            raise optuna.TrialPruned()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        log_history: List[Dict[str, float]] = state.log_history
        eval_log_history: List[Dict[str, float]] = []
        for log in log_history:
            if str_if_contain_in_str_list('eval_', log.keys(), mode='contained'):
                eval_log_history.append(log)
                eval_log_history.append(log)
        self.trial.set_user_attr('epoch_history', pd.DataFrame(eval_log_history).to_dict('list'))

class CombinePruner(optuna.pruners.BasePruner):
    """
    combine pruner
    if and_or is and, all pruners must be True, it will return True
    if and_or is or, one pruner is True, it will return True
    """
    def __init__(self, pruner_list: List[optuna.pruners.BasePruner], and_or: str='and'):
        super().__init__()
        self.and_or = and_or
        self.pruner_list = pruner_list

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if self.and_or == 'and':
            for pruner in self.pruner_list:
                if not pruner.prune(study, trial):
                    return False
            return True
        elif self.and_or == 'or':
            for pruner in self.pruner_list:
                if pruner.prune(study, trial):
                    return True
            return False
        else:
            raise Exception('and_or must be and or or')

# train
class Trainer(TrainerBase):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

def combine_dict(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result

def save_yaml_args(yaml_filepath,data):
    with open(yaml_filepath,'w') as f:
        yaml.dump(data,f)

def modify_dict_with_trial(args, trial:Union[optuna.trial.Trial,optuna.trial.FrozenTrial]):
    """
    modify config_dict with hyperparameters for searching of optuna.
    Args:
        args: config dictionary
        trial: optuna.trial.Trial or optuna.trial.FrozenTrial

    Returns:
        modified config_dict
    """
    args = copy.deepcopy(args)
    for key in args.keys():
        value = args[key]
        if isinstance(value, dict):
            if 'type' in value.keys(): # need modified
                print(value)
                cls = value['type']
                value.pop('type')
                if cls == 'int': # low high step log
                    args[key] = trial.suggest_int(key,**value)
                elif cls == 'float': # low high step log
                    args[key] = trial.suggest_float(key,**value)
                elif cls == 'discrete_uniform': # low high q
                    args[key] = trial.suggest_discrete_uniform(key,**value)
                elif cls == 'uniform': # low high
                    args[key] = trial.suggest_uniform(key,**value)
                elif cls == 'loguniform': # low high
                    args[key] = trial.suggest_loguniform(key,**value)
                elif cls == 'categorical': # choices
                    args[key] = trial.suggest_categorical(key,**value)
                else:
                    raise ValueError('cls must be in [int, float, discrete_uniform, uniform, loguniform, categorical]')
            else:
                args[key] = modify_dict_with_trial(value, trial)
        else:
            pass
    return args

def trainer_one_args(tokenizer_model_args: TokenizerAndModelArguments,
                   data_args: DataTrainingArguments,
                   training_args: TrainingArguments,
                   data:Optional[Union[Dataset,DatasetDict]]=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_args.model_name_or_path)
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    data_args.max_seq_length = max_seq_length

    # set dataset
    if data is None:
        data = load_datasets_func(data_args, training_args, tokenizer)
    datasetsDict = data
    del data
    # set DataCollator
    collate_fn = DataCollatorWithPaddingForMultiRenameInputs(
        [['input_ids','input_ids_2']],
        tokenizer,
        padding=data_args.padding,
        max_length=max_seq_length,
        return_tensors='pt')

    # construct model
    config = AutoConfig.from_pretrained(tokenizer_model_args.model_name_or_path) # for model
    tokenizer_model_dict = asdict(tokenizer_model_args)
    config.update(tokenizer_model_dict)
    def model_init(trial: Optional[Union[optuna.trial.Trial,optuna.trial.FrozenTrial]]):
        # seed for model
        if training_args.tuner:
            transformers.enable_full_determinism(training_args.seed)  # each trial should have same seed
        else:
            pass # for run ordinary model

        config_cp = copy.deepcopy(config) # model default params
        if trial is None:
            model = BiEncoderForClassification(config=config_cp)
            return model

        # update params
        update_params = {key.replace('tuner_',''):value
                         for key,value in tokenizer_model_dict.items() if key.startswith('tuner_')}
        update_params = modify_dict_with_trial(update_params, trial)
        logger.info(f"Trial Model: {update_params}")

        # update config_cp
        config_cp.update(update_params)

        # save config for trial
        # model/data/training
        all_config = combine_dict(config_cp.to_dict(), asdict(data_args), training_args.to_dict())
        trial.set_user_attr('config', all_config)

        return BiEncoderForClassification(config=config_cp)

    def compute_metrics(output: EvalPrediction):
        #inputs = output.inputs
        logits = output.predictions
        labels = output.label_ids
        return {
            'pearsonr': pearsonr(logits, labels)[0],
            'spearmanr': spearmanr(logits, labels)[0],
        }

    # set trainer
    trainer = Trainer(
        model_init=model_init, # we only set seed in tuner
        args=training_args,
        train_dataset=datasetsDict['train'] if training_args.do_train else None,
        eval_dataset=datasetsDict['validation'] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(training_args.early_stopping_patience)],
    )

    # tuner/grid
    if training_args.tuner:
        # only tuner
        training_args.do_train = False
        training_args.do_eval = False

        # set hp_space_hook to initial trial callback each trial
        def hp_space_hook(trial: Union[optuna.trial.Trial,optuna.trial.FrozenTrial]):
            callback_str_list: List[str] = trainer.callback_handler.callback_list.split('\n')
            current_module = sys.modules[__name__]
            trial_callback_class = getattr(current_module, "PruningCallback")
            if str_if_contain_in_str_list('PruningCallback', callback_str_list, mode='contained'):
                trainer.callback_handler.remove_callback(trial_callback_class)
            trainer.add_callback(trial_callback_class(trial, monitor=training_args.metric_for_best_model))
            return dict()
        # set hp search
        if not os.path.exists(training_args.tuner_results_dir):
            os.makedirs(training_args.tuner_results_dir)
        url_path = "sqlite:///" + os.path.join(training_args.tuner_results_dir, "tuner.db")
        best_trial = trainer.hyperparameter_search(
            #hp_space=lambda trial: dict(),
            hp_space=hp_space_hook,
            compute_objective=lambda metrics: metrics[training_args.metric_for_best_model],
            n_trials=training_args.n_trials,
            direction="minimize" if training_args.metric_for_best_model.endswith('loss') else "maximize",
            backend='optuna',
            study_name='CSTS',
            storage=optuna.storages.RDBStorage(
                url_path,
                heartbeat_interval=60,
                grace_period=120,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(3),
            ),
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(),
            gc_after_trial=True,
            pruner=CombinePruner([
                optuna.pruners.MedianPruner(n_warmup_steps=30),
                optuna.pruners.PercentilePruner(0.1, n_warmup_steps=30)
            ])
            # pruner=optuna.pruners.NopPruner(),
        )
        # log hyperparameters
        best_params = best_trial.hyperparameters
        logger.info(f"Best Params: {best_params}")
        
        # use best config to run best result
        run_tokenizer_model_args = copy.deepcopy(tokenizer_model_args)
        for key in best_params.keys():
            if key.startswith('tuner_'):
                run_tokenizer_model_args.__setattr__(key, best_params[key])
        run_data_args = copy.deepcopy(data_args)
        run_training_args = copy.deepcopy(training_args)
        run_training_args.do_train = True
        run_training_args.do_eval = True
        run_training_args.tuner = False
        mean_eval_result, best_config = train_times(
            run_tokenizer_model_args,
            run_data_args,
            run_training_args,
            num_times=training_args.run_times,
            mean_results=True)

        # save best eval metrics and all config
        all_config_and_eval_metrics = combine_dict(mean_eval_result, best_config)
        all_config_and_eval_metrics['tuner'] = False # modify tuner to False for running
        save_dir = training_args.tuner_results_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_yaml_args(save_dir+'conf.yaml', all_config_and_eval_metrics)

    # run once
    eval_metrics = None
    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None:
            train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics('train', metrics)
        if training_args.run_times == 1:
            trainer.save_metrics('train', metrics) # last Metrics in training_args.output_dir
            trainer.save_model() # last Model and Tokenizer in training_args.output_dir
            trainer.save_state() # last Trainer State in training_args.output_dir
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        eval_metrics = {k:v for k, v in metrics.items() if k.startswith('eval_')}
        if training_args.run_times == 1:
            trainer.save_metrics('eval', metrics) # last Metrics in training_args.output_dir

    return eval_metrics, combine_dict(config.to_dict(), asdict(data_args), training_args.to_dict())

def train_times(tokenizer_model_args: TokenizerAndModelArguments,
                data_args: DataTrainingArguments,
                training_args: TrainingArguments,
                num_times: int = 1,
                mean_results: bool = False):
    # initial
    repeat_eval_metrics = []
    config = None
    for idx in range(num_times):
        # seed
        transformers.enable_full_determinism(training_args.seed + idx * 100)
        if idx > 0: # manual set new logging_dir
            training_args.logging_dir = os.path.expanduser(os.path.join(training_args.output_dir, default_logdir()))
        eval_metrics, config = trainer_one_args(tokenizer_model_args, data_args, training_args)
        repeat_eval_metrics.append(eval_metrics)

    if repeat_eval_metrics[0] is not None:
        if mean_results and num_times > 1:
            eval_metrics_df = pd.DataFrame(repeat_eval_metrics)
            mean_ = eval_metrics_df.mean(axis=0).to_dict()
            std_ = eval_metrics_df.std(axis=0).to_dict()
            mean_std_metric_dict = {}
            for k in mean_.keys():
                mean_std_metric_dict[k] = {'mean': mean_[k], 'std': std_[k]}
            return mean_std_metric_dict, config
        else:
            return listdict_map_dictlist(listdict=repeat_eval_metrics, dictlist=None), config
    else:
        return None, config

def main():
    parser = HfArgumentParser(
        [TokenizerAndModelArguments, DataTrainingArguments, TrainingArguments]
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith('.yaml'):
        tokenizer_model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1]), 
            allow_extra_keys=True,
        )
    else:
        tokenizer_model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    if training_args.tuner:
        # seed
        transformers.enable_full_determinism(training_args.seed)
        trainer_one_args(tokenizer_model_args, data_args, training_args)
    else:
        # checkpoint
        if training_args.resume_from_checkpoint is not None:
            training_args.run_times = 1
            if training_args.resume_from_checkpoint == 'last':
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                if last_checkpoint is None:
                    raise ValueError(f"--resume_from_checkpoint '{training_args.resume_from_checkpoint}' not found")
                training_args.resume_from_checkpoint = last_checkpoint
            else:
                if not os.path.exists(training_args.resume_from_checkpoint):
                    raise ValueError(f"--resume_from_checkpoint '{training_args.resume_from_checkpoint}' not found")

        mean_eval_result, config = train_times(tokenizer_model_args, data_args, training_args, training_args.run_times, True)
        if mean_eval_result is None:
            pass # not eval
        else:
            all_config_and_eval_metrics = combine_dict(mean_eval_result, config)
            save_dir = training_args.run_results_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_yaml_args(save_dir + 'conf.yaml', all_config_and_eval_metrics)

if __name__ == '__main__':
    main()
