#!/usr/bin/env python3

import datasets
import pandas as pd
import torch
import os
from peft import LoraConfig, AutoPeftModelForCausalLM
from sklearn.model_selection import train_test_split
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import SFTTrainer, SFTConfig, setup_chat_format

import utils
from leitsatz_summary.summarize_legal_statements import get_ls_data, extractive_str, summary_str, \
    current_path, max_tokens_to_generate, temperature, count_tokens, visualize_eval_results, evaluate, models, \
    max_sequence_lengths, fine_tune_models
from settings import random_state
from utils import server_path, entscheidungsgruende_str, leitsatz_str

directory_extension = 'finetune_for_leitsatz/'
picture_path = 'pictures/' + directory_extension
model_result_path = 'results/' + directory_extension
eval_result_path = 'eval_results/' + directory_extension
log_history_path = 'log_history/' + directory_extension

num_epochs = 10
final_model_str = '/final_model'


def prepare_data_for_finetuning(example):
    """
    Converts one example to the dict for the chat template

    :param example: sample with leitsatz_str and entscheidungsgruende_str
    :return: the dict in correct format
    """
    return {"messages": [
        {"role": "system", "content": "Du bist ein rechtlicher Assistent. "
                                      "Schreibe einen Leitsatz zum folgenden Gerichtsurteil."},
        {"role": "user", "content": example[entscheidungsgruende_str]},
        {"role": "assistant", "content": example[leitsatz_str]}
    ]}


def load_model_for_finetuning(model_id):
    """
    Loads the model with the given id

    :param model_id: one of the ids in models
    :return: model, tokenizer to this model_id
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        models[model_id],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, get_tokenizer(model_id)


def get_tokenizer(model_id):
    """
    Loads the tokenizer for the given model

    :param model_id: Model to load tokenizer for
    :return: the loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(models[model_id], trust_remote_code=True)
    tokenizer.padding_side = 'right'
    return tokenizer


def get_train_conf(output_dir, model_id):
    """
    Creates the sftconfig

    :param output_dir: Directory for saving checkpoints
    :param model_id: the model identifier
    :return: the config
    """
    max_seq_length = max_sequence_lengths[model_id]
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        eval_accumulation_steps=4,
        save_steps=500,
        logging_steps=500,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        packing=True,
        warmup_ratio=0.03,
        max_seq_length=max_seq_length,
        lr_scheduler_type="constant",
        tf32=True, )


def do_finetuning(train_dataset, valid_dataset, model_id):
    """
    Does the finetuning with PEFT and LORA. Also saves the log history.

    Run with at least 2 GPUs 80 GB

    :param train_dataset: Data to train on
    :param valid_dataset: Data for evaluation
    :param model_id: Model to use
    """
    model, tokenizer = load_model_for_finetuning(model_id)
    model, tokenizer = setup_chat_format(model, tokenizer)
    model_dir = server_path(current_path=current_path, path=model_result_path + model_id)
    sft_conf = get_train_conf(model_dir, model_id)

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_conf,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )
    trainer.train()
    # trainer.train(resume_from_checkpoint=True)

    final_model_dir = model_dir + final_model_str
    trainer.save_model(final_model_dir)
    # Load PEFT model
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_model_dir,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(final_model_dir, safe_serialization=True,
                                 max_shard_size="2GB")

    log_hist = pd.DataFrame(trainer.state.log_history)
    utils.create_dir(current_path=current_path, directory_name=log_history_path, delete=False)
    utils.df_to_json(current_path=current_path, path=log_history_path + model_id,
                     dataframe=log_hist)
    visualize_log_history(log_hist, savepath=picture_path + 'log_history/' + model_id)


def visualize_log_history(dataframe, savepath):
    """
    Plots the log history (losses).

    :param dataframe: Dataframe with the log history.
    :param savepath: Path for saving the file.
    """
    utils.create_dir(current_path=current_path, directory_name=savepath, delete=False)
    fig = dataframe.plot.line(x='epoch', y='loss').get_figure()
    fig.savefig(utils.server_path(current_path=current_path, path=savepath + ".png"))


def generate_texts(dataset, model, tokenizer):
    """
    Method applies the model to the dataset. Dataset is prepared for chat prompt template (as dict).
    Run with multiple GPUs

    :param dataset: Dataset to generate texts to
    :param model: model to use
    :param tokenizer: tokenizer to use
    :return: a dataframe with columns leitsatz_str and summary_str
    """
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16)
    prompts_and_leits = [(tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False,
                                                        add_generation_prompt=True),
                          sample["messages"][2]["content"]) for sample in dataset]

    prompts = [p for (p, _) in prompts_and_leits]
    leits = [l for (_, l) in prompts_and_leits]
    outputs = pipe(prompts, temperature=temperature, #do_sample=True,  top_p=0.95,
                   eos_token_id=pipe.tokenizer.eos_token_id, max_new_tokens=max_tokens_to_generate,
                   pad_token_id=pipe.tokenizer.pad_token_id)
    outputs = [entry[0]['generated_text'][len(prompts[ind]):].strip() for ind, entry in enumerate(outputs)]

    result = pd.DataFrame({leitsatz_str: leits, summary_str: outputs})
    return result


def prepare_one_dataset(df):
    """
    Prepares a dataframe for the chat prompt template

    :param df: dataframe to prepare, must include leitsatz_str and entschgr
    :return: a prepared dataset
    """
    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.map(prepare_data_for_finetuning, remove_columns=list(dataset.features),
                          batched=False)
    return dataset


def generate_eval_texts(model_id, test_dataset):
    """
    Generates the texts to evaluate the model. Saves them in a json file.
    Run on multiple GPUs, 80GB might not be needed. 40GB might be enough

    :param model_id: The model to generate the texts. There must be a final model after fine-tuning!
    :param test_dataset: prepared dataset for generation.

    """
    model_dir = server_path(current_path=current_path, path=model_result_path + model_id) + final_model_str
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        attn_implementation="flash_attention_2",

    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'right'

    eval_results = generate_texts(dataset=test_dataset, model=model, tokenizer=tokenizer)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + model_id, dataframe=eval_results)


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
    data = get_ls_data()
    train, test = train_test_split(data, random_state=random_state, test_size=0.3, stratify=data[extractive_str])
    validation, test = train_test_split(test, random_state=random_state, test_size=0.5, stratify=test[extractive_str])

    train_d = prepare_one_dataset(train)
    test_d = prepare_one_dataset(test)
    validation_d = prepare_one_dataset(validation)

    for m_id in fine_tune_models.keys():
        # do_finetuning(train_dataset=train_d, valid_dataset=validation_d, model_id=m_id)
        # generate_eval_texts(model_id=m_id, test_dataset=test_d)
        evaluate(model_id=m_id, eval_result_path=eval_result_path)
        count_tokens(model_id=m_id, complete_df=data, picture_path=picture_path)
        visualize_eval_results(model_id=m_id, test_df=test, picture_path=picture_path,
                               eval_result_path=eval_result_path)
    print('Done finetuning for leitsatz')
