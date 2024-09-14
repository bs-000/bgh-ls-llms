import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM

import utils
from leitsatz_summary.summarize_legal_statements import get_ls_data, extractive_str, max_tokens_to_generate, \
    temperature, summary_str, current_path, evaluate, mod_occiglot_inst, mod_mistral_inst, models, \
    mod_llama3_long_inst, mod_llama3_short_inst, mod_llama2_inst, instruct_models, count_tokens, visualize_eval_results
from settings import random_state
from utils import entscheidungsgruende_str, leitsatz_str


directory_extension = 'prompt_eng/'
picture_path = 'pictures/' + directory_extension
eval_result_path = 'eval_results/' + directory_extension


def prepare_data_for_prompt_engineering(example):
    """
    Converts one example to the dict for the chat template

    :param example: sample with leitsatz_str and entscheidungsgruende_str
    :return: the dict in correct format
    """
    return {"messages": [
        {"role": "system", "content": "Du bist ein hilfreicher rechtlicher Assistent."},
        #  prompt adjusted from:
        #  https://rsw.beck.de/docs/librariesprovider27/default-document-library/redrl_chbeck_leits%C3%A4tze_1_7_2018.pdf?sfvrsn=b783470d_8
        {"role": "user", "content": "Schreibe die Leitsätze zum folgenden Gerichtsurteil. "
                                    "Die Leitsätze sollen die wesentlichen vom Gericht entschiedenen Rechtsfragen "
                                    "einschließlich des konkreten Entscheidungsergebnisses wiedergeben. Jede "
                                    "Rechtsfrage ist dabei grundsätzlich in einem eigenen Leitsatz darzustellen. "
                                    "Nenne dazu erst die Rechtsfrage und dann den dazu passenden Leitsatz. "
                                    "Nenne nur Rechtsfragen, die bisher in der Literatur unbeantwortet waren und im "
                                    "Urteil neu entschieden wurden. "
                                    "Nutze das folgende Format: **RF**:Rechtsfrage\n**LS**:Leitsatz\n######\nUrteil: "
                                    + example[entscheidungsgruende_str]},
    ]}


def generate_texts(dataframe, model_id):
    """
    Method applies the model to the dataframe.
    Run with multiple GPUs

    :param dataframe: Dataset to generate texts to
    :param model_id: model to use
    :return: a dataframe with columns leitsatz_str and summary_str
    """
    if model_id in [mod_occiglot_inst, mod_mistral_inst]:
        model = MistralForCausalLM.from_pretrained(models[model_id], torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2", device_map="auto")
    elif model_id in [mod_llama3_long_inst, mod_llama3_short_inst, mod_llama2_inst]:
        model = AutoModelForCausalLM.from_pretrained(models[model_id], torch_dtype=torch.bfloat16,
                                                     attn_implementation="flash_attention_2", device_map="auto")
    else:
        model = None
    tokenizer = AutoTokenizer.from_pretrained(models[model_id])

    res_df = pd.DataFrame()
    for index, row in dataframe.iterrows():
        text = tokenizer.apply_chat_template(prepare_data_for_prompt_engineering(row)["messages"],
                                             tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_tokens_to_generate,
                                       pad_token_id=tokenizer.eos_token_id, temperature=temperature,
                                       # top_p=0.95, do_sample=True
                                       )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids
                         in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = pd.DataFrame.from_dict({leitsatz_str: [row[leitsatz_str]], summary_str: [response]})
        res_df = pd.concat([res_df, result], ignore_index=True)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + model_id, dataframe=res_df)

    print('Done generate texts')


if __name__ == "__main__":
    data = get_ls_data()
    train, test = train_test_split(data, random_state=random_state, test_size=0.3, stratify=data[extractive_str])
    validation, test = train_test_split(test, random_state=random_state, test_size=0.5, stratify=test[extractive_str])

    for m_id in instruct_models.keys():
        # generate_texts(test, m_id)
        # evaluate(m_id, eval_result_path=eval_result_path)
        count_tokens(model_id=m_id, complete_df=data, picture_path=picture_path)
        # visualize_eval_results(m_id, train, picture_path=picture_path, eval_result_path=eval_result_path)
        print('Done with '+m_id)

    print('Done with prompt engineering for leitsatz')
