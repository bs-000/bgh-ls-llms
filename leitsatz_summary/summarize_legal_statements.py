#!/usr/bin/env python3
from random import randrange

import pandas as pd
from matplotlib import pyplot, pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import seaborn as sns

import metrics
import utils
from custom_rouge import rouge_l, rouge_n
from data.download_rii import get_selected_bgh_data
from settings import random_state
from utils import server_path, entscheidungsgruende_str, leitsatz_str, filter_topical_leitsaetze, aktenzeichen_str

current_path = 'leitsatz_summary/'
directory_extension = 'legal_statements/'
eval_result_path_simple = 'eval_results/leitsatz_summary_simple/'
picture_path_simple = 'pictures/leitsatz_summary_simple/'
picture_path_manual_results = 'pictures/'+directory_extension
extractive_str = 'extractive'
precision_str = 'precision'
recall_str = 'recall'
fscore_str = 'fscore'
summary_str = 'summary'
dataframes_dir = 'dataframes/' + directory_extension
manual_eval_dir = 'manual_eval/' + directory_extension
dataframe_name_original_data = 'original_data.json'
prompts = ['Schreibe einen Satz als Zusammenfassung des folgenden Textes.']
bertscore_str = 'bertscore_'
rouge_l_str = 'rouge_l_'
rouge_1_str = 'rouge_1_'
rouge_2_str = 'rouge_2_'
rouge_3_str = 'rouge_3_'
max_tokens_to_generate = 2048
temperature = 0.1
# https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1
mod_llama3_short_inst = 'llama3_8k_instruct'
# https://huggingface.co/DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1
mod_llama3_long_inst = 'llama3_32k_instruct'
# https://huggingface.co/occiglot/occiglot-7b-de-en-instruct/tree/main
mod_occiglot_inst = 'occiglot_32k_instruct'
# https://huggingface.co/LeoLM/leo-mistral-hessianai-7b-chat
mod_mistral_inst = 'mistral_32k_instruct'
# https://huggingface.co/LeoLM/leo-hessianai-7b-chat
mod_llama2_inst = 'llama2_8k_instruct'
instruct_models = {
    mod_llama3_short_inst: 'DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1',
    mod_llama3_long_inst: 'DiscoResearch/Llama3-DiscoLeo-Instruct-8B-32k-v0.1',
    mod_occiglot_inst: 'occiglot/occiglot-7b-de-en-instruct',
    mod_llama2_inst: 'LeoLM/leo-hessianai-7b-chat',
    mod_mistral_inst: 'LeoLM/leo-mistral-hessianai-7b-chat'
}
# https://huggingface.co/DiscoResearch/Llama3-German-8B
mod_llama3_short = 'llama3_8k'
# https://huggingface.co/DiscoResearch/Llama3-German-8B-32k
mod_llama3_long = 'llama3_32k'
# https://huggingface.co/occiglot/occiglot-7b-de-en
mod_occiglot = 'occiglot_32k'
# https://huggingface.co/LeoLM/leo-mistral-hessianai-7b
mod_mistral = 'mistral_32k'
# https://huggingface.co/LeoLM/leo-hessianai-7b
mod_llama2 = 'llama2_8k'
fine_tune_models = {
    mod_mistral: "LeoLM/leo-mistral-hessianai-7b",
    mod_llama2: 'LeoLM/leo-hessianai-13b',
    mod_llama3_long: 'DiscoResearch/Llama3-German-8B-32k',
    mod_llama3_short: 'DiscoResearch/Llama3-German-8B',
    mod_occiglot: 'occiglot/occiglot-7b-de-en'
}
mod_random = 'random_sentence'
models = {**instruct_models, **fine_tune_models}
model_str = 'model'
original_str = 'Original'
max_sequence_lengths = {mod_mistral_inst: 32768, mod_llama3_short_inst: 8192, mod_llama2_inst: 8192,
                        mod_llama3_long_inst: 32768, mod_occiglot_inst: 32768, mod_mistral: 32768,
                        mod_llama3_short: 8192, mod_llama2: 8192, mod_llama3_long: 32768, mod_occiglot: 32768}
metric_names = [bertscore_str + fscore_str, rouge_l_str + fscore_str, rouge_1_str + fscore_str,
                rouge_2_str + fscore_str]


def prepare_raw_data(row):
    """
    Prepares Leitsatz (removes leading listing etc) and Entscheidungsgruende (Removes Randnummern, but inserts '\n' for
    them) and determines, whether the leitsatz is extractive. For one row in parallel mode.

    :param row: dataframe row, containing aktenzeichen, leitsatz and entscheidungsgruende raw
    :return: a new dataframe with the prepared data (aktenzeichen, leitsatz, entscheidungruende, extractive?)
    """
    index, row_data = row
    ls = utils.prepare_leitsatz(row_data[leitsatz_str])
    eg = utils.prepare_entsch_gr(row_data[entscheidungsgruende_str])
    extractive = set(ls).issubset(eg)
    eg_final = ''
    for i in range(len(eg)):
        current = eg[i]
        if not current.isdigit():
            eg_final += ' ' + current
        else:
            eg_final += '\n'
    eg_final = eg_final.strip()
    res = pd.DataFrame({aktenzeichen_str: [row_data[aktenzeichen_str]],
                        leitsatz_str: [' '.join(ls)],
                        entscheidungsgruende_str: [eg_final],
                        extractive_str: [extractive]})
    return res


def get_ls_data():
    """
    Method to coordinate the loading of the original data. If it is already saved in a file, the file is loaded,
    otherwise the data is prepared and saved

    :return: The data to work on. A dataframe with the columns
            (aktenzeichen, leitsatz, entscheidungsgruende, extractive)
    """
    try:
        data_res = utils.df_from_json(current_path=current_path, path=dataframes_dir + dataframe_name_original_data)
    except OSError as _:
        data_raw = get_selected_bgh_data(case=0, directory=server_path(current_path=current_path, path='../data/'))
        data_raw = data_raw.dropna(subset=[leitsatz_str, entscheidungsgruende_str])
        data_raw = filter_topical_leitsaetze(data_raw)
        data_raw = data_raw[[aktenzeichen_str, leitsatz_str, entscheidungsgruende_str]]
        results = utils.parallel_apply_async(function=prepare_raw_data, data=data_raw)
        data_res = pd.DataFrame()
        for res in results:
            data_res = pd.concat([data_res, res.get()], ignore_index=True)
        utils.create_dir(current_path=current_path, delete=False, directory_name=dataframes_dir)
        utils.df_to_json(current_path=current_path, path=dataframes_dir + dataframe_name_original_data,
                         dataframe=data_res)
    data_res = data_res.drop_duplicates(leitsatz_str)
    return data_res


def apply_rouge(func, created, reference, identifier, n=None):
    """
    Function to apply any rouge function.

    :param func: the function to apply
    :param created: the created summary
    :param reference: the gold summary
    :param identifier: the identifier of the method
    :param n: n for ROUGE-n, None for ROUGE-l
    :return: a dict containing precision, recall and f-score of the function
    """
    if n is None:
        p, r, f = func(reference_summary=reference, created_summary=created, extended_results=True)
    else:
        p, r, f = func(reference_summary=reference, created_summary=created, extended_results=True, n=n)
    return {identifier + recall_str: r, identifier + precision_str: p, identifier + fscore_str: f}


def evaluate(model_id, eval_result_path):
    """
    Coordinates the evaluation of the model after the evaluation texts were created.

    :param model_id: Model to load the evaluation texts.
    :param eval_result_path: path where to find the texts to evaluate
    """
    eval_results = utils.df_from_json(current_path=current_path, path=eval_result_path + model_id)
    eval_results = evaluate_summaries(eval_results)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path + model_id,
                     dataframe=eval_results)


def evaluate_summaries(df):
    """
    Function for evaluation the created summaries. Uses BERTScore and ROUGE-1 -2 -3 -L and prints mean precision,
    recall and fscore

    :param df: Dataframe containing leitsatz_str and summary_str
    :return: the original df with additional columns for precision, recall and fscore
    """
    scores = metrics.bertscore(gold_sum_sents=df[leitsatz_str].values.tolist(),
                               candidate_sum_sents=df[summary_str].values.tolist())
    df_result = df.copy()
    df_result[bertscore_str + precision_str] = scores[0].squeeze()
    df_result[bertscore_str + recall_str] = scores[1].squeeze()
    df_result[bertscore_str + fscore_str] = scores[2].squeeze()
    # rouge-l
    df_result[[rouge_l_str + precision_str, rouge_l_str + recall_str, rouge_l_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_l, created=row[summary_str],
                                                reference=row[leitsatz_str], identifier=rouge_l_str),
                        axis='columns', result_type='expand')
    # rouge-1
    df_result[[rouge_1_str + precision_str, rouge_1_str + recall_str, rouge_1_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_n, created=row[summary_str], n=1,
                                                reference=row[leitsatz_str], identifier=rouge_1_str),
                        axis='columns', result_type='expand')
    # rouge-2
    df_result[[rouge_2_str + precision_str, rouge_2_str + recall_str, rouge_2_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_n, created=row[summary_str], n=2,
                                                reference=row[leitsatz_str], identifier=rouge_2_str),
                        axis='columns', result_type='expand')
    # rouge-3
    df_result[[rouge_3_str + precision_str, rouge_3_str + recall_str, rouge_3_str + fscore_str]] = \
        df_result.apply(lambda row: apply_rouge(rouge_n, created=row[summary_str], n=3,
                                                reference=row[leitsatz_str], identifier=rouge_3_str),
                        axis='columns', result_type='expand')
    return df_result


def get_tokenizer(model_id):
    """
    Loads the tokenizer for the given model

    :param model_id: Model to load tokenizer for
    :return: the loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(models[model_id], trust_remote_code=True)
    tokenizer.padding_side = 'right'
    return tokenizer


def count_tokens(model_id, complete_df, picture_path):
    """
    Counts the tokens of leitsatz and entscheidunsgruende in each case as tokenized by the tokenizer of the given model.

    :param model_id: Model for tokenizer
    :param complete_df: Dataframe with the cases
    :param picture_path: path to save the resulting figrues at
    """
    tokenizer = get_tokenizer(model_id)
    count_tokens_one_type(dataframe=complete_df, model_id=model_id, tokenizer=tokenizer, to_count=leitsatz_str,
                          picture_path=picture_path)
    count_tokens_one_type(dataframe=complete_df, model_id=model_id, tokenizer=tokenizer,
                          to_count=entscheidungsgruende_str, picture_path=picture_path)
    count_tokens_combined(dataframe=complete_df, model_id=model_id, tokenizer=tokenizer,
                          picture_path=picture_path)
    print('Done counting tokens')


def count_tokens_one_type(dataframe, to_count, tokenizer, model_id, picture_path):
    """
    Counts the tokens for one text type. Plots the result as bar plot.

    :param dataframe: dataframe containing the data as columm
    :param to_count: column to count
    :param tokenizer: the tokenizer
    :param model_id: the model
    :param picture_path: path to save the resulting figrues at
    """
    num_tokens = dataframe[to_count].apply(lambda x: len(tokenizer(x).encodings[0].ids))
    num_tokens["bins"] = pd.cut(num_tokens, [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ax = num_tokens["bins"].value_counts(sort=False).plot.bar()
    ax.bar_label(ax.containers[0])
    plot = ax.get_figure()
    plot.tight_layout()
    savepath = picture_path + 'tokens/'
    utils.create_dir(current_path=current_path, directory_name=savepath, delete=False)
    plot.savefig(utils.server_path(current_path=current_path, path=savepath + model_id + '_' + to_count + '.png'))
    pyplot.clf()


def count_tokens_combined(dataframe, tokenizer, model_id, picture_path):
    """
    Counts the tokens for leitsatz and entscheidungsgruende concatenated. Plots the result as bar plot.

    :param dataframe: dataframe containing the data as columm
    :param tokenizer: the tokenizer
    :param model_id: the model
    :param picture_path: path to save the resulting figrues at
    """
    num_tokens = (dataframe[entscheidungsgruende_str]+dataframe[leitsatz_str]).\
        apply(lambda x: len(tokenizer(x).encodings[0].ids))
    num_tokens["bins"] = pd.cut(num_tokens, [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ax = num_tokens["bins"].value_counts(sort=False).plot.bar()
    ax.bar_label(ax.containers[0])
    plot = ax.get_figure()
    plot.tight_layout()
    savepath = picture_path + 'tokens/'
    utils.create_dir(current_path=current_path, directory_name=savepath, delete=False)
    plot.savefig(utils.server_path(current_path=current_path, path=savepath + model_id + '_concatenated.png'))
    pyplot.clf()


def visualize_eval_results(model_id, test_df, eval_result_path, picture_path):
    """
    Plots results for the evaluation of the summaries (ROUGE / BERTScore).
    As Boxplots and Correlation between tokennumber or extractive /abstractive.

    :param model_id: Model to plot for
    :param test_df: dataframe with the test data for token counting
    :param eval_result_path: path where to find the data to visualize
    :param picture_path: path where to save the resulting figures
    """
    res_df = utils.df_from_json(current_path, path=eval_result_path + model_id)
    cols = res_df.columns.values.tolist()
    cols.remove(leitsatz_str)
    cols.remove(summary_str)
    ax = res_df.boxplot(column=cols, rot=90)
    plot = ax.get_figure()
    plot.tight_layout()
    savepath = picture_path + 'eval/'
    utils.create_dir(current_path=current_path, directory_name=savepath, delete=False)
    plot.savefig(utils.server_path(current_path=current_path, path=savepath + model_id + '.png'))
    pyplot.clf()

    if model_id not in [mod_random]:
        joined_df = res_df.join(test_df.set_index(leitsatz_str), on=leitsatz_str)
        tokenizer = get_tokenizer(model_id)
        counts_str = 'token_counts'
        joined_df[counts_str] = joined_df[entscheidungsgruende_str].apply(lambda x: len(tokenizer(x).encodings[0].ids))
        joined_df = joined_df.drop(
            columns=[entscheidungsgruende_str, leitsatz_str, summary_str, utils.aktenzeichen_str])
        joined_df_tokens = joined_df.drop(columns=[extractive_str])
        joined_df_extractive = joined_df.drop(columns=[counts_str])

        index_str = 'index'
        # dann abhängigkeit der Werte zur länge plotten (ganz, geteilt)
        cor_extractive = joined_df_extractive.corrwith(joined_df_extractive[extractive_str]).drop(
            labels=[extractive_str])
        cor_extractive[index_str] = 'extractive'
        cor_tokens_all = joined_df_tokens.corrwith(joined_df_tokens[counts_str]).drop(labels=counts_str)
        cor_tokens_all[index_str] = 'all tokens'
        cor_tokens_smaller = joined_df_tokens[joined_df_tokens[counts_str] < max_sequence_lengths[model_id]].corrwith(
            joined_df_tokens[counts_str]).drop(labels=counts_str)
        cor_tokens_smaller[index_str] = 'tokens less than limit'
        cor_tokens_greater = joined_df_tokens[joined_df_tokens[counts_str] >= max_sequence_lengths[model_id]].corrwith(
            joined_df_tokens[counts_str]).drop(labels=counts_str)
        cor_tokens_greater[index_str] = 'tokens more than limit'
        cor_df = pd.DataFrame([cor_extractive, cor_tokens_all, cor_tokens_smaller, cor_tokens_greater]) \
            .set_index(index_str).T
        ax = cor_df.plot(rot=90)
        pyplot.xticks(range(0, len(cor_df.index)), cor_df.index)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(utils.server_path(current_path=current_path, path=savepath + model_id + '_cor.png'))
        pyplot.clf()
    print('Done with visualize results')


def save_texts_for_eval(test_df):
    """
    Selects 50 random cases from the dataframe and prints their generated result for every model in an excel-file to
    evaluate.

    :param test_df: Dataframe to select from
    """
    res_df = pd.DataFrame()
    for model_id in instruct_models:
        text_df = utils.df_from_json(current_path=current_path, path='eval_results/prompt_eng/'+model_id)[[leitsatz_str,
                                                                                                           summary_str]]
        joined_df = text_df.join(test_df.set_index(leitsatz_str), on=leitsatz_str)[[aktenzeichen_str, leitsatz_str,
                                                                                    summary_str]]
        joined_df[model_str] = model_id
        res_df = pd.concat([res_df, joined_df], ignore_index=True)

    for model_id in [mod_random]:
        text_df = utils.df_from_json(current_path=current_path, path=eval_result_path_simple + model_id)[
            [leitsatz_str, summary_str]]
        text_df[leitsatz_str] = text_df[leitsatz_str].str.strip()
        test_df_copy = test_df.copy()
        test_df_copy[leitsatz_str] = test_df_copy[leitsatz_str].str.strip()
        joined_df = test_df_copy.join(text_df.set_index(leitsatz_str), on=leitsatz_str)[
            [aktenzeichen_str, leitsatz_str, summary_str]]
        joined_df[leitsatz_str] = test_df[leitsatz_str]
        joined_df[model_str] = model_id
        res_df = pd.concat([res_df, joined_df], ignore_index=True)

    for model_id in fine_tune_models:
        text_df = utils.df_from_json(current_path=current_path, path='eval_results/finetune_for_leitsatz/' + model_id)[
            [leitsatz_str, summary_str]]
        joined_df = text_df.join(test_df.set_index(leitsatz_str), on=leitsatz_str)[
            [aktenzeichen_str, leitsatz_str, summary_str]]
        joined_df[model_str] = model_id
        res_df = pd.concat([res_df, joined_df], ignore_index=True)

    save_df = pd.DataFrame()
    for aktenzeichen in res_df[aktenzeichen_str].unique():
        if pd.isna(aktenzeichen):
            continue
        az_data = res_df[res_df[aktenzeichen_str] == aktenzeichen]
        save_df = pd.concat([save_df,
                             pd.DataFrame.from_dict({aktenzeichen_str: [aktenzeichen],
                                                     leitsatz_str: [az_data[leitsatz_str].values[0]],
                                                     model_str: [original_str]})])
        az_data[leitsatz_str] = az_data[summary_str]
        az_data = az_data.drop(columns=[summary_str])
        save_df = pd.concat([save_df, az_data])
    save_df = save_df.reset_index()
    eval_classes = ['Klasse '+str(i+1) for i in range(9)]
    for i in eval_classes:
        save_df[i] = ''
    utils.create_dir(current_path=current_path, delete=False, directory_name=manual_eval_dir)
    save_df = save_df[save_df[aktenzeichen_str].isin(save_df[aktenzeichen_str].sample(50, random_state=random_state).
                                                     tolist())]

    save_df.to_excel(utils.server_path(current_path=current_path, path=manual_eval_dir + 'combined.xlsx'),
                     index=False, columns=[aktenzeichen_str, leitsatz_str, model_str]+eval_classes)
    print('Done save texts for eval')


def select_random_leitsaetze(test_df):
    """
    Method selects random sentences as leitsaetze and saves them in a file.

    :param test_df: Dataframe with the data
    """
    res_df = pd.DataFrame()
    for index, row in test_df.iterrows():
        sentences = [utils.remove_leading_listing(sent) for sent in
                     utils.split_into_sentences(row[entscheidungsgruende_str])]
        sentences = [sent for sent in sentences if len(sent.split()) > 2]
        rand_index = randrange(len(sentences))
        result = pd.DataFrame.from_dict({leitsatz_str: [row[leitsatz_str]], summary_str: [sentences[rand_index]]})
        res_df = pd.concat([res_df, result], ignore_index=True)

    utils.create_dir(current_path=current_path, directory_name=eval_result_path_simple, delete=False)
    utils.df_to_json(current_path=current_path, path=eval_result_path_simple + mod_random, dataframe=res_df)


def create_random_leitsaetze(test_df):
    """
    Wrapper method for creating random leitsaetze. First selects the sentences, then evaluates them and visualizes the
    results.

    :param test_df: data to work on
    """
    select_random_leitsaetze(test_df=test_df)
    evaluate(model_id=mod_random, eval_result_path=eval_result_path_simple)
    visualize_eval_results(model_id=mod_random, eval_result_path=eval_result_path_simple, test_df=test_df,
                           picture_path=picture_path_simple)
    print('Done creating random leitsaetze.')


def visualize_manual_eval():
    """
    Plots figures of the results of the manual evaluation.
    """
    eval_classes = ['Klasse ' + str(i + 1) for i in range(9)]
    # class 4 was removed after some reconsiderations, so class 5 became class 4 and so on
    eval_classes.remove('Klasse 4')
    df = pd.read_excel(server_path(current_path=current_path, path=manual_eval_dir+"combined_nw.xlsx"))
    df = df[df[model_str] != original_str]
    df = df.drop(['Unnamed: 6'], axis=1)
    df = df.drop(df[df[eval_classes[0]].isna()].index)

    save_extension = 'metrics/'
    print_df = pd.DataFrame()
    for model in ['eval_results/finetune_for_leitsatz/' + elem for elem in list(fine_tune_models.keys())] + \
                 ['eval_results/prompt_eng/' + elem for elem in list(instruct_models.keys())] + \
                 [eval_result_path_simple+mod_random]:
        metrics_df = utils.df_from_json(current_path, path=model)
        dataframe = df
        if model != '':
            dataframe = df[df[model_str] == model.split('/')[-1]]
        joined = dataframe.join(metrics_df.set_index(summary_str), on=leitsatz_str, lsuffix='_dataframe')
        classes_str = 'classes'
        joined[classes_str] = (joined[eval_classes] == 'y').sum(axis=1).div(len(eval_classes))
        res = joined[metric_names+[aktenzeichen_str, classes_str]]
        res = res.sort_values(classes_str)
        res = res.drop(aktenzeichen_str, axis=1)
        res_mean = res.mean()
        print(model)
        print(res_mean)
        print_df[model.split('/')[-1]] = res_mean

    axes = print_df.plot(rot=90)
    utils.create_dir(current_path=current_path, directory_name=picture_path_manual_results + save_extension,
                     delete=False)
    fig = axes.get_figure()
    fig.tight_layout()
    fig.savefig(picture_path_manual_results + save_extension + 'metrics_all.png')
    plt.clf()

    axes = print_df[[mod_random] + list(fine_tune_models.keys())].plot(rot=90)
    fig = axes.get_figure()
    fig.tight_layout()
    fig.savefig(picture_path_manual_results + save_extension + 'metrics_finetune.png')
    plt.clf()

    axes = print_df[[mod_random] + list(instruct_models.keys())].plot(rot=90)
    fig = axes.get_figure()
    fig.tight_layout()
    fig.savefig(picture_path_manual_results + save_extension + 'metrics_prompt.png')
    plt.clf()

    save_extension = 'correlations/'
    for model in list(fine_tune_models.keys()) + list(instruct_models.keys()) + ['', mod_random]:
        dataframe = df
        if model != '':
            dataframe = df[df[model_str] == model]
        dataframe = dataframe[eval_classes] == 'y'
        dataframe.replace('y', 1)
        dataframe.replace('n', 0)
        corr = dataframe.corr()
        axes = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
        utils.create_dir(current_path=current_path, directory_name=picture_path_manual_results + save_extension,
                         delete=False)
        axes.get_figure().savefig(picture_path_manual_results + save_extension + 'average_class_' + model + '.png')
        plt.clf()

    save_extension = 'class_counts_per_case/'
    for model in list(fine_tune_models.keys()) + list(instruct_models.keys()) + ['', mod_random]:
        dataframe = df
        if model != '':
            dataframe = df[df[model_str] == model]
        yes = (dataframe[eval_classes] == 'y').sum(axis=1)
        res = yes.value_counts().sort_index()
        if model == '':
            res = res.div(12)
        axes = res.plot.bar()
        utils.create_dir(current_path=current_path, directory_name=picture_path_manual_results+save_extension,
                         delete=False)
        axes.get_figure().savefig(picture_path_manual_results+save_extension+'average_class_'+model+'.png')
        plt.clf()

    save_extension = 'total_class_counts/'
    for model in list(fine_tune_models.keys()) + list(instruct_models.keys()) + ['', mod_random]:
        dataframe = df
        if model != '':
            dataframe = df[df[model_str] == model]
        yes = (dataframe[eval_classes] == 'y').sum()
        no = (dataframe[eval_classes] == 'n').sum()
        combined = pd.concat([yes, no], axis=1)
        combined.columns = ['y', 'n']
        if model == '':
            combined = combined.div(12)
        print(model)
        print(combined)
        axes = combined.plot.bar()
        utils.create_dir(current_path=current_path, directory_name=picture_path_manual_results+save_extension,
                         delete=False)
        axes.get_figure().savefig(picture_path_manual_results+save_extension+'average_class_'+model+'.png')
        plt.clf()
    print('Done visualizing manual evaluation')


def correlation_scores_judgement_length(test_df):
    """
    Calculates correlation values between the length of the input data and the calculated metrics (ROUGE + BERTScore)
    :param test_df: Dataframe with the original data for mapping
    """
    length_str = 'length'
    for model_id in [mod_random]:
        text_df = utils.df_from_json(current_path=current_path, path=eval_result_path_simple + model_id)
        text_df[leitsatz_str] = text_df[leitsatz_str].str.strip()
        test_df_copy = test_df.copy()
        test_df_copy[leitsatz_str] = test_df_copy[leitsatz_str].str.strip()
        joined_df = test_df_copy.join(text_df.set_index(leitsatz_str), on=leitsatz_str)[
            [aktenzeichen_str, leitsatz_str, entscheidungsgruende_str, summary_str]+metric_names]
        joined_df[leitsatz_str] = test_df[leitsatz_str]
        joined_df[length_str] = joined_df[entscheidungsgruende_str].apply(lambda x: len(x.split()))
        # spearman because variables aren't normal distributed
        print(model_id)
        print(joined_df[metric_names+[length_str]].corr(method='spearman')[length_str])

    print('instruct')
    for model_id in instruct_models:
        text_df = utils.df_from_json(current_path=current_path, path='eval_results/prompt_eng/'+model_id)
        joined_df = text_df.join(test_df.set_index(leitsatz_str), on=leitsatz_str)[
            [aktenzeichen_str, leitsatz_str, summary_str, entscheidungsgruende_str]+metric_names]
        tokenizer = get_tokenizer(model_id)
        joined_df[length_str] = joined_df[entscheidungsgruende_str].apply(
                  lambda x: len(tokenizer(x).encodings[0].ids))
        print(model_id)
        print(joined_df[metric_names + [length_str]].corr(method='spearman')[length_str])

    print('fine-tune')
    for model_id in fine_tune_models:
        text_df = utils.df_from_json(current_path=current_path, path='eval_results/finetune_for_leitsatz/' + model_id)
        joined_df = text_df.join(test_df.set_index(leitsatz_str), on=leitsatz_str)[
            [aktenzeichen_str, leitsatz_str, summary_str, entscheidungsgruende_str]+metric_names]
        joined_df[leitsatz_str+length_str] = joined_df[leitsatz_str].apply(lambda x: len(tokenizer(x).encodings[0].ids))
        joined_df[entscheidungsgruende_str+length_str] = joined_df[entscheidungsgruende_str].\
            apply(lambda x: len(tokenizer(x).encodings[0].ids))
        joined_df[length_str] = joined_df[leitsatz_str+length_str] + joined_df[entscheidungsgruende_str+length_str]
        print(model_id)
        print(joined_df[metric_names + [length_str]].corr(method='spearman')[length_str])


def correlation_scores_classes(test_df):
    """
    Calculates correlations between the manually assigned classes and the metric scores(and legnth)

    :param test_df: Original test data for the mapping
    """
    length_str = 'length'
    eval_classes = ['Klasse ' + str(i + 1) for i in range(9)]
    eval_classes.remove('Klasse 4')
    df = pd.read_excel(server_path(current_path=current_path, path=manual_eval_dir+"combined_nw.xlsx"))
    df = df[df[model_str] != original_str]
    df = df.drop(['Unnamed: 6'], axis=1)
    df = df.drop(df[df[eval_classes[0]].isna()].index)
    joined_df = df.join(test_df.set_index(aktenzeichen_str), on=aktenzeichen_str, lsuffix='left', how='inner')

    res_df = pd.DataFrame()
    for model_id in instruct_models:
        var = utils.df_from_json(current_path=current_path, path='eval_results/prompt_eng/'+model_id)
        var[model_str] = model_id
        tokenizer = get_tokenizer(model_id)
        var = joined_df.join(var.set_index([leitsatz_str, model_str]), on=[leitsatz_str, model_str], how='inner')
        var[length_str] = var[entscheidungsgruende_str].apply(
            lambda x: len(tokenizer(x).encodings[0].ids))
        var = var[[leitsatz_str, aktenzeichen_str, model_str, length_str]+eval_classes+metric_names]

        res_df = pd.concat([res_df, var])

    for model_id in fine_tune_models:
        var = utils.df_from_json(current_path=current_path, path='eval_results/finetune_for_leitsatz/' + model_id)
        var[model_str] = model_id
        tokenizer = get_tokenizer(model_id)
        var = joined_df.join(var.set_index([leitsatz_str, model_str]), on=[leitsatz_str, model_str], how='inner')

        var[leitsatz_str + length_str] = var[leitsatz_str].apply(
            lambda x: len(tokenizer(x).encodings[0].ids))
        var[entscheidungsgruende_str + length_str] = var[entscheidungsgruende_str].apply(
            lambda x: len(tokenizer(x).encodings[0].ids))
        var[length_str] = var[leitsatz_str + length_str] + var[entscheidungsgruende_str + length_str]
        var = var[[leitsatz_str, aktenzeichen_str, model_str, length_str] + eval_classes + metric_names]

        res_df = pd.concat([res_df, var])

    res = res_df.replace('y', 1).replace('n', 0)
    class_sum = 'class_sum'
    res[class_sum] = res[eval_classes].sum(axis=1)
    for model_id in res_df[model_str].unique():
        print(model_id)
        corr_df = res[res[model_str] == model_id].corr('spearman')
        print(corr_df[length_str])


def analyse_lengths():
    eval_classes = ['Klasse ' + str(i + 1) for i in range(9)]
    eval_classes.remove('Klasse 4')
    df = pd.read_excel(server_path(current_path=current_path, path=manual_eval_dir+"combined_nw.xlsx"))
    df = df.drop(['Unnamed: 6'], axis=1)
    for model in list(instruct_models.keys())+list(fine_tune_models.keys()):
        var = df[df[model_str] == model]
        orig = df[df[model_str] == original_str]
        tokenizer = get_tokenizer(model)
        ls_length = 'leitsatz length'
        var = var.drop(var[var[leitsatz_str].isna()].index)
        var = var.drop(var[var[eval_classes[0]].isna()].index)
        var[ls_length] = var[leitsatz_str].apply(lambda x: len(tokenizer(str(x)).encodings[0].ids))
        orig[ls_length] = orig[leitsatz_str].apply(lambda x: len(tokenizer(str(x)).encodings[0].ids))
        orig = orig[orig[aktenzeichen_str].isin(var[aktenzeichen_str])]
        print(model)
        print('original: ' + str(orig[ls_length].mean()))
        print('model: ' + str(var[ls_length].mean()))
        for ev_class in eval_classes[:-1]:
            var = var.drop(var[var[ev_class] == 'n'].index)
        var = var.drop(var[var[eval_classes[-1]] == 'y'].index)
        print('equal to original: '+str(len(var)))
    print('dsf')


if __name__ == "__main__":
    data = get_ls_data()
    train, test = train_test_split(data, random_state=random_state, test_size=0.3, stratify=data[extractive_str])
    validation, test = train_test_split(test, random_state=random_state, test_size=0.5, stratify=test[extractive_str])
    # create_random_leitsaetze(test_df=test)
    # save_texts_for_eval(test)
    # visualize_manual_eval()
    # correlation_scores_judgement_length(test_df=test)
    analyse_lengths()
    correlation_scores_classes(test_df=test)
    print('Done summarize by finding legal statements')
