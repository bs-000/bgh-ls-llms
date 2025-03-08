import itertools
import re

import numpy as np
import pandas as pd

import settings
import utils
from data import download_rii
from custom_rouge import rouge_n, rouge_l

pp_options = [utils.pp_option_lemmatize, utils.pp_option_stopwords, utils.pp_option_case_normalize]
tfidf_dataframe_name = 'tfidf_casesummarizer.json'
df_dataframe_name = 'df_casesummarizer.json'
preprocessed_suffix = '_pp.json'
dataframe_name = 'data_casesummarizer.json'
abstractive_dataframe_name = 'abstractive_data_casesummarizer.json'
leitsatz_info_filename = 'leitsatz_info.json'
param_dates = 'number_of_dates'
param_entities = 'number_of_entites'
param_section_begin = 'begin_section'
param_legal_entities = 'include_legal_entites'
param_ignore_tense = 'use_all_tenses'
param_nesting_function = 'punishment_param_for_nesting_function'
param_pos_para_function = 'punishment_param_for_position_in_paragraph_function'
param_para_number_function = 'punishment_param_for_number_of_paragraph_function'
param_branching_function = 'punishment_param_for_branching_function'
f_rank1 = 'rank_phase_one'
f_rank2 = 'rank_phase_two'
extractive = 'extr'
abstractive = 'abstr'
evaluation_filename = 'evaluation.txt'
current_path = 'leitsatz_summary'
dataframe_path = 'dataframes/case_summarizer/'
manual_path = 'manual_eval/case_summarizer/'
picture_path = 'pictures/case_summarizer/'
feature_dist_path = 'feat_dist/'
learned_func_path = 'learned_fucntions/'


def get_tfidf_dicts():
    """
    Creates or loads the tfidf and df values in dicts.

    :return: (tfidf, df) with tfidf the dict of tfidf values {aktenzeichen: {word: word_count}}
                and df the dict of df values {word: document_count}
    """
    try:
        if settings.remove_brackets:
            return utils.data_from_json(current_path=current_path,
                                        path=dataframe_path + settings.no_brackets_suffix + tfidf_dataframe_name), \
                   utils.data_from_json(current_path=current_path,
                                        path=dataframe_path + settings.no_brackets_suffix + df_dataframe_name)
        else:
            return utils.data_from_json(current_path=current_path, path=dataframe_path + tfidf_dataframe_name), \
                   utils.data_from_json(current_path=current_path, path=dataframe_path + df_dataframe_name)
    except (OSError, IOError) as _:
        all_data = download_rii.get_selected_bgh_data(case=0, directory='../data/')
        all_data = all_data[~all_data[utils.entscheidungsgruende_str].isnull()]

        if settings.remove_brackets:
            tfidf_path = dataframe_path + settings.no_brackets_suffix + tfidf_dataframe_name
            df_path = dataframe_path + settings.no_brackets_suffix + df_dataframe_name
        else:
            tfidf_path = dataframe_path + tfidf_dataframe_name
            df_path = dataframe_path + df_dataframe_name
        return utils.create_tfidf_dicts(all_data=all_data, tfidf_dataframe_path=tfidf_path, df_dataframe_path=df_path,
                                        calling_path=current_path, pp_options=pp_options)


def count_dates(input_sentence):
    """
    Counts the occurances of dates in a sentence. Finds months and their abbreviations and
    looks for dates with a regex.

    :param input_sentence: Sentence to look in
    :return: The count of occurances
    """
    count = 0
    months = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August',
              'September', 'Oktober', 'November', 'Dezember', 'Jan.', 'Feb.', 'Apr.',
              'Mär', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Okt.', 'Nov.', 'Dez.']
    for mon in months:
        count += input_sentence.count(mon)

    matches = re.findall('[0-2]?\\d\\.(0?[1-9]|1[0-2])\\.\\d{2,4}', input_sentence)
    count += len(matches)

    return count


def count_entities(input_sentence, nlp_doc_sent, param_weights, custom_path=None):
    """
    Finds the entities in the sentence and counts them

    :param input_sentence: Sentence to inspect (original))
    :param nlp_doc_sent: Sentence to inspect as nlp_doc
    :param param_weights: param weights and settings
    :return: Count of found entities
    """
    if not param_weights[param_legal_entities]:
        entities = nlp_doc_sent.ents
    else:
        if custom_path is None:
            custom_ner = utils.get_custom_ner_model()
        else:
            custom_ner = utils.get_custom_ner_model(custom_path)
        entities = custom_ner(input_sentence).ents
    return len(entities)


def check_section_beginning(input_sentence):
    """
    Checks for the beginning of the paragraphs or the document (sentence is None).
    Intended to be used on the previous sentence of a sentence (which would be the paragraph number)

    :param input_sentence: (previous) sentence to check
    :return: 1 if the sentence is at the beginning of a section or the document, 0 otherwise
    """
    # Randnummern
    if input_sentence == '' or input_sentence.strip().isdigit():
        return 1
    return 0


def get_position_distance(max_pos, current_pos):
    """
    Calculates the distance of the current position in the paragraph to the optimum

    :param max_pos: max_pos in current paragraph
    :param current_pos: current position of the sentence in the paragraph
    :return: calculated distance
    """
    if max_pos < 4:
        defined_points = [(1, 1), (3, 2.5)]
    else:
        defined_points = [(1, 2.5), (2, 2.5)]
    return get_distance_point_to_line(defined_points, (max_pos, current_pos))


def get_para_number_distance(max_para_number, current_para_number):
    """
    Calculates the distance of the current paragraph number to the optimum

    :param max_para_number: maximal paragraph number in judgement
    :param current_para_number: current number of paragraph
    :return: calculated distance
    """
    if max_para_number < 30:
        defined_points = [(27, 17), (15, 5)]
    else:
        defined_points = [(1, 5), (2, 5)]
    return get_distance_point_to_line(defined_points,
                                      (max_para_number, current_para_number))


def get_branching_distance(max_branching, current_branching):
    """
    Calculates the distance of the current branching to the optimum

    :param max_branching: maximal branching number in judgement
    :param current_branching: current branching of the sentence
    :return: calculated distance
    """
    defined_points = [(1, 2), (2, 2)]
    return get_distance_point_to_line(defined_points, (max_branching, current_branching))


def get_nesting_distance(max_nesting, current_nesting):
    """
    Calculates the distance of the current nesting to the optimum

    :param max_nesting: max_nesting of the judgmenet
    :param current_nesting: current nesting of the sentence
    :return: calculated distance
    """
    # approximate line of optimal nesting depth defined by P and Q
    defined_points = [(3, 3), (6, 4)]
    return get_distance_point_to_line(defined_points, (max_nesting, current_nesting))


def get_distance_point_to_line(points_for_line, point_for_distance, coefficients=None):
    """
    Calculates the distance of a point to a line. Only distance in y-direction
    (Original Based on :
    https://de.abcdef.wiki/wiki/Distance_from_a_point_to_a_line)


    :param points_for_line: [(,)(,)]two points to define the line
    :param point_for_distance: (x,y) point to compare
    :param coefficients: coefficients for the complex function
    :return: the distance
    """
    x, y = point_for_distance
    if coefficients is not None:
        # fucntion of degree 4
        optimal_y = coefficients[0] + coefficients[1] * x + coefficients[2] * (x ^ 2) \
                    + coefficients[3] * (x ^ 3) + coefficients[4] * (x ^ 4)
    else:
        # old code
        # absol = abs((px - qx) * (qy - y) - (qx - x) * (py - qy))
        # root = np.sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy))
        # distance = absol / root
        [(px, py), (qx, qy)] = points_for_line
        m = (py - qy) / (px - qx)
        b = py - (m * px)
        optimal_y = (m * x) + b
    return abs(y - optimal_y)


def adjust_ranking(sentence_dict, standard_deviation, param_weights):
    """
    Adjusts the rankings by number of dates and entites and the information on whether
    the sentence is at the beginning of a Randnummer

    :param sentence_dict: dict containing the features of the sentence
    :param standard_deviation: standard_deviation of all rankings
    :param param_weights for the calculations
    :return: new ranking
    """
    date_count = count_dates(sentence_dict[utils.f_original_sent])
    entity_count = count_entities(sentence_dict[utils.f_original_sent], sentence_dict[utils.f_nlp_doc_sent],
                                  param_weights=param_weights)
    begin_section = check_section_beginning(sentence_dict[utils.f_prev_sent])
    return sentence_dict[f_rank1] + standard_deviation * \
           (param_weights[param_nesting_function] *
            get_nesting_distance(sentence_dict[utils.f_max_nesting_depth], sentence_dict[utils.f_nesting_depth]) +
            param_weights[param_pos_para_function] *
            get_position_distance(sentence_dict[utils.f_max_pos_in_para], sentence_dict[utils.f_pos_in_para]) +
            param_weights[param_para_number_function] *
            get_para_number_distance(sentence_dict[utils.f_max_number_of_para], sentence_dict[utils.f_number_of_para]) +
            param_weights[param_branching_function] *
            get_branching_distance(sentence_dict[utils.f_max_ho_branch_number], sentence_dict[utils.f_ho_branch_number]) +
            param_weights[param_dates] * date_count +
            param_weights[param_entities] * entity_count +
            param_weights[param_section_begin] * begin_section)


def do_one_judgement(judgement_data):
    """
    Analyses one judgement. ranks the sentences and returns the final ranking.

    :param judgement_data: list with content [row, tfidf] row the content (dict) and tfidf the tfidf dict
    :return: [result, leitsatz, aktenzeichen, param_weights] with result being a list of tuples (pos, score, % pos) for
            each sentence.
            pos is the rank of the found sentence (0 ist the first), score the sentence score
            and %, leitsatz is already prepared with preprocessed and nlp_doc sentence
    """
    aktenzeichen, row, tfidf, param_weights = judgement_data
    row = utils.prepare_ls_entsch_gr((aktenzeichen, row), pp_options=pp_options)
    entscheidungsgruende = row[utils.entscheidungsgruende_str]

    entscheidungsgruende = utils.add_arg_struct_features(entscheidungsgruende)
    # remove sentences without present tense and misfitting arg depth
    entscheidungsgruende = [dictionary for dictionary in entscheidungsgruende
                            if (param_weights[param_ignore_tense] or
                                utils.has_present_tense(dictionary[utils.f_nlp_doc_sent]))]

    tfidf = tfidf[aktenzeichen]
    # tf*idf ranking / part 1 ranking
    ranked_sentences = [{**{f_rank1: utils.calculate_tfidf_value(sent_dict[utils.f_pp_sent], tfidf)}, **sent_dict}
                        for sent_dict in entscheidungsgruende]

    rankings = [sentence_dict[f_rank1] for sentence_dict in ranked_sentences]
    std_dev = np.std(rankings)

    # adjust ranking / part 2 ranking
    adjusted_ranked_sentences = sorted([{**{f_rank2: adjust_ranking(sent_dict, std_dev, param_weights)}, **sent_dict}
                                        for sent_dict in ranked_sentences],
                                       reverse=True, key=lambda x: x[f_rank2])

    return [adjusted_ranked_sentences, row[utils.leitsatz_str], aktenzeichen]


def get_final_summary(ranked_sentences):
    """
    Selects the sentences for the final summary from the ranked sentences. At least one sentence
    and at max 2.47% of the complete sentence count

    :param ranked_sentences: list of ranked sentences, the highest ranking at position 0
    :return: a list of sentences to be in the summary
    """
    avg_letsatz_length = 0.024714153195615134  # found out by calculating in statistics
    result = []
    for i in range(len(ranked_sentences)):
        result.append(ranked_sentences[i])  # always at least one sentence
        percentage_now = len(result) / len(ranked_sentences)
        if percentage_now < avg_letsatz_length:  # might add another sentence
            percentage_with_next_sentence = len(result) + 1 / len(ranked_sentences)
            if percentage_with_next_sentence <= avg_letsatz_length:  # percentage not reached
                continue  # just add the next sentence
            else:  # intended percentage passed
                diff_now = abs(percentage_now - avg_letsatz_length)
                diff_next = abs(percentage_with_next_sentence - avg_letsatz_length)
                if diff_now > diff_next:  # with next sentence closer to intended percentage
                    continue  # add the sentence
                else:
                    return result  # don't add and return result
        else:  # done
            return result


def evaluate_one_result(package):
    """
    Evaluates one result for one case

    :param package: (res, leitsatz, aktenzeichen, positions, print), with res the resulting ranking (sorted),
                    leitsatz the orignial leitsatz and print indicating wether these results should be printed and
                    positions indicating wether positions should be calculated
    :return: a dict with the values utils.aktenzeichen, 'position' and 'rouge'=(rouge-1,rouge-2,rouge-3,rouge-4,rouge-l)
            and 'positions' a list of tuples (position, ranked_score, percent_position)
    """
    final_ranking, leitsatz, aktenzeichen, position, print_res = package
    leitsatz_pp = ''
    leitsatz_orignal_sents = []
    for i in range(len(leitsatz)):
        sent_dict = leitsatz[i]
        leitsatz_pp += sent_dict[utils.f_pp_sent] + ' '
        leitsatz_orignal_sents.append(sent_dict[utils.f_original_sent])
    leitsatz_pp = leitsatz_pp.strip()

    result = {utils.aktenzeichen_str: aktenzeichen}
    # rouge evaluation
    resulting_summary = get_final_summary(final_ranking)
    resulting_summary_pp = ' '.join([sent_dict[utils.f_pp_sent] for sent_dict in resulting_summary])
    rouge_1 = rouge_n(leitsatz_pp, resulting_summary_pp, 1)
    rouge_2 = rouge_n(leitsatz_pp, resulting_summary_pp, 2)
    rouge_3 = rouge_n(leitsatz_pp, resulting_summary_pp, 3)
    rouge_4 = rouge_n(leitsatz_pp, resulting_summary_pp, 4)
    rouge_l_score = rouge_l(leitsatz_pp, resulting_summary_pp)
    result['rouge'] = (rouge_1, rouge_2, rouge_3, rouge_4, rouge_l_score)
    if print_res:
        print('Number of sentences in the judgement: ' + str(len(final_ranking)))
        print('Rouge -1: ' + str(rouge_1) + ' -2: ' + str(rouge_2) + ' -3: ' + str(rouge_3) + ' -4: ' + str(rouge_4)
              + ' -l :' + str(rouge_l_score))

    # original sentence placement
    leitsatz_orignal_sents = [utils.remove_brackets(sent) for sent in leitsatz_orignal_sents]
    if position:
        positions = []
        for pos in range(len(final_ranking)):
            sent_dict = final_ranking[pos]
            if utils.remove_brackets(sent_dict[utils.f_original_sent]) in leitsatz_orignal_sents:
                positions.append((pos, sent_dict[f_rank2], pos / len(final_ranking)))
                leitsatz_orignal_sents.remove(utils.remove_brackets(sent_dict[utils.f_original_sent]))
            if len(leitsatz_orignal_sents) == 0:
                break
        if len(leitsatz_orignal_sents) > 0:
            print('Not all sentences in the summary were found in the ranking: ' + str(aktenzeichen))
            print(str(leitsatz_orignal_sents))
        result['positions'] = positions
        if print_res:
            print('Found sentences (rank, score, % rank)\n' + str(positions))

    return result


def evaluate_ranking(ranking, print_results, calculate_positions):
    """
    Prints the evaluation of one ranking

    :param ranking: ranking to judge, prepared for evaluate_one_result
    :param print_results: wether individual results should be printet
    :param calculate_positions: wether positions should be calculated
    :return the calculated rouge values
    """
    evaluations = utils.parallel_imap(evaluate_one_result, ranking)
    # This creates the leitsatz-info needed for the figures, uncomment if needed
    # write_leitsatz_info(ranking)

    evaluations = [evaluation for evaluation in evaluations]

    rouges = [evaluation['rouge'] for evaluation in evaluations]

    print('Mean Rouge -1: ' + str(np.mean([rouge[0] for rouge in rouges])) +
          ' -2: ' + str(np.mean([rouge[1] for rouge in rouges])) +
          ' -3: ' + str(np.mean([rouge[2] for rouge in rouges])) +
          ' -4: ' + str(np.mean([rouge[3] for rouge in rouges])) +
          ' -l :' + str(np.mean([rouge[4] for rouge in rouges])))

    if print_results and calculate_positions:
        positions = [evaluation['positions'] for evaluation in evaluations]
        avg_first = np.mean([position[0][2] for position in positions])
        avg_last = np.mean([position[-1][2] for position in positions])
        print('Average first found sentence: ' + str(avg_first * 100) + '%')
        print('Average last found sentence: ' + str(avg_last * 100) + '%')
    return rouges


def run_casesummarizer(data, calculate_positions, print_results, param_weights):
    """
    Runs the original casesummarizer algorithm on the given data

    :param data: list of dict containing the preprocessed utils.leitsatz,
                utils.aktenzeichen and utils.entscheidungsgruende
    :param calculate_positions: indicates wether the leitsaetze are completely extractive and positions of the leitsatz
        sentences can be calculated
    :param print_results: wether the detailed evaluation results should be printed to console
    :param param_weights: weights and adjustments for the caluclation
    :return calculated_rouge
    """
    start_time = utils.get_starttime()
    tfidf_dict, df_dict = get_tfidf_dicts()

    # calculate ranking
    package = []
    for ind, row_data in data.iterrows():
        package.append([ind, row_data, tfidf_dict, param_weights])
    ranking = utils.parallel_imap(do_one_judgement, package)
    ranking = [[res, leitsatz, aktenzeichen, calculate_positions, print_results]
               for [res, leitsatz, aktenzeichen] in ranking]

    # evaluate ranking
    calc_rouge = evaluate_ranking(ranking, calculate_positions=calculate_positions, print_results=print_results)
    utils.print_duration(start_time)
    return calc_rouge


def select_params_to_df(data, params):
    keys, values = zip(*params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    final_res = pd.DataFrame()
    for param_dict in permutations_dicts:
        c_rouge = run_casesummarizer(data, False, False, param_dict)
        res = pd.concat([pd.DataFrame.from_dict([param_dict]),
                         pd.DataFrame(c_rouge,
                                      columns=['rouge-1', 'rouge-2', 'rouge-3', 'rouge-4', 'rouge-l'])
                        .mean().to_frame().T], axis=1)
        final_res = pd.concat([final_res, res], ignore_index=True)
    final_res.to_csv('Case_summm_Params.csv')
