import os
import re
import json
import math
import subprocess
import time
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from ilp_config import background_knowledge
from scipy import stats
from feature_extractor.dataset import load_data
from feature_extractor.network import PlayingCardNet
from experiment_config import custom_args, process_custom_args
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/unstructured_test_data'
saved_model_path = 'feature_extractor/saved_model/model.pth'

problog_query_str = '''
final_winner(1) :- winner(1), not winner(2), not winner(3), not winner(4).
final_winner(2) :- winner(2), not winner(1), not winner(3), not winner(4).
final_winner(3) :- winner(3), not winner(1), not winner(2), not winner(4).
final_winner(4) :- winner(4), not winner(1), not winner(2), not winner(3).
invalid :- not final_winner(_).

% Query
query(final_winner(1)).
query(final_winner(2)).
query(final_winner(3)).
query(final_winner(4)).
query(invalid).
'''


def perform_feature_extraction_for_problog(deck):
    # Load data
    _, test_loader = load_data(root_dir='feature_extractor', deck=deck)

    # Instantiate network and load trained weights
    net = PlayingCardNet()
    network_state_dict = torch.load(saved_model_path)
    net.load_state_dict(network_state_dict)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    net.eval()

    # Initialise prediction dictionary
    predictions = {}

    image_ids = test_loader.dataset.playing_cards
    card_mapping = test_loader.dataset.mapping

    # Perform forward pass on network
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data.to(dev)
            output = net(data)
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)

            start_num_samples = test_loader.batch_size * batch_idx
            batch_image_ids = image_ids.loc[start_num_samples:start_num_samples+len(data)-1]['img'].values

            for idx, img_id in enumerate(batch_image_ids):
                predictions[img_id] = []
                for pred in range(52):
                    # Don't add 0 confidence predictions
                    if softmax_output[idx][pred].item() > 0.00001:
                        predictions[img_id].append((card_mapping[pred], softmax_output[idx][pred].item()))

    # Save predictions to cache
    with open(cache_dir+'/card_predictions/softmax/'+deck+'_test_set_for_problog.json', 'w') as cache_out:
        cache_out.write(json.dumps(predictions))

    return predictions


def convert_learned_rules_to_problog(rules):
    rules = rules.replace('identity(winner,p1).', 'winner(X) :- p1(X).')
    rules = rules.replace('inverse(winner,p1).', 'winner(X) :- not p1(X), player(X).')
    rules = rules.replace('!= ', '\\= ')
    rules = rules.replace(';', ',')

    # Move != to end if present
    rules_split = rules.split('\n')
    new_rules = []
    for rule in rules_split:
        if '\\=' in rule:
            # Remove dot
            rule = rule.replace('.', '')
            rule_head = rule.split(':- ')[0]
            predicates = rule.split(':- ')[1].split(', ')
            to_append = []
            for p in predicates:
                if '\\=' in p:
                    to_append.append(p)
                    predicates.remove(p)
            predicates += to_append
            new_rule = rule_head + ':- ' + ', '.join(predicates) + '.'
            new_rules.append(new_rule)
        else:
            new_rules.append(rule)

    return '\n'.join(new_rules)


def convert_example_to_problog_format(row, preds, rules, net_type):
    bk = background_knowledge.replace('player(1..4).', 'player(1).\nplayer(2).\nplayer(3).\nplayer(4).')
    prog = bk + '\n' + '% Learned Rules\n' + rules
    example_str = '\n\n% Example\n'

    # For each image in the row, -1 to ignore label
    for i_idx, img in enumerate(row[:-1]):
        card_preds = preds[str(img.item())+'.jpg']
        ad_str = ''
        # if net_type == 'edl_gen':
        #     # Add top 3 predictions for each card to the ADs.
        #     card_confs = []
        #     trim = 0.01
        #     for cp in card_preds:
        #         card_confs.append(cp[1] - trim)
        #     top_3_idx = np.argsort(card_confs)[-3:]
        #
        #     for i in top_3_idx:
        #         card = card_preds[i][0]
        #         conf = card_preds[i][1] - trim
        #         suit = card[-1]
        #         rank = card[:-1]
        #         ad_str += '{0}::card({1},{2},{3}); '.format(conf, i_idx + 1, rank, suit)
        #
        # else:
        for cp in card_preds:
            card = cp[0]
            conf = cp[1] - 0.000001
            suit = card[-1]
            rank = card[:-1]
            ad_str += '{0}::card({1},{2},{3}); '.format(conf, i_idx+1, rank, suit)
        ad_str = ad_str[:-2] + '.\n'
        example_str += ad_str
    prog = prog + example_str + '\n% Query' + problog_query_str
    return prog, example_str


def test(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    run_feat_extrac = cmd_args['perform_feature_extraction']

    for net_type in networks:
        for d in decks:
            print('Running Deck: '+d)
            deck_results = {}

            # For each noise pct
            for noise_idx, noise_pct in enumerate(noise_pcts):
                # Only run once for standard
                if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                    break
                else:
                    # Obtain feature predictions over card image deck test set
                    cached_card_pred_file = d + '_test_set_for_problog.json'

                    if net_type == 'softmax' and run_feat_extrac:
                        # Perform feature extraction
                        perturbed_preds = perform_feature_extraction_for_problog(d)
                    else:
                        print('Loading neural network predictions from cache')
                        # Read from cache
                        perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_type + '/' +
                                                          cached_card_pred_file, 'r').read())

                    # Load feature predictions for non perturbed deck
                    non_perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_type + '/' +
                                                          non_perturbed_deck+'_test_set_for_problog.json', 'r').read())

                    noise_pct_accuracy_results = []
                    noise_pct_prob_accuracy_results = []
                    csv_file = pd.read_csv('data/unstructured_data/small/test.csv')
                    for train_idx in repeats:
                        print('Deck: {0}, Noise pct: {1}, Split: {2}'.format(d, noise_pct, train_idx))
                        # Structured data evaluation
                        # Convert test examples into problog format and save if not already in cache
                        train_f_correct = 0
                        train_f_correct_prob = 0
                        if d == non_perturbed_deck:
                            num_perturbed_examples = 0
                        else:
                            num_perturbed_examples = math.floor((noise_pct / 100) * len(csv_file))

                        for idx, row in enumerate(csv_file.values):
                            if idx < num_perturbed_examples:
                                preds = perturbed_preds
                            else:
                                preds = non_perturbed_preds

                            if d == non_perturbed_deck:
                                lr_file = open(
                                    cache_dir + '/learned_rules/' + net_type + '/' + d + '/train_{0}_rules.txt'.
                                    format(str(train_idx))).read()
                            else:
                                lr_file = open(
                                    cache_dir + '/learned_rules/' + net_type + '/' + d +
                                    '/train_{0}_noise_pct_{1}_rules.txt'.format(str(train_idx), noise_pct)).read()

                            learned_rules = convert_learned_rules_to_problog(lr_file)
                            problog_program, problog_example = convert_example_to_problog_format(row, preds,
                                                                                                 learned_rules,
                                                                                                 net_type)

                            # Save to cache
                            if d == non_perturbed_deck:
                                file_name = cache_dir + '/test_problog_programs/' + net_type + '/' + d \
                                            + '/example_{0}_split_{1}.pl'.format(idx, train_idx)
                            else:

                                file_name = cache_dir + '/test_problog_programs/' + net_type + '/' + d\
                                        + '/example_{0}_noise_pct_{1}_split_{2}.pl'.format(idx, noise_pct, train_idx)
                            with open(file_name, 'w') as outf:
                                outf.write(problog_program)

                            # Run problog and see if correct
                            # problog_cmd = 'problog '+file_name
                            # result = subprocess.run(problog_cmd,
                            #                         stdout=subprocess.PIPE,
                            #                         stderr=subprocess.PIPE,
                            #                         shell=True,
                            #                         executable='/bin/bash')
                            # output = result.stdout.decode()
                            # matches = re.findall(r'winner\((\d)\):\s+(.*)', output)
                            # max_score = 0
                            # problog_prediction = ''
                            #
                            # if len(matches) == 0:
                            #     print('PROBLOG ERROR. Can\'t get prediction. Output: ')
                            #     print(output)
                            # for match in matches:
                            #     player = match[0]
                            #     score = match[1]
                            #     if float(score) > max_score:
                            #         problog_prediction = int(player)
                            #         max_score = float(score)
                            #
                            # if problog_prediction == row[-1]:
                            #     train_f_correct += 1
                            #     train_f_correct_prob += max_score

                        noise_pct_accuracy_results.append(train_f_correct / len(csv_file.values))
                        noise_pct_prob_accuracy_results.append(train_f_correct_prob / len(csv_file.values))

                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    deck_results[res_key] = {
                        'accuracy': {
                            'mean': np.mean(noise_pct_accuracy_results),
                            'std': np.std(noise_pct_accuracy_results),
                            'std_err': stats.sem(noise_pct_accuracy_results),
                            'raw': noise_pct_accuracy_results
                        },
                        'prob_accuracy': {
                            'mean': np.mean(noise_pct_prob_accuracy_results),
                            'std': np.std(noise_pct_prob_accuracy_results),
                            'std_err': stats.sem(noise_pct_prob_accuracy_results),
                            'raw': noise_pct_prob_accuracy_results
                        }
                    }
            # print('Finished Deck: '+d+'. Results: ')
            # print(deck_results)
            # with open(results_dir+'/'+net_type+'/'+d+'_extra.json', 'w') as outf:
            #     outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Run inference evaluation', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    test(cmd_args)
