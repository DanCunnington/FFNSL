import os
import json
import sys
import subprocess
import pandas as pd
import numpy as np
from ilp_config import background_knowledge
from scipy import stats
from os.path import dirname, realpath

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/structured_test_data'
structured_test_csv_file = pd.read_csv('data/structured_data/small/test.csv')

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args


def convert_examples_to_clingo_format(csv_file):

    def process_example(ex_id, row):
        gt_label = row[4].split('_')[1]
        all_players = [1, 2, 3, 4]
        all_players.remove(int(gt_label))
        new_ex = 'correct({0}) :- eg({0}), winner({1}), not incorrect({0}).\n'.format(str(ex_id), gt_label)
        for p in all_players:
            new_ex += 'incorrect({0}) :- eg({0}), winner({1}).\n'.format(str(ex_id), p)
        for c in range(4):
            card = row[c]
            rank = card[:-1]
            suit = card[-1]
            new_ex += 'card({0},{1},{2}) :- eg({3}).\n'.format(str(c+1), rank, suit, ex_id)

        return new_ex

    examples = [process_example(e_id, e) for e_id, e in enumerate(csv_file.values)]

    clingo_examples = '\n'.join(examples)
    clingo_examples += '\n1 {{ eg(0..{0}) }} 1.\n'.format(len(examples)-1)
    clingo_examples += '#show correct/1.\n'
    clingo_examples += '#show incorrect/1.\n'

    return clingo_examples


def test(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    save_file_ext = cmd_args['save_file_ext']

    # For each network type
    for net_type in networks:
        for d in decks:
            print('Running Deck: '+d)
            deck_results = {}

            # For each noise pct
            for noise_idx, noise_pct in enumerate(noise_pcts):
                noise_pct_accuracy_results = []
                noise_pct_learning_time_results = []
                noise_pct_num_rules_results = []
                noise_pct_num_predicates_results = []
                # Only run once for non_perturbed_deck
                if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                    break
                else:
                    # For each train file
                    for train_idx in repeats:
                        # Structured data evaluation
                        # Convert test examples into clingo format and save
                        if d == non_perturbed_deck:
                            cached_examples_file_name = 'train_{0}_test_structured.las'.format(str(train_idx))
                        else:
                            cached_examples_file_name = 'train_{0}_noise_pct_{1}_test_structured.las'.\
                                format(str(train_idx), noise_pct)
                        cached_file_dir = cache_dir + '/test_examples_for_clingo/' + net_type + '/' + d

                        clingo_examples_structured = convert_examples_to_clingo_format(structured_test_csv_file)

                        # Save to cache
                        with open(cached_file_dir+'/'+cached_examples_file_name, 'w') as outf:
                            outf.write(clingo_examples_structured)

                        # Load learned rules and add background knowledge
                        if d == non_perturbed_deck:
                            cached_learned_rules_bk_file_name = 'train_{0}_lh_bk.las'.format(str(train_idx))
                        else:
                            cached_learned_rules_bk_file_name = 'train_{0}_noise_pct_{1}_lh_bk.las'.\
                                format(str(train_idx), noise_pct)

                        if d == non_perturbed_deck:
                            rules = open(cache_dir+'/learned_rules/'+net_type+'/'+d+'/train_{0}_rules.txt'.
                                         format(str(train_idx))).read()
                        else:
                            rules = open(cache_dir+'/learned_rules/'+net_type+'/'+d+
                                         '/train_{0}_noise_pct_{1}_rules.txt'.
                                         format(str(train_idx), noise_pct)).read()

                        # Swap invented predicates
                        rules = rules.replace('identity(winner,p1).', 'winner(X) :- p1(X).')
                        rules = rules.replace('inverse(winner,p1).', 'winner(X) :- not p1(X), player(X).')
                        complete_file = background_knowledge + '\n% Learned Hypothesis\n' + rules

                        # Save to cache
                        with open(cached_file_dir+'/'+cached_learned_rules_bk_file_name, 'w') as outf:
                            outf.write(complete_file)

                        # Evaluate using clingo
                        clingo_cmd = 'clingo --enum-mode brave --quiet=1 --outf=2 -n 0 ' + cached_file_dir + '/' +\
                                     cached_learned_rules_bk_file_name + ' ' + cached_file_dir + '/' + \
                                     cached_examples_file_name
                        result = subprocess.run(clingo_cmd,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                shell=True,
                                                executable='/bin/bash')
                        clingo_output = json.loads(result.stdout.decode())
                        results = clingo_output["Call"][0]["Witnesses"][0]["Value"]
                        num_correct = sum(s.startswith('correct') for s in results)
                        accuracy = num_correct / len(structured_test_csv_file)
                        noise_pct_accuracy_results.append(accuracy)
                        if d == non_perturbed_deck:
                            print('Split: {0}. Correct: {1}/{2}, Accuracy: {3}'.format(str(train_idx), num_correct,
                                                                                       len(results),
                                                                                       accuracy))
                        else:
                            print('Split: {0}. Noise pct: {1}. Correct: {2}/{3}, Accuracy: {4}'.
                                  format(str(train_idx), noise_pct, num_correct, len(results),
                                         accuracy))

                        # Load output info and combine learning time and interpretability for each split
                        if d == non_perturbed_deck:
                            output_info = json.loads(open(cache_dir+'/learned_rules/'+net_type+'/'+d+
                                                          '/train_{0}_info.txt'.format(str(train_idx))).read())
                        else:
                            output_info = json.loads(open(cache_dir + '/learned_rules/' + net_type + '/' + d +
                                                          '/train_{0}_noise_pct_{1}_info.txt'.
                                                          format(str(train_idx), noise_pct)).read())
                        noise_pct_learning_time_results.append(output_info['learning_time'])
                        noise_pct_num_predicates_results.append(output_info['interpretability']['total_predicates'])
                        noise_pct_num_rules_results.append(output_info['interpretability']['num_rules'])
                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_'+str(noise_pct)
                    deck_results[res_key] = {
                        'accuracy': {
                            'mean': np.mean(noise_pct_accuracy_results),
                            'std': np.std(noise_pct_accuracy_results),
                            'std_err': stats.sem(noise_pct_accuracy_results),
                            'raw': noise_pct_accuracy_results
                        },
                        'interpretability': {
                            'num_rules': {
                                'mean': np.mean(noise_pct_num_rules_results),
                                'std': np.std(noise_pct_num_rules_results),
                                'std_err': stats.sem(noise_pct_num_rules_results),
                                'raw': noise_pct_num_rules_results
                            },
                            'num_predicates': {
                                'mean': np.mean(noise_pct_num_predicates_results),
                                'std': np.std(noise_pct_num_predicates_results),
                                'std_err': stats.sem(noise_pct_num_predicates_results),
                                'raw': noise_pct_num_predicates_results
                            }
                        },
                        'learning_time': {
                            'mean': np.mean(noise_pct_learning_time_results),
                            'std': np.std(noise_pct_learning_time_results),
                            'std_err': stats.sem(np.array(noise_pct_learning_time_results)),
                            'raw': noise_pct_learning_time_results
                        }
                    }
                print('Finished Deck: '+d+'. Results: ')
                print(deck_results)
                with open(results_dir+'/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                    outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Run hypothesis testing', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    test(cmd_args)
