import os
import json
import re
import math
import sys
import subprocess
import pandas as pd
import numpy as np
from ilp_config import background_knowledge
from experiment_config import custom_args, process_custom_args
from scipy import stats
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
results_dir = 'results/nsl/structured_test_data'
structured_test_csv_file = pd.read_csv('data/structured_data/small/test.csv')


def convert_examples_to_clingo_format(csv_file):
    def process_example(ex_id, row):
        gt_label = row[1]
        board = row[0]
        if gt_label == 'valid':
            id_str = 'tp({0}) :- eg({0}), not invalid.\nfn({0}) :- eg({0}), invalid.'.format(ex_id)
        else:
            id_str = 'fp({0}) :- eg({0}), not invalid.\ntn({0}) :- eg({0}), invalid.'.format(ex_id)
        value_str = ''

        for idx, cell in enumerate(board.split(' ')):
            if cell != '0':
                row_number = math.ceil((idx + 1) / 9)
                col_number = (idx + 1) % 9
                if col_number == 0:
                    col_number = 9
                value_str = value_str + 'value("{0}, {1}", {2}) :- eg({3}).\n'.format(row_number,
                                                                                      col_number,
                                                                                      cell,
                                                                                      ex_id)
        example = id_str + '\n' + value_str
        return example

    examples = [process_example(e_id, e) for e_id, e in enumerate(csv_file.values)]
    clingo_examples = '\n'.join(examples)
    clingo_examples += '\n1 {{ eg(0..{0}) }} 1.\n'.format(len(examples)-1)
    clingo_examples += '#show tp/1.\n'
    clingo_examples += '#show fp/1.\n'
    clingo_examples += '#show tn/1.\n'
    clingo_examples += '#show fn/1.\n'

    return clingo_examples


def test(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    save_file_ext = cmd_args['save_file_ext']

    # For each network type
    for net_type in networks:
        for d in datasets:
            print('Running Dataset: '+d)
            dataset_results = {}

            # For each noise pct
            for noise_idx, noise_pct in enumerate(noise_pcts):
                noise_pct_accuracy_results = []
                noise_pct_learning_time_results = []
                noise_pct_num_rules_results = []
                noise_pct_num_predicates_results = []
                # Only run once for non_perturbed_dataset
                if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                    break
                else:
                    # For each train file
                    for train_idx in repeats:
                        # Structured data evaluation
                        # Convert test examples into clingo format and save if not already in cache
                        if d == non_perturbed_dataset:
                            cached_examples_file_name = 'train_{0}_test_structured.las'.format(str(train_idx))
                        else:
                            cached_examples_file_name = 'train_{0}_noise_pct_{1}_test_structured.las'.\
                                format(str(train_idx), noise_pct)
                        cached_file_dir = cache_dir + '/test_examples_for_clingo/' + net_type + '/' + d

                        if cached_examples_file_name not in os.listdir(cached_file_dir):
                            clingo_examples_structured = convert_examples_to_clingo_format(structured_test_csv_file)
                            # Save to cache
                            with open(cached_file_dir+'/'+cached_examples_file_name, 'w') as outf:
                                outf.write(clingo_examples_structured)

                        # Load learned rules and add background knowledge
                        if d == non_perturbed_dataset:
                            cached_learned_rules_bk_file_name = 'train_{0}_lh_bk.las'.format(str(train_idx))
                        else:
                            cached_learned_rules_bk_file_name = 'train_{0}_noise_pct_{1}_lh_bk.las'.\
                                format(str(train_idx), noise_pct)

                        if cached_learned_rules_bk_file_name not in os.listdir(cached_file_dir):
                            if d == non_perturbed_dataset:
                                rules = open(cache_dir+'/learned_rules/'+net_type+'/'+d+'/train_{0}_rules.txt'.
                                             format(str(train_idx))).read()
                            else:
                                rules = open(cache_dir+'/learned_rules/'+net_type+'/'+d+
                                             '/train_{0}_noise_pct_{1}_rules.txt'.
                                             format(str(train_idx), noise_pct)).read()

                            # Replace neq rules
                            binop_r_str = r'neq\((V\d),(V\d)\)'
                            matches = re.findall(binop_r_str, rules)
                            for m in matches:
                                first_var = m[0]
                                second_var = m[1]
                                replace_str = 'neq({0},{1})'.format(first_var, second_var)
                                rules = rules.replace(replace_str, '{0} != {1}'.format(first_var, second_var))
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
                        tp = sum(s.startswith('tp') for s in results)
                        tn = sum(s.startswith('tn') for s in results)
                        num_correct = tp + tn
                        accuracy = num_correct / len(results)
                        noise_pct_accuracy_results.append(accuracy)
                        if d == non_perturbed_dataset:
                            print('Split: {0}. Correct: {1}/{2}, Accuracy: {3}'.format(str(train_idx), num_correct,
                                                                                       len(results),
                                                                                       num_correct / len(results)))
                        else:
                            print('Split: {0}. Noise pct: {1}. Correct: {2}/{3}, Accuracy: {4}'.
                                  format(str(train_idx), noise_pct, num_correct, len(results),
                                         num_correct / len(results)))

                        # Load output info and combine learning time and interpretability for each split
                        if d == non_perturbed_dataset:
                            output_info = json.loads(open(cache_dir+'/learned_rules/'+net_type+'/'+d+
                                                          '/train_{0}_info.txt'.format(str(train_idx))).read())
                        else:
                            output_info = json.loads(open(cache_dir + '/learned_rules/' + net_type + '/' + d +
                                                          '/train_{0}_noise_pct_{1}_info.txt'.
                                                          format(str(train_idx), noise_pct)).read())
                        noise_pct_learning_time_results.append(output_info['learning_time'])
                        noise_pct_num_predicates_results.append(output_info['interpretability']['total_predicates'])
                        noise_pct_num_rules_results.append(output_info['interpretability']['num_rules'])
                    if d == non_perturbed_dataset:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_'+str(noise_pct)
                    dataset_results[res_key] = {
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
                print('Finished Dataset: '+d+'. Results: ')
                print(dataset_results)
                with open(results_dir+'/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                    outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 9x9 task. Run hypothesis testing', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    test(cmd_args)
