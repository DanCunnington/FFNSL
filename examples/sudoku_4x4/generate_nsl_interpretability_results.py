import os
import json
import re
import sys
import math
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/structured_test_data'


from nsl.utils import add_cmd_line_args
from examples.sudoku_4x4.experiment_config import custom_args, process_custom_args


def run(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']

    for net_type in networks:
        for d in datasets:
            print('Running Dataset: '+d)
            dataset_results = {}

            # For each noise pct
            for noise_idx, noise_pct in enumerate(noise_pcts):
                noise_pct_num_rules_results = []
                noise_pct_num_predicates_results = []
                # Only run once for non_perturbed_dataset
                if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                    break
                else:
                    # For each train file
                    for train_idx in repeats:
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

                        # Remove unnecessary != atoms
                        rules = re.sub('div_same\d\(([A-Z]\d,)+\d\) != value\(([A-Z]\d,)+[A-Z]\d\),\s', '', rules)
                        rules = re.sub('div_same\d\(([A-Z]\d,)+\d\) != div_same\d\(([A-Z]\d,)+\d\),\s', '', rules)
                        rules = re.sub('value\(([A-Z]\d,)+[A-Z]\d\) != div_same\d\(([A-Z]\d,)+\d\),\s', '', rules)
                        rules = rules.rstrip()
                        rules = [r for r in rules.split('\n') if r != '']

                        # Count num rules, num predicates

                        num_rules = len(rules)
                        total_predicates = 0
                        for rule in rules:
                            if ':- ' in rule:
                                rule = rule.split(':- ')[1]

                            total_predicates += len(rule.split(', '))

                        noise_pct_num_predicates_results.append(total_predicates)
                        noise_pct_num_rules_results.append(num_rules)

                    if d == non_perturbed_dataset:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_'+str(noise_pct)
                    dataset_results[res_key] = {
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
                        }
                    }
                print('Finished Dataset: '+d+'. Results: ')
                print(dataset_results)
                with open(results_dir+'/'+net_type+'/'+d+'_interpretability.json', 'w') as outf:
                    outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 4x4 task. Generate interpretability results', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)

