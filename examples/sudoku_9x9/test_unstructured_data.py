import os
import re
import json
import math
import subprocess
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from ilp_config import background_knowledge
from scipy import stats
from examples.sudoku_9x9.feature_extractor.dataset import load_data
from examples.sudoku_9x9.feature_extractor.network import MNISTNet
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/unstructured_test_data'
saved_model_path = 'feature_extractor/saved_model/model.pth'

problog_query_str = '''
valid :- \\+ invalid.
query(valid).
query(invalid).
'''


def perform_feature_extraction_for_problog(net_type, ds):
    # Load data
    _, test_loader = load_data(root_dir='feature_extractor', data_type=ds)

    # Instantiate network and load trained weights
    net = MNISTNet()
    network_state_dict = torch.load(saved_model_path)
    net.load_state_dict(network_state_dict)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    net.eval()

    # Initialise prediction dictionary
    predictions = {}

    # Perform forward pass on network
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data.to(dev)
            output = net(data)
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)

            predictions[str(batch_idx)+'.jpg'] = []
            for pred in range(9):
                # Don't add 0 confidence predictions
                if softmax_output[0][pred].item() > 0.00001:
                    predictions[str(batch_idx)+'.jpg'].append((pred+1, softmax_output[0][pred].item()))

    # Save predictions to cache
    with open(cache_dir+'/digit_predictions/'+net_type+'/'+ds+'_test_set_for_problog.json', 'w') as cache_out:
        cache_out.write(json.dumps(predictions))

    return predictions


def convert_learned_rules_to_problog(rules):
    binop_r_str = r'neq\((V\d),(V\d)\)'
    matches = re.findall(binop_r_str, rules)
    for m in matches:
        first_var = m[0]
        second_var = m[1]
        replace_str = 'neq({0},{1})'.format(first_var, second_var)
        rules = rules.replace(replace_str, '{0} != {1}'.format(first_var, second_var))
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


def convert_example_to_problog_format(row, preds, rules):
    prog = background_knowledge + '\n' + '% Learned Rules\n' + rules
    example_str = '\n\n% Example\n'
    prog = prog.replace('value', 'digit')
    board_str = row[0]
    for idx, cell in enumerate(board_str.split(' ')):
        if cell != '_':
            cell_preds = preds[str(cell)+'.jpg']
            ad_str = ''
            row_number = math.ceil((idx + 1) / 9)
            col_number = (idx + 1) % 9
            if col_number == 0:
                col_number = 9
            for pred, conf in cell_preds:
                # Prevent AD overspill with floating point error
                conf = conf - 0.000001

                # Don't add 0 prediction
                if pred != 0:
                    ad_str += '{0}::digit({1},{2},{3}); '.format(conf, row_number, col_number, pred)
            ad_str = ad_str[:-2] + '.\n'
            example_str += ad_str

    prog = prog + example_str + '\n% Query' + problog_query_str
    return prog, example_str


def test(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    run_feat_extrac = cmd_args['perform_feature_extraction']

    for net_type in networks:
        if 'constant' not in net_type:
            for d in datasets:
                print('{0}: Running Dataset: {1}'.format(net_type, d))
                dataset_results = {}

                # For each noise pct
                for noise_idx, noise_pct in enumerate(noise_pcts):
                    print('Noise pct: {0}'.format(noise_pct))
                    # Only run once for standard
                    if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                        break
                    else:
                        # Obtain feature predictions over card image deck test set
                        cached_card_pred_file = d + '_test_set_for_problog.json'
                        if net_type == 'softmax' and run_feat_extrac:
                            print('Running feature extraction')
                            # Perform feature extraction
                            perturbed_preds = perform_feature_extraction_for_problog(net_type, d)
                        else:
                            print('Loading neural network predictions from cache')
                            # Read from cache
                            perturbed_preds = json.loads(open(cache_dir + '/digit_predictions/' + net_type + '/' +
                                                              cached_card_pred_file, 'r').read())

                        # Load feature predictions for non perturbed deck
                        non_perturbed_preds = json.loads(open(cache_dir + '/digit_predictions/' + net_type + '/' +
                                                         non_perturbed_dataset+'_test_set_for_problog.json', 'r').read())

                        noise_pct_accuracy_results = []
                        noise_pct_prob_accuracy_results = []
                        csv_file = pd.read_csv('data/unstructured_data/small/test.csv')
                        for train_idx in repeats:
                            print('Split: {0}'.format(train_idx))
                            # Structured data evaluation
                            # Convert test examples into problog format and save if not already in cache
                            train_f_correct = 0
                            train_f_correct_prob = 0
                            if d == non_perturbed_dataset:
                                num_perturbed_examples = 0
                            else:
                                num_perturbed_examples = math.floor((noise_pct / 100) * len(csv_file))

                            for idx, row in enumerate(csv_file.values):
                                if idx < num_perturbed_examples:
                                    preds = perturbed_preds
                                else:
                                    preds = non_perturbed_preds

                                if d == non_perturbed_dataset:
                                    lr_file = open(
                                        cache_dir + '/learned_rules/' + net_type + '/' + d + '/train_{0}_rules.txt'.
                                        format(str(train_idx))).read()
                                else:
                                    lr_file = open(
                                        cache_dir + '/learned_rules/' + net_type + '/' + d +
                                        '/train_{0}_noise_pct_{1}_rules.txt'.format(str(train_idx), noise_pct)).read()

                                learned_rules = convert_learned_rules_to_problog(lr_file)
                                problog_program, problog_example = convert_example_to_problog_format(row, preds,
                                                                                                     learned_rules)
                                # Save to cache
                                if d == non_perturbed_dataset:
                                    file_name = cache_dir + '/test_problog_programs/' + net_type + '/' + d \
                                                + '/example_{0}_split_{1}.pl'.format(idx, train_idx)
                                else:

                                    file_name = cache_dir + '/test_problog_programs/' + net_type + '/' + d \
                                                + '/example_{0}_noise_pct_{1}_split_{2}.pl'.format(idx, noise_pct,
                                                                                                   train_idx)
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
                                # matches = re.findall(r'valid:\s+(.*)', output)
                                #
                                # if len(matches) == 2:
                                #     invalid_score = float(matches[0])
                                #     valid_score = float(matches[1])
                                #     if invalid_score > valid_score:
                                #         problog_prediction = 'invalid'
                                #     else:
                                #         problog_prediction = 'valid'
                                #
                                #     if problog_prediction == row[-1]:
                                #         train_f_correct += 1
                                #
                                #     if row[-1] == 'valid':
                                #         train_f_correct_prob += valid_score
                                #     else:
                                #         train_f_correct_prob += invalid_score
                                #
                                # elif len(matches) == 0:
                                #     print('PROBLOG ERROR. Can\'t get prediction. Output: ')
                                #     print(output)
                                #     sys.exit(1)

                            # print('Split complete.')
                            # print('Acc: ')
                            # print(train_f_correct / len(csv_file.values))
                            # print('Prob acc: ')
                            # print(train_f_correct_prob / len(csv_file.values))
                            # noise_pct_accuracy_results.append(train_f_correct / len(csv_file.values))
                            # noise_pct_prob_accuracy_results.append(train_f_correct_prob / len(csv_file.values))

                        # if d == non_perturbed_dataset:
                        #     res_key = 'noise_pct_0'
                        # else:
                        #     res_key = 'noise_pct_' + str(noise_pct)
                        # dataset_results[res_key] = {
                        #     'accuracy': {
                        #         'mean': np.mean(noise_pct_accuracy_results),
                        #         'std': np.std(noise_pct_accuracy_results),
                        #         'std_err': stats.sem(noise_pct_accuracy_results),
                        #         'raw': noise_pct_accuracy_results
                        #     },
                        #     'prob_accuracy': {
                        #         'mean': np.mean(noise_pct_prob_accuracy_results),
                        #         'std': np.std(noise_pct_prob_accuracy_results),
                        #         'std_err': stats.sem(noise_pct_prob_accuracy_results),
                        #         'raw': noise_pct_prob_accuracy_results
                        #     }
                        # }
                print('Finished Dataset: '+d+'. Results: ')
                # print(dataset_results)
                # with open(results_dir+'/'+net_type+'/'+d+'.json', 'w') as outf:
                #     outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 9x9 task. Run inference evaluation.', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    test(cmd_args)
