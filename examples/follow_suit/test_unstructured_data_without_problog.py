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

from scipy import stats
from examples.follow_suit.feature_extractor.dataset import load_data
from examples.follow_suit.feature_extractor.network import PlayingCardNet
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/unstructured_test_data/without_problog'
saved_model_path = 'feature_extractor/saved_model/model.pth'


def convert_unstruc_example_to_clingo_format(ex_id, row, preds):
    gt_label = row[4]
    all_players = [1, 2, 3, 4]
    all_players.remove(int(gt_label))
    new_ex = 'correct({0}) :- eg({0}), winner({1}), not incorrect({0}).\n'.format(str(ex_id), gt_label)
    for p in all_players:
        new_ex += 'incorrect({0}) :- eg({0}), winner({1}).\n'.format(str(ex_id), p)
    for c in range(4):
        card = row[c]
        # Get maximum neural net prediction
        image_preds = preds[str(card) + '.jpg']
        max_conf = image_preds[0][1]
        max_pred = image_preds[0][0]
        for i in image_preds:
            if i[1] > max_conf:
                max_conf = i[1]
                max_pred = i[0]
        rank = max_pred[:-1]
        suit = max_pred[-1]
        new_ex += 'card({0},{1},{2}) :- eg({3}).\n'.format(str(c+1), rank, suit, ex_id)

    return new_ex


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


def test(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    run_feat_extrac = cmd_args['perform_feature_extraction']
    save_file_ext = cmd_args['save_file_ext']

    for net_type in networks:
        if 'constant' not in net_type:
            for d in decks:
                print('{0}: Running Deck: {1}'.format(net_type, d))
                dataset_results = {}

                # For each noise pct
                for noise_idx, noise_pct in enumerate(noise_pcts):
                    print('Noise pct: {0}'.format(noise_pct))
                    # Only run once for standard
                    if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                        break
                    else:
                        net_preds = net_type
                        # Obtain feature predictions over card image deck test set
                        cached_card_pred_file = d + '_test_set_for_problog.json'
                        if net_type == 'softmax' and run_feat_extrac:
                            print('Running feature extraction')
                            # Perform feature extraction
                            perturbed_preds = perform_feature_extraction_for_problog(net_preds, d)
                        else:
                            print('Loading neural network predictions from cache')
                            # Read from cache
                            perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_preds + '/' +
                                                              cached_card_pred_file, 'r').read())

                        # Load feature predictions for non perturbed deck
                        non_perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_preds + '/' +
                                                         non_perturbed_deck+'_test_set_for_problog.json', 'r').read())

                        noise_pct_accuracy_results = []
                        csv_file = pd.read_csv('data/unstructured_data/small/test.csv')
                        for train_idx in repeats:
                            # Structured data evaluation
                            # Convert test examples into problog format and save if not already in cache
                            if d == non_perturbed_deck:
                                num_perturbed_examples = 0
                            else:
                                num_perturbed_examples = math.floor((noise_pct / 100) * len(csv_file))

                            # Load learned rules and add background knowledge
                            if d == non_perturbed_deck:
                                cached_learned_rules_bk_file_name = 'train_{0}_lh_bk.las'.format(str(train_idx))
                            else:
                                cached_learned_rules_bk_file_name = 'train_{0}_noise_pct_{1}_lh_bk.las'. \
                                    format(str(train_idx), noise_pct)

                            examples = []
                            for idx, row in enumerate(csv_file.values):
                                if idx < num_perturbed_examples:
                                    preds = perturbed_preds
                                else:
                                    preds = non_perturbed_preds

                                # Create clingo file and run evaluation
                                # Firstly, create digit facts
                                ex = convert_unstruc_example_to_clingo_format(idx, row, preds)
                                examples.append(ex)
                            clingo_examples = '\n'.join(examples)
                            clingo_examples += '\n1 {{ eg(0..{0}) }} 1.\n'.format(len(examples) - 1)
                            clingo_examples += '#show correct/1.\n'
                            clingo_examples += '#show incorrect/1.\n'

                            if d == non_perturbed_deck:
                                cached_examples_file_name = 'train_{0}_test_unstructured.las'.format(str(train_idx))
                            else:
                                cached_examples_file_name = 'train_{0}_noise_pct_{1}_test_unstructured.las'. \
                                    format(str(train_idx), noise_pct)
                            cached_file_dir = cache_dir + '/test_unstructured_examples_for_clingo/' + net_type \
                                              + '/' + d
                            cached_lr_file_dir = cache_dir + '/test_examples_for_clingo/' + net_type \
                                              + '/' + d
                            # Save to cache
                            with open(cached_file_dir + '/' + cached_examples_file_name, 'w') as outf:
                                outf.write(clingo_examples)

                            # Evaluate using clingo
                            clingo_cmd = 'clingo --enum-mode brave --quiet=1 --outf=2 -n 0 ' + cached_lr_file_dir + '/' + \
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
                            accuracy = num_correct / len(results)
                            noise_pct_accuracy_results.append(accuracy)
                            if d == non_perturbed_deck:
                                print('Split: {0}. Correct: {1}/{2}, Accuracy: {3}'.format(str(train_idx), num_correct,
                                                                                           len(results),
                                                                                           num_correct / len(results)))
                            else:
                                print('Split: {0}. Noise pct: {1}. Correct: {2}/{3}, Accuracy: {4}'.
                                      format(str(train_idx), noise_pct, num_correct, len(results),
                                             num_correct / len(results)))

                        if d == non_perturbed_deck:
                            res_key = 'noise_pct_0'
                        else:
                            res_key = 'noise_pct_' + str(noise_pct)
                        dataset_results[res_key] = {
                            'accuracy': {
                                'mean': np.mean(noise_pct_accuracy_results),
                                'std': np.std(noise_pct_accuracy_results),
                                'std_err': stats.sem(noise_pct_accuracy_results),
                                'raw': noise_pct_accuracy_results
                            }
                        }
                        print('Finished Dataset: ' + d + '. Results: ')
                        print(dataset_results)
                        with open(results_dir + '/' + net_type + '/' + d + save_file_ext + '.json', 'w') as outf:
                            outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner. Run inference evaluation without problog.',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    test(cmd_args)
