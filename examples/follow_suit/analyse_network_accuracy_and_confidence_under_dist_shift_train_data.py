import json
import math
import pandas as pd
import sys
from os.path import dirname, realpath
import numpy as np
from scipy import stats

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from examples.follow_suit.test_unstructured_data import perform_feature_extraction_for_problog
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/network_acc/'
saved_model_path = 'feature_extractor/saved_model/model.pth'
cards_test_csv = pd.read_csv('feature_extractor/data/standard/test.csv')


def sum_correct_cards_and_get_conf_scores(row, preds):
    num_correct = 0
    confs = []
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

        confs.append(max_conf)
        # Check if correct
        gt_card_label = cards_test_csv[cards_test_csv['img'] == str(card) + '.jpg'].iloc[0]['label']
        if gt_card_label == max_pred:
            num_correct += 1
    return num_correct, confs


def analyse(cmd_args):
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

                        noise_pct_results = []
                        noise_pct_conf_results = []
                        for train_idx in repeats:
                            repeat_correct = 0
                            csv_file = pd.read_csv('data/unstructured_data/small/train_{0}.csv'.format(train_idx))

                            if d == non_perturbed_deck:
                                num_perturbed_examples = 0
                            else:
                                num_perturbed_examples = math.floor((noise_pct / 100) * len(csv_file))

                            repeat_confidence_scores = {
                                ">0%": 0,
                                ">25%": 0,
                                ">50%": 0,
                                ">75%": 0,
                                ">90%": 0,
                                ">95%": 0
                            }
                            for idx, row in enumerate(csv_file.values):
                                if idx < num_perturbed_examples:
                                    preds = perturbed_preds
                                else:
                                    preds = non_perturbed_preds

                                num_correct_cards, conf_scores = sum_correct_cards_and_get_conf_scores(row, preds)
                                repeat_correct += num_correct_cards
                                for c in conf_scores:
                                    if c > 0.95:
                                        repeat_confidence_scores[">95%"] += 1
                                    elif c > 0.9:
                                        repeat_confidence_scores[">90%"] += 1
                                    elif c > 0.75:
                                        repeat_confidence_scores[">75%"] += 1
                                    elif c > 0.5:
                                        repeat_confidence_scores[">50%"] += 1
                                    elif c > 0.25:
                                        repeat_confidence_scores[">25%"] += 1
                                    else:
                                        repeat_confidence_scores[">0%"] += 1

                            accuracy = repeat_correct / (len(csv_file)*4)
                            noise_pct_results.append(accuracy)
                            noise_pct_conf_results.append(repeat_confidence_scores)
                            if d == non_perturbed_deck:
                                print('Split: {0}. Correct: {1}/{2}, Accuracy: {3}'.format(str(train_idx),
                                                                                           repeat_correct,
                                                                                           (len(csv_file)*4),
                                                                                           accuracy))
                            else:
                                print('Split: {0}. Noise pct: {1}. Correct: {2}/{3}, Accuracy: {4}'.
                                      format(str(train_idx), noise_pct, repeat_correct, (len(csv_file)*4),
                                             accuracy))

                        if d == non_perturbed_deck:
                            res_key = 'noise_pct_0'
                        else:
                            res_key = 'noise_pct_' + str(noise_pct)
                        dataset_results[res_key] = {}
                        dataset_results[res_key]['card_accuracy'] = {
                            'mean': np.mean(noise_pct_results),
                            'std': np.std(noise_pct_results),
                            'std_err': stats.sem(noise_pct_results),
                            'raw': noise_pct_results
                        }
                        conf_totals = {
                            ">0%": 0,
                            ">25%": 0,
                            ">50%": 0,
                            ">75%": 0,
                            ">90%": 0,
                            ">95%": 0
                        }
                        for rcs in noise_pct_conf_results:
                            conf_totals['>0%'] += rcs['>0%']
                            conf_totals['>25%'] += rcs['>25%']
                            conf_totals['>50%'] += rcs['>50%']
                            conf_totals['>75%'] += rcs['>75%']
                            conf_totals['>90%'] += rcs['>90%']
                            conf_totals['>95%'] += rcs['>95%']
                        npcd_len = sum(conf_totals.values())
                        dataset_results[res_key]['card_confidence_dist'] = {
                            ">0%": conf_totals['>0%'] / npcd_len,
                            ">25%": conf_totals['>25%'] / npcd_len,
                            ">50%": conf_totals['>50%'] / npcd_len,
                            ">75%": conf_totals['>75%'] / npcd_len,
                            ">90%": conf_totals['>90%'] / npcd_len,
                            ">95%": conf_totals['>95%'] / npcd_len,
                        }

                print('Finished Dataset: ' + d + '. Results: ')
                print(dataset_results)
                with open(results_dir + '/' + net_type + '/' + d + '_train' + save_file_ext + '.json', 'w') as outf:
                    outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner. Analyse neural network predictions and confidence scores'
                                      'on train sets',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    analyse(cmd_args)
