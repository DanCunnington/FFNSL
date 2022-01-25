import json
import math
import pandas as pd
import sys
from os.path import dirname, realpath

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


def sum_correct_cards(row, preds):
    num_correct = 0
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
        gt_card_label = cards_test_csv[cards_test_csv['img'] == str(card) + '.jpg'].iloc[0]['label']
        if gt_card_label == max_pred:
            num_correct += 1
    return num_correct


def analyse(cmd_args):
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

                        csv_file = pd.read_csv('data/unstructured_data/small/test.csv')
                        # Structured data evaluation
                        # Convert test examples into problog format and save if not already in cache
                        if d == non_perturbed_deck:
                            num_perturbed_examples = 0
                        else:
                            num_perturbed_examples = math.floor((noise_pct / 100) * len(csv_file))

                        noise_pct_correct = 0
                        for idx, row in enumerate(csv_file.values):
                            if idx < num_perturbed_examples:
                                preds = perturbed_preds
                            else:
                                preds = non_perturbed_preds

                            num_correct_cards = sum_correct_cards(row, preds)
                            noise_pct_correct += num_correct_cards

                        accuracy = noise_pct_correct / (len(csv_file)*4)
                        if d == non_perturbed_deck:
                            print('Correct: {0}/{1}, Accuracy: {2}'.format(noise_pct_correct,
                                                                           (len(csv_file)*4),
                                                                           accuracy))
                        else:
                            print('Noise pct: {0}. Correct: {1}/{2}, Accuracy: {3}'.
                                  format(noise_pct, noise_pct_correct, (len(csv_file)*4), accuracy))

                        if d == non_perturbed_deck:
                            res_key = 'noise_pct_0'
                        else:
                            res_key = 'noise_pct_' + str(noise_pct)
                        dataset_results[res_key] = {
                            'card_accuracy': accuracy
                        }
                print('Finished Dataset: ' + d + '. Results: ')
                print(dataset_results)
                with open(results_dir + '/' + net_type + '/' + d + save_file_ext + '.json', 'w') as outf:
                    outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner. Analyse neural network predictions on small test set',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    analyse(cmd_args)
