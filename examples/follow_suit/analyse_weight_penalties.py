from experiment_config import custom_args, process_custom_args
from os.path import dirname, realpath
from scipy import stats
import numpy as np
import re
import json


import sys
# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from nsl.utils import add_cmd_line_args

cache_dir = './cache'
example_regex = r'#pos\(eg\(id\d+\)@(\d+),\s\{\swinner\((\d)\)\s.+\s+(card\(\d,.+\s)\s(card\(\d,.+\s)\s(card\(\d,.+\s)\s(card\(\d,.+\s)}\)\.'
card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
               '8': 8, '9': 9, '10': 10, 'j': 11, 'q': 12, 'k': 13, 'a': 14}


def build_example(r, p):
    # For each image in the row, -1 to ignore label
    ex = {}
    for i_idx, img in enumerate(r[:-1]):
        card_pred = p[str(img.item()) + '.jpg']
        card = card_pred[0]
        suit = card[-1]
        rank = card[:-1]
        ex['suit_c{0}'.format(i_idx+1)] = suit
        ex['val_c{0}'.format(i_idx+1)] = card_values[rank]
    ex['label'] = r[-1]
    return ex


def run(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    save_file_ext = cmd_args['save_file_ext']

    lt_regex = re.compile(example_regex)
    for net_type in networks:
        for d in decks:
            deck_results = {}
            for noise_pct in noise_pcts:
                np_correct_penalties = []
                np_correct_constant_penalties = []
                # Only run once for standard
                if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                    break
                else:
                    for train_idx in repeats:
                        # Open learning task parse examples
                        if d == non_perturbed_deck:
                            lt = open(cache_dir+'/learning_tasks/'+net_type+'/'+d+'/train_{0}.las'.format(train_idx)).read()
                        else:
                            lt = open(cache_dir + '/learning_tasks/' + net_type + '/' + d
                                      + '/train_{0}_noise_pct_{1}.las'.format(train_idx, noise_pct)).read()

                        examples = re.findall(lt_regex, lt)
                        all_penalties = 0
                        correct_penalties = 0
                        num_correct_examples = 0

                        for ex in examples:
                            penalty = ex[0]
                            all_penalties += int(penalty)
                            label = ex[1]
                            card_1 = ex[2]
                            card_2 = ex[3]
                            card_3 = ex[4]
                            card_4 = ex[5]

                            suit_c1 = card_1.split('.\n')[0].split(',')[2][0]
                            val_c1 = card_values[card_1.split('.\n')[0].split(',')[1]]
                            suit_c2 = card_2.split('.\n')[0].split(',')[2][0]
                            val_c2 = card_values[card_2.split('.\n')[0].split(',')[1]]
                            suit_c3 = card_3.split('.\n')[0].split(',')[2][0]
                            val_c3 = card_values[card_3.split('.\n')[0].split(',')[1]]
                            suit_c4 = card_4.split('.\n')[0].split(',')[2][0]
                            val_c4 = card_values[card_4.split('.\n')[0].split(',')[1]]

                            # Get cards that match suit
                            matching = [val_c1]
                            if suit_c2 == suit_c1:
                                matching.append(val_c2)
                            else:
                                matching.append(0)

                            if suit_c3 == suit_c1:
                                matching.append(val_c3)
                            else:
                                matching.append(0)

                            if suit_c4 == suit_c1:
                                matching.append(val_c4)
                            else:
                                matching.append(0)

                            # If there is one player with a higher card
                            highest_rank = max(matching)
                            if matching.count(highest_rank) == 1:
                                winning_player = matching.index(max(matching)) + 1
                                if winning_player == int(label):
                                    num_correct_examples += 1
                                    correct_penalties += int(penalty)

                        np_correct_penalties.append(correct_penalties / all_penalties)
                        np_correct_constant_penalties.append(num_correct_examples / len(examples))

                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    deck_results[res_key] = {
                        "correct": {
                            'mean_penalty_ratio': np.mean(np_correct_penalties),
                            'std': np.std(np_correct_penalties),
                            'std_err': stats.sem(np.array(np_correct_penalties)),
                            'raw': np_correct_penalties
                        },
                        "constant_correct": {
                            'mean_penalty_ratio': np.mean(np_correct_constant_penalties),
                            'std': np.std(np_correct_constant_penalties),
                            'std_err': stats.sem(np.array(np_correct_constant_penalties)),
                            'raw': np_correct_constant_penalties
                        }
                    }
            print('Finished deck: '+d+'. Results: ')
            print(deck_results)

            with open('results/weight_penalty_ratios/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. ILP example weight penalty analysis',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
