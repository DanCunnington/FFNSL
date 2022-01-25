from os.path import dirname, realpath
import sys
# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args
from scipy import stats
import numpy as np
import re
import json

cache_dir = './cache'
card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
               '8': 8, '9': 9, '10': 10, 'j': 11, 'q': 12, 'k': 13, 'a': 14}
lt_regex = re.compile(r'#pos\(eg\(id\d+\)@\d+,\s\{\swinner\((\d)\)\s.+\s+(card\(\d,.+\s)\s(card\(\d,.+\s)\s(card\(\d,.+\s)\s(card\(\d,.+\s)}\)\.')


def run(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    save_file_ext = cmd_args['save_file_ext']

    for net_type in networks:
        for d in decks:
            deck_results = {}
            for noise_pct in noise_pcts:
                noise_pct_results = []
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
                        correct_examples = 0
                        for ex in examples:
                            label = ex[0]
                            card_1 = ex[1]
                            card_2 = ex[2]
                            card_3 = ex[3]
                            card_4 = ex[4]

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
                                    correct_examples += 1

                        noise_pct_results.append(1 - correct_examples / len(examples))

                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    deck_results[res_key] = {
                        'pct_incorrect_examples': np.mean(noise_pct_results),
                        'std': np.std(noise_pct_results),
                        'std_err': stats.sem(np.array(noise_pct_results)),
                        'raw': noise_pct_results
                    }
                    print(d, noise_pct, np.mean(noise_pct_results))
            print('Finished deck: '+d+'. Results: ')
            print(deck_results)

            with open('results/incorrect_ILP_example_analysis/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Incorrect ILP examples',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
