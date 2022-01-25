import os
import json
import sys
import re
import numpy as np
from os.path import dirname, realpath

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/higher_ranked_cards'

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

card_regex = r'id(\d+).+\{\swinner\((\d)\)\s},.+\s+card\((\d,.+)\).\s+card\((\d,.+)\).\s+card\((\d,.+)\).\s+card\((\d,.+)\).'

cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']


def analyse(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    save_file_ext = cmd_args['save_file_ext']

    # For each network type
    for net_type in networks:
        if 'constant' not in net_type:
            print()
            print()
            print('Running Network: ' + net_type)
            for d in decks:
                print('Running Deck: '+d)
                deck_results = {}

                # For each noise pct
                for noise_idx, noise_pct in enumerate(noise_pcts):

                    # Only run once for non_perturbed_deck
                    if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                        break
                    else:
                        repeat_results = []
                        # For each train file
                        for train_idx in repeats:
                            # Load learning task and analyse card preds
                            if d == non_perturbed_deck:
                                lt_fname = cache_dir+'/learning_tasks/{0}/{1}/train_{2}.las'.format(net_type, d,
                                                                                                    train_idx)
                            else:
                                lt_fname = cache_dir+'/learning_tasks/{0}/{1}/train_{2}_noise_pct_{3}.las'.format(net_type,
                                                                                                                  d,
                                                                                                                  train_idx,
                                                                                                                  noise_pct)
                            examples_with_gt_winner_higher = 0
                            with open(lt_fname, 'r') as lt_file:
                                lt = lt_file.read()
                                examples = re.findall(card_regex, lt)
                                for e in examples:
                                    e_id = int(e[0])
                                    gt_winner = e[1]
                                    gt_winning_card = e[int(gt_winner)+1]
                                    players = [e[2], e[3], e[4], e[5]]
                                    del players[int(gt_winner) -1]

                                    gt_winner_higher_rank = True
                                    for p in players:
                                        if cards.index(gt_winning_card.split(',')[1]) <= cards.index(p.split(',')[1]):
                                            gt_winner_higher_rank = False
                                    if gt_winner_higher_rank:
                                        examples_with_gt_winner_higher += 1

                            repeat_results.append(examples_with_gt_winner_higher/104)

                        if d == non_perturbed_deck:
                            res_key = 'noise_pct_0'
                        else:
                            res_key = 'noise_pct_' + str(noise_pct)
                        deck_results[res_key] = np.mean(repeat_results)
                print('Finished Deck: '+d+'. Results: ')
                print(deck_results)
                with open(results_dir+'/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                    outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Investigate how many examples contain a higher '
                                      'ranked card for the winning player',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    analyse(cmd_args)
