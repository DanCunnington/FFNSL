import os
import json
import sys
import re
from os.path import dirname, realpath

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/network_predictions'

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

card_regex = r'card\(\d,(.+),(.+)\).'


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
            for d in decks:
                print('Running Deck: '+d)
                deck_results = {}

                # For each noise pct
                for noise_idx, noise_pct in enumerate(noise_pcts):
                    noise_pct_rank_dist = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0,
                                                   'j': 0, 'q': 0, 'k': 0, 'a': 0}
                    noise_pct_suit_dist = {'h': 0, 's': 0, 'c': 0, 'd': 0}

                    # Only run once for non_perturbed_deck
                    if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                        break
                    else:
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
                            with open(lt_fname, 'r') as lt_file:
                                lt = lt_file.read()
                                examples = re.findall(card_regex, lt)
                                for e in examples:
                                    noise_pct_rank_dist[e[0]] += 1
                                    noise_pct_suit_dist[e[1]] += 1

                    # Normalise
                    sum_rank_dist = sum(noise_pct_rank_dist.values())
                    sum_suit_dist = sum(noise_pct_suit_dist.values())
                    for key in noise_pct_rank_dist:
                        noise_pct_rank_dist[key] = noise_pct_rank_dist[key] / sum_rank_dist

                    for key in noise_pct_suit_dist:
                        noise_pct_suit_dist[key] = noise_pct_suit_dist[key] / sum_suit_dist

                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    deck_results[res_key] = {
                        'rank_dist': noise_pct_rank_dist,
                        'suit_dist': noise_pct_suit_dist
                    }
                print('Finished Deck: '+d+'. Results: ')
                print(deck_results)
                with open(results_dir+'/'+net_type+'/'+d+save_file_ext+'.json', 'w') as outf:
                    outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Investigate rank and suit distribution',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    analyse(cmd_args)
