import os
import json
import sys
import re
from os.path import dirname, realpath

# Configuration
cache_dir = 'cache'
results_dir = 'results/nsl/learned_rule_breakdown'

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args


def analyse(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    save_file_ext = cmd_args['save_file_ext']
    noise_pcts = cmd_args['noise_pcts']

    for d in decks:
        if d != non_perturbed_deck:
            deck_results = {}
            for network in networks:
                for noise_pct in noise_pcts:
                    # Read in learned rules
                    np_total = 0
                    np_includes_rank_higher = 0
                    np_includes_suit = 0
                    np_includes_incorrect_rank_higher_with_suit = 0
                    np_includes_only_incorrect_rank_higher = 0
                    for repeat in repeats:
                        lr_file = '{0}/learned_rules/{1}/{2}/train_{3}_noise_pct_{4}_rules.txt'.format(cache_dir, network,
                                                                                                       d, repeat, noise_pct)
                        with open(lr_file, 'r') as lr:
                            lr = lr.read()
                            if 'rank_higher(V2,V1)' in lr:
                                np_includes_rank_higher += 1
                            if 'V2 != V3;' in lr:
                                np_includes_suit += 1

                            if 'rank_higher(V1,V2); suit(1,V3);' in lr:
                                np_includes_incorrect_rank_higher_with_suit += 1

                            if 'rank_higher(V1,V2); player(V1); player(V2).' in lr:
                                np_includes_only_incorrect_rank_higher += 1


                            np_total += 1

                    deck_results['noise_pct_{0}'.format(noise_pct)] = {
                        "correct_rank_higher": np_includes_rank_higher / np_total,
                        "correct_suit": np_includes_suit / np_total,
                        "incorrect_rank_higher_suit": np_includes_incorrect_rank_higher_with_suit / np_total,
                        "incorrect_rank_higher_only": np_includes_only_incorrect_rank_higher / np_total
                    }

                print("Network: ",network)
                print('Finished Deck: '+d+'. Results: ')
                print(deck_results)
                print('-----')
                with open(results_dir+'/'+network+'/'+d+save_file_ext+'.json', 'w') as outf:
                    outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Investigate learned rules',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    analyse(cmd_args)
