import numpy as np
import math
import argparse
import json
import sys
import re


def add_cmd_line_args(desc, custom_args=None):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--noise_pcts', type=str, default='[10,20,30,40,50,60,70,80,90,100]',
                        help='percentages of distributional shift to apply')
    parser.add_argument('--networks', type=str, default='["softmax","edl_gen","constant_softmax","constant_edl_gen"]',
                        help='which neural networks to use')
    parser.add_argument('--repeats', type=str, default='range(1,6)', help='how many repeats to run. Specified as'
                                                                          'range(start,end+1). Default is 5 repeats')
    parser.add_argument('--perform_feature_extraction', action='store_true', default=False,
                        help='get predictions from the neural network. If False, use the cache')
    parser.add_argument('--baseline_data_sizes', type=str, default='["small", "large"]',
                        help='which dataset sizes to use for the baselines')
    parser.add_argument('--save_file_ext', type=str, default='', help='string to append to results filename')

    if custom_args:
        for a in custom_args:
            parser.add_argument(a['flag'], type=a['type'], default=a['default'], help=a['help'])

    parsed = parser.parse_args()
    # Process arguments to convert to lists
    try:
        noise_pcts = json.loads(parsed.noise_pcts)
    except json.decoder.JSONDecodeError as e:
        print('Error decoding noise_pcts argument, stack trace:')
        print(e)
        sys.exit(1)

    try:
        networks = json.loads(parsed.networks)
    except json.decoder.JSONDecodeError as e:
        print('Error decoding networks argument, stack trace:')
        print(e)
        sys.exit(1)

    try:
        baseline_data_sizes = json.loads(parsed.baseline_data_sizes)
    except json.decoder.JSONDecodeError as e:
        print('Error decoding baseline_data_sizes argument, stack trace:')
        print(e)
        sys.exit(1)

    ptn = r'range\((\d+),(\d+)\)'
    matches = re.findall(ptn, parsed.repeats)
    if len(matches) == 0:
        print('repeats argument not passed correctly, should be a string in the format: range(start,end+1)')
        sys.exit(1)
    else:
        start = int(matches[0][0])
        end = int(matches[0][1])
        repeats = list(range(start,end))

    processed_args = {
        "noise_pcts": noise_pcts,
        "networks": networks,
        "repeats": repeats,
        "perform_feature_extraction": parsed.perform_feature_extraction,
        "save_file_ext": parsed.save_file_ext,
        "baseline_data_sizes": baseline_data_sizes
    }
    for a in custom_args:
        arg_name = a['flag'].split('--')[1]
        processed_args[arg_name] = vars(parsed)[arg_name]
    return processed_args


def luk_t_norm(a, b):
    x = a+b -1
    x = float("%0.7f" % x)
    return max(0, x)


def recursive_luk_t_norm(confs):
    if len(confs) == 1:
        return confs[0]
    else:
        return luk_t_norm(confs[0], recursive_luk_t_norm(confs[1:]))


def calc_example_penalty(conf_scores, agg_func='min'):
    if agg_func == 'min':
        agg = np.min(conf_scores)

    elif agg_func == 'prod':
        agg = np.prod(conf_scores)

    elif agg_func == 'luk':
        agg = recursive_luk_t_norm(conf_scores)

    penalty = (math.floor(agg * 100)) + 1
    return penalty
