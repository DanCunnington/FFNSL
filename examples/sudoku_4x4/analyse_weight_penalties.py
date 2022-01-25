import sys
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from examples.sudoku_4x4.experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args
from scipy import stats
import numpy as np
import re
import json
from examples.sudoku_4x4.analyse_incorrect_ILP_examples import is_valid, NUM_ROWS, NUM_COLS

cache_dir = './cache'
example_regex = r'#pos\(eg\(id\d+\)@(\d+),\s\{.([a-z]+|)(\s}|\}),\s\{.([a-z]+|)(\s}|\}),\s\{(\s+(value\(\d.\d,\d\)\.\s+|)+)\}\)\.'

def run(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    save_file_ext = cmd_args['save_file_ext']

    lt_regex = re.compile(example_regex)
    for net_type in networks:
        for d in datasets:
            dataset_results = {}
            for noise_pct in noise_pcts:
                np_correct_penalties = []
                np_correct_constant_penalties = []
                # Only run once for standard
                if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                    break
                else:
                    for train_idx in repeats:
                        # Open learning task parse examples
                        if d == non_perturbed_dataset:
                            lt = open(cache_dir+'/learning_tasks/'+net_type+'/'+d+'/train_{0}.las'.
                                      format(train_idx)).read()
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
                            inclusion = ex[1]
                            exclusion = ex[3]
                            ctx = ex[5].replace('\n', '').replace('\t', '').split('.')
                            if inclusion == 'invalid':
                                label = 'invalid'
                            else:
                                label = 'valid'

                            # Generate board_str
                            board = ['0'] * (NUM_ROWS * NUM_COLS)
                            for val in ctx:
                                if val != '':
                                    row = int(val.split(',')[0].split('(')[1])
                                    col = int(val.split(',')[1])
                                    number = val.split(',')[2].split(')')[0]

                                    target_idx = ((row-1)*NUM_ROWS) + (col-1)
                                    board[target_idx] = number

                            board_str = ' '.join(board)

                            if is_valid(board_str) and label == 'valid':
                                correct_penalties += int(penalty)
                                num_correct_examples += 1
                            elif not is_valid(board_str) and label == 'invalid':
                                correct_penalties += int(penalty)
                                num_correct_examples += 1

                        np_correct_penalties.append(correct_penalties / all_penalties)
                        np_correct_constant_penalties.append(num_correct_examples / len(examples))

                    if d == non_perturbed_dataset:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    dataset_results[res_key] = {
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
            print('Finished dataset: '+d+'. Results: ')
            print(dataset_results)

            with open('results/weight_penalty_ratios/' + net_type + '/' + d + save_file_ext +'.json', 'w') as outf:
                outf.write(json.dumps(dataset_results))

if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 4x4 task. Run weight penalty analysis', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)