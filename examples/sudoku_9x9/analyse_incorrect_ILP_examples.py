from scipy import stats
import numpy as np
import re
import json
import sys

from experiment_config import custom_args, process_custom_args
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from nsl.utils import add_cmd_line_args

cache_dir = './cache'


# Test valid/invalid board gt
NUM_ROWS = 9
NUM_COLS = 9

blocks = {
        '1': {
            'rows': [0,1,2],
            'cols': [0,1,2]
        },
        '2': {
            'rows': [0,1,2],
            'cols': [3,4,5]
        },
        '3': {
            'rows': [0,1,2],
            'cols': [6,7,8]
        },
        '4': {
            'rows': [3,4,5],
            'cols': [0,1,2]
        },
        '5': {
            'rows': [3,4,5],
            'cols': [3,4,5]
        },
        '6': {
            'rows': [3,4,5],
            'cols': [6,7,8]
        },
        '7': {
            'rows': [6,7,8],
            'cols': [0,1,2]
        },
        '8': {
            'rows': [6,7,8],
            'cols': [3,4,5]
        },
        '9': {
            'rows': [6,7,8],
            'cols': [6,7,8]
        }
    }


def is_valid(bs):
    cells = bs.split(" ")
    rows = np.array_split(np.array(cells), NUM_ROWS)
    # Check no two values exist in the same row
    for r in rows:
        trimmed = list(filter(('0').__ne__, r))
        dups = len(trimmed) != len(set(trimmed))
        if dups:
            return False

    # Check no two values exist in the same column
    cols = []
    for i in range(NUM_COLS):
        col = []
        for r in rows:
            col.append(r[i])
        cols.append(col)

    for c in cols:
        trimmed = list(filter(('0').__ne__, c))
        dups = len(trimmed) != len(set(trimmed))
        if dups:
            return False

    # Check no two values exist in the same block
    cells_blocks = []
    for b in blocks:
        this_block = []
        for r in blocks[b]['rows']:
            for c in blocks[b]['cols']:
                this_block.append(rows[r][c])
        cells_blocks.append(this_block)

    for cbl in cells_blocks:
        trimmed = list(filter(('0').__ne__, cbl))
        dups = len(trimmed) != len(set(trimmed))
        if dups:
            return False

    return True


def run(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']

    lt_regex = re.compile(r'#pos\(eg\(id\d+\)@\d+,\s\{.([a-z]+|)(\s}|\}),\s\{.([a-z]+|)(\s}|\}),\s\{(\s+(value\(\d.\d,\d\)\.\s+|)+)\}\)\.')
    for net_type in networks:
        for d in datasets:
            deck_results = {}
            for noise_pct in noise_pcts:
                noise_pct_results = []
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
                        correct_examples = 0
                        perturbed_to_valid = 0
                        perturbed_to_invalid = 0
                        remaining_valid = 0
                        remaining_invalid = 0
                        start_valid = 0
                        start_invalid = 0
                        for ex in examples:
                            inclusion = ex[0]
                            exclusion = ex[2]
                            ctx = ex[4].replace('\n', '').replace('\t', '').split('.')
                            if inclusion == 'invalid':
                                label = 'invalid'
                            else:
                                label = 'valid'

                            # Generate board_str
                            board = ['0']*81
                            for val in ctx:
                                if val != '':
                                    row = int(val.split(',')[0].split('(')[1])
                                    col = int(val.split(',')[1])
                                    number = val.split(',')[2].split(')')[0]

                                    target_idx = ((row-1)*9) + (col-1)
                                    board[target_idx] = number

                            board_str = ' '.join(board)

                            if is_valid(board_str) and label == 'valid':
                                correct_examples += 1
                                remaining_valid += 1
                            elif not is_valid(board_str) and label == 'invalid':
                                correct_examples += 1
                                remaining_invalid += 1
                            elif is_valid(board_str) and label == 'invalid':
                                perturbed_to_valid += 1
                            elif not is_valid(board_str) and label == 'valid':
                                perturbed_to_invalid += 1

                            if label == 'valid':
                                start_valid += 1
                            elif label == 'invalid':
                                start_invalid += 1

                        noise_pct_results.append(1 - correct_examples / len(examples))

                    if d == non_perturbed_dataset:
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
            print('Finished dataset: '+d+'. Results: ')
            print(deck_results)

            with open('results/incorrect_ILP_example_analysis/'+net_type+'/'+d+'_more_repeats.json', 'w') as outf:
                outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 9x9 task. Incorrect ILP examples',
                                 custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)
    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
