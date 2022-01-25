import torch
import torch.nn as nn
import json
import os
import sys
import math
from sudoku_dataset import load_sudoku_data
from feature_extractor.dataset import load_data
from feature_extractor.network import MNISTNet
from ilp_config import background_knowledge, mode_declarations
from experiment_config import custom_args, process_custom_args

from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from nsl.utils import calc_example_penalty
from nsl.FastLAS import FastLASSession, FastLASSystem
from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
saved_model_path = 'feature_extractor/saved_model/model.pth'


def perform_feature_extraction(ds, net_type):
    # Load data
    _, tl = load_data(root_dir='feature_extractor', data_type=ds)

    # Instantiate network and load trained weights
    net = MNISTNet()
    network_state_dict = torch.load(saved_model_path)
    net.load_state_dict(network_state_dict)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    net.eval()

    # Initialise prediction dictionary
    predictions = {}

    # Perform forward pass on network
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tl):
            data.to(dev)
            output = net(data)
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)
            pred = softmax_output.data.max(1, keepdim=True)[1]
            confidence = softmax_output.data.max(1, keepdim=True)[0]
            predictions[str(batch_idx)+'.jpg'] = (pred[0].item()+1, confidence[0].item())

    # Save predictions to cache
    with open(cache_dir+'/digit_predictions/'+net_type+'/'+ds+'_test_set.json', 'w') as cache_out:
        cache_out.write(json.dumps(predictions))
    return predictions


def create_ILP_example(data,
                       target,
                       preds,
                       example_id,
                       net_type):

    # Build inclusion and exclusion sets
    if target.item() == 0:
        inclusion_exclusion_str = '{ }, { invalid }'
    else:
        inclusion_exclusion_str = '{ invalid }, { }'

    # Build context - get predictions for each cell
    ctx = ''
    conf_scores = []
    for idx, cell in enumerate(data):
        if cell.item() != 0:
            pred, conf = preds[str(cell.item()) + '.jpg']

            # Don't add 0 predictions
            if pred != 0:
                row_number = math.ceil((idx + 1) / 9)
                col_number = (idx + 1) % 9
                if col_number == 0:
                    col_number = 9
                ctx += '\tvalue({0},{1},{2}).\n'.format(row_number, col_number, pred)
                conf_scores.append(conf)

    if 'constant' in net_type:
        penalty = 10
    else:
        penalty = calc_example_penalty(conf_scores)

    example = '#pos(eg(id{0})@{1}, {2}, {{\n'.format(example_id, penalty, inclusion_exclusion_str)
    example += ctx
    example += '}).'

    return example


def train(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    run_feat_extrac = cmd_args['perform_feature_extraction']

    # Load follow_suit datasets
    sud_train_loaders, _ = load_sudoku_data(repeats=repeats)

    # For each network type
    for net_type in networks:

        if 'constant' in net_type:
            n = net_type.split('constant_')[1]
            net_preds = n
        elif 'without' in net_type:
            net_preds = 'softmax'
        else:
            net_preds = net_type

        # For each dataset
        for d in datasets:

            # For each noise pct
            for noise_pct in noise_pcts:
                # Only run once for standard
                if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                    break
                else:
                    print('Dataset: ' + d)
                    # Obtain feature predictions over digit image test set
                    cached_card_pred_file = d + '_test_set.json'
                    if net_type == 'softmax' and run_feat_extrac:
                        print('Running feature extraction')
                        # Perform feature extraction
                        perturbed_preds = perform_feature_extraction(d, net_preds)
                    else:
                        print('Loading neural network predictions from cache')
                        # Read from cache
                        perturbed_preds = json.loads(open(cache_dir + '/digit_predictions/' + net_preds + '/' +
                                                          cached_card_pred_file, 'r').read())

                    # Load feature predictions for non perturbed deck
                    non_perturbed_preds = json.loads(open(cache_dir + '/digit_predictions/' + net_preds + '/' +
                                                          non_perturbed_dataset+'_test_set.json', 'r').read())

                    # For each training file on sudoku data
                    for train_idx, train_loader in enumerate(sud_train_loaders):
                        out_train_idx = repeats[train_idx]

                        # If learning task does not exist in cache

                        # Standard deck don't apply noise percents
                        if d == non_perturbed_dataset:
                            cached_lt_file_name = 'train_{0}.las'.format(str(out_train_idx))
                        else:
                            cached_lt_file_name = 'train_{0}_noise_pct_{1}.las'.format(str(out_train_idx), noise_pct)
                        cached_lt_file = cache_dir+'/learning_tasks/'+net_type+'/'+d+'/'+cached_lt_file_name

                        # Iterate over follow_suit data, replace img ids with predictions
                        # and confidence scores for this deck and create an ILP example.
                        examples = []

                        if d == non_perturbed_dataset:
                            num_perturbed_examples = 0
                        else:
                            num_perturbed_examples = math.floor((noise_pct / 100) * len(train_loader))

                        for batch_idx, (data, target) in enumerate(train_loader):
                            if batch_idx < num_perturbed_examples:
                                preds = perturbed_preds
                            else:
                                preds = non_perturbed_preds
                            ilp_example = create_ILP_example(data,
                                                             target,
                                                             preds,
                                                             batch_idx,
                                                             net_type)
                            examples.append(ilp_example)

                        # Generate ILP learning task and save
                        # Adjust mode declarations for ablation
                        if net_type == 'without_block':
                            md = mode_declarations.replace('\n#modeb(block(var(cell), var(block))).', '')
                        elif net_type == 'without_block_col':
                            md = mode_declarations.replace('\n#modeb(block(var(cell), var(block))).', '')
                            md = md.replace('\n#modeb(col(var(cell), var(col))).', '')
                        elif net_type == 'without_block_col_row':
                            md = mode_declarations.replace('\n#modeb(block(var(cell), var(block))).', '')
                            md = md.replace('\n#modeb(col(var(cell), var(col))).', '')
                            md = md.replace('\n#modeb(row(var(cell), var(row))).', '')
                        else:
                            md = mode_declarations

                        ilp_sess = FastLASSession(examples=examples,
                                                  background_knowledge=background_knowledge,
                                                  mode_declarations=md)

                        with open(cached_lt_file, 'w') as lt_file:
                            lt_file.write(ilp_sess.learning_task)

                        # Load learning task from cache
                        lt = open(cached_lt_file).read()
                        ilp_sess = FastLASSession(load_from_cache=True,
                                                  cached_lt=lt)

                        # Run Learning
                        if d == non_perturbed_dataset:
                            print('Running learning task. Dataset: {0}, Train File: {1}'.
                                  format(d, str(out_train_idx)))
                        else:
                            print('Running learning task. Dataset: {0}, Train File: {1}, Noise Pct: {2}'.
                                  format(d, str(out_train_idx), noise_pct))

                        ilp_sys = FastLASSystem()
                        learned_rules, output_info = ilp_sys.run(ilp_sess)

                        # Save Rules and output_info
                        lr_dir = cache_dir + '/learned_rules/' + net_type + '/' + d
                        if d == non_perturbed_dataset:
                            rfname = lr_dir + '/train_{0}_rules.txt'.format(str(out_train_idx))
                            ifname = lr_dir + '/train_{0}_info.txt'.format(str(out_train_idx))
                        else:
                            rfname = lr_dir + '/train_{0}_noise_pct_{1}_rules.txt'.format(str(out_train_idx), noise_pct)
                            ifname = lr_dir + '/train_{0}_noise_pct_{1}_info.txt'.format(str(out_train_idx), noise_pct)

                        with open(rfname, 'w') as out:
                            out.write(learned_rules)

                        with open(ifname, 'w') as out:
                            out.write(json.dumps(output_info))

                        print('Done. Saved to: '+lr_dir)


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Sudoku 9x9 task. Run rule learning', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)

    print('Calling with command line args:')
    print(cmd_args)
    train(cmd_args)
