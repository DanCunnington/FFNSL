import torch
import torch.nn as nn
import json
import sys
import math

from experiment_config import custom_args, process_custom_args
from follow_suit_dataset import load_follow_suit_data
from feature_extractor.dataset import load_data
from feature_extractor.network import PlayingCardNet
from ilp_config import background_knowledge, mode_declarations
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
root_dir = dirname(parent_dir)
sys.path.append(root_dir)

from nsl.utils import calc_example_penalty
from nsl.ilasp import ILASPSession, ILASPSystem
from nsl.utils import add_cmd_line_args

# Configuration
cache_dir = 'cache'
saved_model_path = 'feature_extractor/saved_model/model.pth'


def perform_feature_extraction(nt, deck):
    # Load data
    _, test_loader = load_data(root_dir='feature_extractor', deck=deck)

    # Instantiate network and load trained weights
    net = PlayingCardNet()
    network_state_dict = torch.load(saved_model_path)
    net.load_state_dict(network_state_dict)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    net.eval()

    # Initialise prediction dictionary
    predictions = {}

    image_ids = test_loader.dataset.playing_cards
    card_mapping = test_loader.dataset.mapping

    # Perform forward pass on network
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data.to(dev)
            output = net(data)
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)
            pred = softmax_output.data.max(1, keepdim=True)[1]
            confidence = softmax_output.data.max(1, keepdim=True)[0]

            start_num_samples = test_loader.batch_size * batch_idx
            batch_image_ids = image_ids.loc[start_num_samples:start_num_samples+len(data)-1]['img'].values

            for idx, img_id in enumerate(batch_image_ids):
                predictions[img_id] = (card_mapping[pred[idx][0].item()], confidence[idx][0].item())

    # Save predictions to cache
    with open(cache_dir+'/card_predictions/'+nt+'/'+deck+'_test_set.json', 'w') as cache_out:
        cache_out.write(json.dumps(predictions))

    return predictions


def create_ILP_example(data,
                       target,
                       preds,
                       example_id,
                       possible_classes,
                       net_type):

    classes = possible_classes.copy()
    # Build inclusion and exclusion sets
    inclusion = target.item()
    classes.remove(inclusion)
    inclusion_str = 'winner({0})'.format(inclusion)

    exclusion_str = ''
    for c in classes:
        exclusion_str += 'winner({0}), '.format(c)
    exclusion_str = exclusion_str[:-2]

    # Build context - get predictions for each card
    ctx = ''
    conf_scores = []
    for idx, card in enumerate(data):
        pred, conf = preds[str(card.item())+'.jpg']
        suit = pred[-1]
        rank = pred[:-1]
        ctx += '\tcard({0},{1},{2}).\n'.format(str(idx+1), rank, suit)
        conf_scores.append(conf)

    if 'constant' in net_type:
        penalty = 10
    else:
        penalty = calc_example_penalty(conf_scores)

    example = '#pos(eg(id{0})@{1}, {{ {2} }}, {{ {3} }}, {{\n'.format(example_id, penalty, inclusion_str, exclusion_str)
    example += ctx
    example += '}).'

    return example


def train(cmd_args):
    repeats = cmd_args['repeats']
    networks = cmd_args['networks']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    run_feat_extrac = cmd_args['perform_feature_extraction']

    # Load follow_suit datasets
    fs_train_loaders, _ = load_follow_suit_data(repeats=repeats)

    # For each network type
    for net_type in networks:
        if 'constant' in net_type:
            n = net_type.split('constant_')[1]
            net_preds = n
        elif 'without' in net_type:
            net_preds = 'edl_gen'
        else:
            net_preds = net_type

        # For each deck of cards
        for d in decks:

            # For each noise pct
            for noise_pct in noise_pcts:
                # Only run once for standard
                if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                    break
                else:
                    print('Deck: ' + d)
                    # Obtain feature predictions over card image deck test set
                    cached_card_pred_file = d + '_test_set.json'
                    if net_type == 'softmax' and run_feat_extrac:
                        print('Running feature extraction')
                        # Perform feature extraction
                        perturbed_preds = perform_feature_extraction(net_preds, d)
                    else:
                        print('Loading neural network predictions from cache')
                        # Read from cache
                        perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_preds + '/' +
                                                          cached_card_pred_file, 'r').read())

                    # Load feature predictions for non perturbed deck
                    non_perturbed_preds = json.loads(open(cache_dir + '/card_predictions/' + net_preds + '/' +
                                                          non_perturbed_deck+'_test_set.json', 'r').read())

                    # For each training file on follow suit data
                    for train_idx, train_loader in enumerate(fs_train_loaders):
                        out_train_idx = repeats[train_idx]

                        # Standard deck don't apply noise percents
                        if d == non_perturbed_deck:
                            cached_lt_file_name = 'train_{0}.las'.format(str(out_train_idx))
                        else:
                            cached_lt_file_name = 'train_{0}_noise_pct_{1}.las'.format(str(out_train_idx), noise_pct)
                        cached_lt_file = cache_dir+'/learning_tasks/'+net_type+'/'+d+'/'+cached_lt_file_name

                        # Iterate over follow_suit data, replace img ids with predictions
                        # and confidence scores for this deck and create an ILP example.
                        examples = []

                        if d == non_perturbed_deck:
                            num_perturbed_examples = 0
                        else:
                            num_perturbed_examples = math.floor((noise_pct / 100) * len(train_loader))

                        for batch_idx, (data, target) in enumerate(train_loader):
                            for idx, data_item in enumerate(data):
                                if batch_idx+idx < num_perturbed_examples:
                                    preds = perturbed_preds
                                else:
                                    preds = non_perturbed_preds
                                ilp_example = create_ILP_example(data_item,
                                                                 target[idx],
                                                                 preds,
                                                                 batch_idx+idx,
                                                                 train_loader.dataset.classes,
                                                                 net_type)
                                examples.append(ilp_example)

                        # Generate ILP learning task and save
                        if net_type == 'without_rank_higher':
                            md = mode_declarations.\
                                replace('\n#modeb(1, rank_higher(var(player), var(player)), (positive)).', '')
                        elif net_type == 'without_suit':
                            md = mode_declarations.replace('#modeb(1, var(suit) != var(suit)).', '')
                            md = md.replace('#modeb(1, suit(var(player), var(suit)), (positive)).', '')
                            md = md.replace('#modeb(1, suit(const(player), var(suit)), (positive)).', '')

                        else:
                            md = mode_declarations
                        ilp_sess = ILASPSession(examples=examples,
                                                background_knowledge=background_knowledge,
                                                mode_declarations=md)
                        with open(cached_lt_file, 'w') as lt_file:
                            lt_file.write(ilp_sess.learning_task)

                        # Load learning task
                        lt = open(cached_lt_file).read()
                        ilp_sess = ILASPSession(load_from_cache=True,
                                                cached_lt=lt)

                        # Run Learning
                        if d == non_perturbed_deck:
                            print('Running learning task. Deck: {0}, Train File: {1}'.
                                  format(d, str(out_train_idx)))
                        else:
                            print('Running learning task. Deck: {0}, Train File: {1}, Noise Pct: {2}'.
                                  format(d, str(out_train_idx), noise_pct))

                        if net_type == 'softmax':
                            run_with_pylasp = True
                        else:
                            run_with_pylasp = False
                        ilp_sys = ILASPSystem(run_with_pylasp=run_with_pylasp)
                        learned_rules, output_info = ilp_sys.run(ilp_sess)

                        # Save Rules and output_info
                        lr_dir = cache_dir + '/learned_rules/' + net_type + '/' + d
                        if d == non_perturbed_deck:
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
    cmd_args = add_cmd_line_args(desc='Follow suit winner task. Run rule learning', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)

    print('Calling with command line args:')
    print(cmd_args)
    train(cmd_args)

