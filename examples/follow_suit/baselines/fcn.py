import pandas as pd
import numpy as np
import torch.optim as optim
import json
import sys
import time
import torch
import math

from torch import nn
# from skorch import NeuralNetClassifier
from rf import one_h_suits, get_tree_info, create_unstructured_example
from rf import process_structured_test_examples, cache_dir, data_dir
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from scipy import stats
from subprocess import call
from os.path import dirname, realpath
from torch.utils.data import Dataset

# Add parent directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
gparent_dir = dirname(parent_dir)
ggparent_dir = dirname(gparent_dir)
sys.path.append(ggparent_dir)
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

results_dir = '../results/fcn'

num_epochs = 50

# For RMSProp
learning_rate = 0.0001
momentum = 0
epsilon = 1e-7
alpha = 0.9
log_interval = 50

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)


def brier_multi(targets, probs):
    t = np.array(targets)
    p = np.array(probs)
    return np.mean(np.sum((p - t)**2, axis=1))


# Define Network
class FollowSuitNet(nn.Module):
    def __init__(self, input_size=20, hl_1=32, hl_2=64, drp=0.5):
        super(FollowSuitNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hl_1)
        self.fc2 = nn.Linear(hl_1, hl_2)
        self.fc3 = nn.Linear(hl_2, 4)
        self.dropout = nn.Dropout(drp)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLU(self.fc2(x))
        x = self.dropout(x)
        x = self.ReLU(self.fc3(x))
        return x


class ProcessedFollowSuit(Dataset):
    def __init__(self, X, y):
        """
        Args:
            csv_file (string): Path to the csv file with image indexes and class label annotations.
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x.float(), y-1


def load_processed_fs(X, y, batch_size=8):
    ds = ProcessedFollowSuit(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    return loader


def eval_network(net, dev, dataset_loader):
    correct = 0
    net.eval()
    preds = []
    one_h_gt = []
    with torch.no_grad():
        for data, target in dataset_loader:
            data = data.to(dev)
            target = target.to(dev)
            output = net(data)
            softmax_fn = nn.Softmax(dim=1)
            softmax_output = softmax_fn(output)
            pred = softmax_output.data.max(1, keepdim=True)[1]
            this_batch_gt = []
            preds += softmax_output.tolist()

            for batch_idx, batch_item in enumerate(softmax_output):
                # correct_prob += batch_item[target[batch_idx]].item()
                one_h = [0, 0, 0, 0]
                one_h[target[batch_idx]] = 1
                this_batch_gt.append(one_h)

            one_h_gt += this_batch_gt
            correct += pred.eq(target.data.view_as(pred)).sum()

        acc = (correct.item() / len(dataset_loader.dataset))
        bs = brier_multi(one_h_gt, preds)
        # prob_acc = (correct_prob / len(dataset_loader.dataset))
        return acc, bs


def run(cmd_args):
    repeats = cmd_args['repeats']
    noise_pcts = cmd_args['noise_pcts']
    decks = cmd_args['decks']
    non_perturbed_deck = cmd_args['non_perturbed_deck']
    baseline_data_sizes = cmd_args['baseline_data_sizes']

    # Load structured and unstructured test examples
    # Same test set both data sizes
    structured_X_test, structured_y_test = process_structured_test_examples(
        pd.read_csv(data_dir + '/structured_data/small/test.csv'))
    structured_test_loader = load_processed_fs(structured_X_test, structured_y_test)
    unstructured_test_data = pd.read_csv('../data/unstructured_data/small/test.csv')

    # Load neural network predictions
    network_predictions = {}
    for d in decks:
        network_predictions[d] = json.loads(
            open(cache_dir + '/card_predictions/softmax/' + d + '_test_set.json').read())

    for data_size in baseline_data_sizes:
        # Load follow_suit datasets
        train_files = {}
        for tf in repeats:
            tf = str(tf)
            train_files[tf] = pd.read_csv(data_dir + '/unstructured_data/' + data_size + '/train_' + tf + '.csv')

        # For each deck of cards
        for d in decks:
            deck_results = {}

            # For each noise pcts
            for noise_pct in noise_pcts:

                # Only run once for standard
                if d == non_perturbed_deck and noise_pct > noise_pcts[0]:
                    break
                else:
                    if d == non_perturbed_deck:
                        noise_pct = 0

                    print('Running deck: ' + d + ' noise pct: ' + str(noise_pct))
                    # Create unstructured test sets
                    unstruc_X_test_examples = []
                    unstruc_y_test_examples = []

                    # Calculate number of perturbed test examples
                    num_perturbed_test_examples = math.floor(noise_pct / 100 * len(unstructured_test_data))
                    for idx, row in enumerate(unstructured_test_data.values):
                        if idx < num_perturbed_test_examples:
                            preds = network_predictions[d]
                        else:
                            preds = network_predictions[non_perturbed_deck]
                        processed_x, processed_y = create_unstructured_example(row[:-1], row[len(row) - 1], preds)
                        unstruc_X_test_examples.append(processed_x)
                        unstruc_y_test_examples.append(processed_y)

                    # Create test loader
                    unstruc_test_loader = load_processed_fs(unstruc_X_test_examples, unstruc_y_test_examples,
                                                            batch_size=1)

                    # Setup noise pct results
                    noise_pct_results = {
                        'interpretability': {
                            'num_predicates': [],
                            'num_rules': []
                        },
                        'learning_time': [],
                        'structured_data_accuracy': [],
                        'unstructured_data_accuracy': [],
                        'unstructured_data_brier_score': []
                    }

                    # For each training file, create examples and fit rf
                    X_train_files_processed = {}
                    y_train_files_processed = {}
                    for tf in train_files:
                        X_train_files_processed[tf] = []
                        y_train_files_processed[tf] = []

                        # Calculate number of perturbed train examples
                        num_perturbed_train_examples = math.floor(noise_pct / 100 * len(train_files[tf]))

                        for idx, row in enumerate(train_files[tf].values):
                            if idx < num_perturbed_train_examples:
                                preds = network_predictions[d]
                            else:
                                preds = network_predictions[non_perturbed_deck]
                            processed_x, processed_y = create_unstructured_example(row[:-1], row[len(row)-1], preds)
                            X_train_files_processed[tf].append(processed_x)
                            y_train_files_processed[tf].append(processed_y)

                        # Create PyTorch train loader
                        tr_loader = load_processed_fs(X_train_files_processed[tf], y_train_files_processed[tf])

                        # Perform Hyper-parameter tuning
                        # hyp_tune_net = NeuralNetClassifier(
                        #     FollowSuitNet,
                        #     max_epochs=10,
                        #     lr=0.01,
                        #     # Shuffle training data on each epoch
                        #     iterator_train__shuffle=True,
                        # )
                        # hyp_tune_net.set_params(train_split=False, verbose=0)
                        # params = {
                        #     'module__hl_1': [20, 32, 46, 52],
                        #     'module__hl_2': [52, 64, 74, 80],
                        #     'module__drp': [0.1, 0.2, 0.5]
                        # }
                        # gs = GridSearchCV(hyp_tune_net, params, refit=False, cv=3, scoring='accuracy', verbose=2)
                        # hyp_X = train_examples_df.loc[:, train_examples_df.columns != 'label']
                        # hyp_y = train_examples_df['label']
                        # gs.fit(torch.tensor(hyp_X.values).float(), hyp_y.values)
                        # print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

                        # Cache to save running every time
                        best_params = {'drp': 0.1, 'hl_1': 20, 'hl_2': 74}

                        # Run Learning
                        net = FollowSuitNet(input_size=20, hl_1=best_params['hl_1'],
                                            hl_2=best_params['hl_2'], drp=best_params['drp'])
                        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                        net.to(dev)
                        optimizer = optim.Adam(net.parameters(), lr=learning_rate, eps=epsilon)
                        net.train()
                        start_time = time.time()
                        for epoch in range(num_epochs):
                            for batch_idx, (tr_data, tr_target) in enumerate(tr_loader):
                                optimizer.zero_grad()
                                tr_data = tr_data.to(dev)
                                tr_target = tr_target.to(dev)
                                output = net(tr_data)
                                lf = nn.CrossEntropyLoss()
                                loss = lf(output, tr_target)
                                loss.backward()
                                optimizer.step()
                                # if batch_idx % log_interval == 0:
                                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                                #         100. * batch_idx / len(train_loader), loss.item()))
                        finish_time = time.time()

                        # Score
                        structured_score, _ = eval_network(net, dev, structured_test_loader)
                        unstructured_score, unstruc_brier_score = eval_network(net, dev, unstruc_test_loader)
                        noise_pct_results['structured_data_accuracy'].append(structured_score)
                        noise_pct_results['unstructured_data_accuracy'].append(unstructured_score)
                        noise_pct_results['unstructured_data_brier_score'].append(unstruc_brier_score)
                        noise_pct_results['learning_time'].append(finish_time - start_time)

                        # Interpretability - Fit Decision tree on training set and model predictions
                        # Get predictions on training set
                        train_preds = []
                        net.eval()
                        with torch.no_grad():
                            for interp_data, _ in tr_loader:
                                interp_data = interp_data.to(dev)
                                output = net(interp_data)
                                for batch_item in output:
                                    bi = batch_item.cpu()
                                    train_preds.append(np.argmax(bi).item())

                        X_train = X_train_files_processed[tf]
                        clf = DecisionTreeClassifier()
                        clf = clf.fit(X_train, train_preds)
                        columns = []
                        for p in range(4):
                            columns.append('value_p{0}'.format(p + 1))
                            for s in one_h_suits:
                                columns.append('suit_{0}_p{1}'.format(s, p + 1))
                        columns.append('label')
                        if len(set(train_preds)) > 1:
                            total_predicates, num_rules = get_tree_info(clf, columns)
                            noise_pct_results['interpretability']['num_predicates'].append(total_predicates)
                            noise_pct_results['interpretability']['num_rules'].append(num_rules)

                            #     # Save tree
                            #     tree_dir = results_dir+'/'+data_size+'/trees/'+d
                            #     if d == non_perturbed_deck:
                            #         tree_name = 'train_{0}_tree'.format(train_idx + 1)
                            #     else:
                            #         tree_name = 'train_{0}_noise_pct_{1}_tree'.format(train_idx + 1, noise_pct)
                            #     export_graphviz(clf, out_file=tree_dir + '/' + tree_name + '.dot',
                            #                     feature_names=columns[:-1],
                            #                     class_names=['player_1', 'player_2', 'player_3', 'player_4'],
                            #                     rounded=True,
                            #                     proportion=False,
                            #                     precision=2,
                            #                     filled=True)
                            #
                            #     # Convert to png and save
                            #     call(['dot', '-Tpng', tree_dir + '/' + tree_name + '.dot', '-o',
                            #           tree_dir + '/' + tree_name + '.png', '-Gdpi=600'])
                        else:
                            print('WARNING: Network predicting same class, can\'t fit decision tree')

                    # Save final results
                    if d == non_perturbed_deck:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    deck_results[res_key] = {
                        'structured_test_accuracy': {
                            'mean': np.mean(noise_pct_results['structured_data_accuracy']),
                            'std': np.std(noise_pct_results['structured_data_accuracy']),
                            'std_err': stats.sem(noise_pct_results['structured_data_accuracy']),
                            'raw': noise_pct_results['structured_data_accuracy']
                        },
                        'unstructured_test_accuracy': {
                            'mean': np.mean(noise_pct_results['unstructured_data_accuracy']),
                            'std': np.std(noise_pct_results['unstructured_data_accuracy']),
                            'std_err': stats.sem(noise_pct_results['unstructured_data_accuracy']),
                            'raw': noise_pct_results['unstructured_data_accuracy']
                        },
                        'unstructured_test_brier_score': {
                            'mean': np.mean(noise_pct_results['unstructured_data_brier_score']),
                            'std': np.std(noise_pct_results['unstructured_data_brier_score']),
                            'std_err': stats.sem(noise_pct_results['unstructured_data_brier_score']),
                            'raw': noise_pct_results['unstructured_data_brier_score']
                        },
                        'interpretability': {
                            'num_predicates': {
                                'mean': np.mean(noise_pct_results['interpretability']['num_predicates']),
                                'std': np.std(noise_pct_results['interpretability']['num_predicates']),
                                'std_err': stats.sem(noise_pct_results['interpretability']['num_predicates']),
                                'raw': noise_pct_results['interpretability']['num_predicates']
                            },
                            'num_rules': {
                                'mean': np.mean(noise_pct_results['interpretability']['num_rules']),
                                'std': np.std(noise_pct_results['interpretability']['num_rules']),
                                'std_err': stats.sem(noise_pct_results['interpretability']['num_rules']),
                                'raw': noise_pct_results['interpretability']['num_rules']
                            }
                        },
                        'learning_time': {
                            'mean': np.mean(noise_pct_results['learning_time']),
                            'std': np.std(noise_pct_results['learning_time']),
                            'std_err': stats.sem(noise_pct_results['learning_time']),
                            'raw': noise_pct_results['learning_time']
                        }
                    }
            print('Finished Deck: ' + d + '. Results: ')
            print(deck_results)
            with open(results_dir + '/' + data_size + '/' + d + '.json', 'w') as outf:
                outf.write(json.dumps(deck_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='FCN Baseline Follow suit winner task.', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)

    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
