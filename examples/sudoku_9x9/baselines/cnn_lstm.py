import pandas as pd
import numpy as np
import torch.optim as optim
import json
import sys
import time
import torch
import math

from torch import nn
#from skorch import NeuralNetBinaryClassifier
from rf import get_tree_info, create_unstructured_example, process_structured_test_examples
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from scipy import stats
#from subprocess import call
from os.path import dirname, realpath
from torch.utils.data import Dataset

# Add parent directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args

cache_dir = '../cache'
data_dir = '../data'
results_dir = '../results/cnn_lstm'

num_epochs = 5

# For RMSProp
learning_rate = 0.0001
epsilon = 1e-7
log_interval = 1000

# Set random seeds
torch.manual_seed(0)
np.random.seed(0)


def brier_multi(targets, probs):
    t = np.array(targets)
    p = np.array(probs)
    return np.mean(np.sum((p - t)**2, axis=1))


# Define Network
class SudokuNet(nn.Module):
    def __init__(self, batch_size=8, num_embeddings=10, seq_len=81, embedding_dim=26,
                 cnn_out=32, lstm_hidden=20, drp=0.1):
        super(SudokuNet, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_dim
        self.seq_length = seq_len
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, out_channels=cnn_out, kernel_size=3)
        self.lstm = nn.LSTM(39, hidden_size=lstm_hidden, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(cnn_out*lstm_hidden*2, 1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(drp)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.view(x.shape[0], self.embedding_size, self.seq_length)
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x.to(torch.float32)


class ProcessedSudoku(Dataset):
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
        return x.long(), y


def load_processed_sudoku(X, y, batch_size=1):
    ds = ProcessedSudoku(X, y)
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
            batch_preds = []
            batch_gt = []
            for batch_idx, batch_item in enumerate(output):
                # Target
                if target[batch_idx].item() == 0:
                    batch_gt.append([1,0])
                else:
                    batch_gt.append([0,1])

                # Prediction
                if batch_item.item() > 0.5:
                    prediction = 1
                    batch_preds.append([0, 1])

                else:
                    prediction = 0
                    batch_preds.append([1, 0])

                if prediction == target[batch_idx].item():
                    correct += 1
            preds += batch_preds
            one_h_gt += batch_gt

        acc = (correct / len(dataset_loader.dataset))
        bs = brier_multi(one_h_gt, preds)
        return acc, bs


def run(cmd_args):
    repeats = cmd_args['repeats']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    baseline_data_sizes = cmd_args['baseline_data_sizes']

    # Load structured and unstructured test examples
    # Same test set both data sizes
    structured_X_test, structured_y_test = process_structured_test_examples(
        pd.read_csv(data_dir + '/structured_data/small/test.csv'))
    structured_test_loader = load_processed_sudoku(structured_X_test, structured_y_test)
    unstructured_test_data = pd.read_csv('../data/unstructured_data/small/test.csv')

    # Load neural network predictions
    standard_preds = json.loads(open(cache_dir + '/digit_predictions/softmax/standard_test_set.json').read())
    rotated_preds = json.loads(open(cache_dir + '/digit_predictions/softmax/rotated_test_set.json').read())

    for data_size in baseline_data_sizes:
        # Load sudoku datasets
        train_files = {}
        for tf in repeats:
            tf = str(tf)
            train_files[tf] = pd.read_csv(data_dir + '/unstructured_data/' + data_size + '/train_' + tf + '.csv')

        # For each dataset
        for d in datasets:
            dataset_results = {}

            # For each noise pct
            for noise_pct in noise_pcts:
                # Only run once for standard
                if d == non_perturbed_dataset and noise_pct > noise_pcts[0]:
                    break
                else:
                    if d == non_perturbed_dataset:
                        noise_pct = 0

                    print('Running dataset: ' + d + ' noise pct: ' + str(noise_pct))

                    # Create unstructured test sets
                    unstruc_X_test_examples = []
                    unstruc_y_test_examples = []

                    # Calculate number of perturbed test examples
                    num_perturbed_test_examples = math.floor(noise_pct / 100 * len(unstructured_test_data))
                    for idx, row in enumerate(unstructured_test_data.values):
                        if idx < num_perturbed_test_examples:
                            preds = rotated_preds
                        else:
                            preds = standard_preds
                        processed_x, processed_y = create_unstructured_example(row[0], row[1], preds)
                        unstruc_X_test_examples.append(processed_x)
                        unstruc_y_test_examples.append(processed_y)

                    # Create test loader
                    unstruc_test_loader = load_processed_sudoku(unstruc_X_test_examples, unstruc_y_test_examples)

                    # For each training file, create examples and fit rf
                    X_train_files_processed = {}
                    y_train_files_processed = {}
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
                    for tf in train_files:
                        X_train_files_processed[tf] = []
                        y_train_files_processed[tf] = []

                        # Calculate number of perturbed train examples
                        num_perturbed_train_examples = math.floor(noise_pct / 100 * len(train_files[tf]))

                        for idx, row in enumerate(train_files[tf].values):
                            if idx < num_perturbed_train_examples:
                                preds = rotated_preds
                            else:
                                preds = standard_preds
                            processed_x, processed_y = create_unstructured_example(row[0], row[1], preds)
                            X_train_files_processed[tf].append(processed_x)
                            y_train_files_processed[tf].append(processed_y)

                        # Create PyTorch train loader
                        tr_loader = load_processed_sudoku(X_train_files_processed[tf],
                                                          y_train_files_processed[tf],
                                                          batch_size=8)

                        # Perform Hyper-parameter tuning
                        # print('Running hyp tuning')
                        # hyp_tune_net = NeuralNetBinaryClassifier(
                        #     SudokuNet,
                        #     max_epochs=10,
                        #     lr=0.01,
                        #     iterator_train__shuffle=True,
                        #     batch_size=8
                        # )
                        # hyp_tune_net.set_params(train_split=False, verbose=0)
                        # params = {
                        #     'module__embedding_dim': [20, 32, 64],
                        #     'module__cnn_out': [16, 32, 64, 128],
                        #     'module__lstm_hidden': [16, 32, 64, 28],
                        #     'module__drp': [0.1, 0.2, 0.5]
                        # }
                        # print('Grid: ')
                        # print(params)
                        # gs = GridSearchCV(hyp_tune_net, params, refit=False, cv=3, scoring='accuracy', verbose=0,
                        #                   n_jobs=-1)
                        # hyp_X = train_examples_df.loc[:, train_examples_df.columns != 'label']
                        # hyp_y = train_examples_df['label']
                        #
                        # gs.fit(X_train_files_processed[tf], y_train_files_processed[tf])
                        # print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

                        # Cache best params to avoid re-tuning every time
                        best_params = {
                            'module__embedding_dim': 96,
                            'module__cnn_out': 64,
                            'module__lstm_hidden': 96,
                            'module__drp': 0.01
                        }

                        # Run Learning
                        net = SudokuNet(cnn_out=best_params['module__cnn_out'],
                                        drp=best_params['module__drp'],
                                        embedding_dim=best_params['module__embedding_dim'],
                                        lstm_hidden=best_params['module__lstm_hidden'])
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
                                output = output.squeeze(dim=1)
                                output = output.to(torch.float32)
                                lf = nn.BCELoss()
                                tr_target = tr_target.to(torch.float32)
                                loss = lf(output, tr_target)
                                loss.backward()
                                optimizer.step()
                                if batch_idx % log_interval == 0:
                                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                        epoch+1, batch_idx * len(tr_data), len(tr_loader.dataset),
                                        100. * batch_idx / len(tr_loader), loss.item()))
                        finish_time = time.time()

                        # Score
                        structured_score, _ = eval_network(net, dev, structured_test_loader)
                        print(structured_score)
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
                                    if batch_item.item() > 0.5:
                                        train_preds.append(1)
                                    else:
                                        train_preds.append(0)

                        X_train = X_train_files_processed[tf]
                        clf = DecisionTreeClassifier()
                        clf = clf.fit(X_train, train_preds)
                        columns = []
                        for c in range(81):
                            columns.append('cell_{0}'.format(c + 1))
                        columns.append('label')
                        if len(set(train_preds)) > 1:
                            total_predicates, num_rules = get_tree_info(clf, columns)
                            noise_pct_results['interpretability']['num_predicates'].append(total_predicates)
                            noise_pct_results['interpretability']['num_rules'].append(num_rules)

                            # Save tree
                            # tree_dir = results_dir+'/'+data_size+'/trees/'+d
                            # if d == non_perturbed_dataset:
                            #     tree_name = 'train_{0}_tree'.format(train_idx + 1)
                            # else:
                            #     tree_name = 'train_{0}_noise_pct_{1}_tree'.format(train_idx + 1, noise_pct)
                            # export_graphviz(clf, out_file=tree_dir + '/' + tree_name + '.dot',
                            #                 feature_names=columns[:-1],
                            #                 class_names=['valid', 'invalid'],
                            #                 rounded=True,
                            #                 proportion=False,
                            #                 precision=2,
                            #                 filled=True)
                            #
                            # # Convert to png and save
                            # call(['dot', '-Tpng', tree_dir + '/' + tree_name + '.dot', '-o',
                            #       tree_dir + '/' + tree_name + '.png', '-Gdpi=600'])
                        else:
                            print('WARNING: Network predicting same class, can\'t fit decision tree')

                    # Save final results
                    if d == non_perturbed_dataset:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_' + str(noise_pct)
                    dataset_results[res_key] = {
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
            print('Finished Dataset: ' + d + '. Results: ')
            print(dataset_results)
            with open(results_dir + '/' + data_size + '/' + d + '.json', 'w') as outf:
                outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='CNN-LSTM Baseline Sudoku 9x9 task.', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)

    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
