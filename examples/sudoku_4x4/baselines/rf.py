import json
import sys
import numpy as np
import pandas as pd
import time
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from scipy import stats
from subprocess import call
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
from experiment_config import custom_args, process_custom_args
from nsl.utils import add_cmd_line_args


# Configuration
cache_dir = '../cache'
data_dir = '../data'
results_dir = '../results/rf'

random_seed = 0


def brier_multi(targets, probs):
    t = np.array(targets)
    p = np.array(probs)
    return np.mean(np.sum((p - t)**2, axis=1))


def get_tree_info(tree, feature_names):
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]

    # get ids of child nodes
    idx = np.argwhere(left == -1)[:, 0]

    def recurse(l, r, c, lineage=None):
        if lineage is None:
            lineage = [c]
        if c in l:
            parent = np.where(l == c)[0].item()
            split = 'l'
        else:
            parent = np.where(r == c)[0].item()
            split = 'r'
        lineage.append((parent, split, threshold[parent], features[parent]))

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(l, r, parent, lineage)

    total_num_predicates = 0
    num_rules = 0
    for child in idx:
        for node in recurse(left, right, child):
            if type(node) == tuple:
                total_num_predicates += 1
            else:
                num_rules += 1
    return total_num_predicates, num_rules


def rf_hyp_tuning(X_train, y_train):
    # Number of trees in random forest
    n_estimators = [10, 20, 50, 100, 200]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators}
    rf = RandomForestClassifier()

    # Random search of parameters, using 3 fold cross validation
    rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, cv=3, verbose=0, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_


def create_unstructured_example(data, target, predictions):
    if target == 'valid':
        label = 0
    else:
        label = 1
    ex = []
    data = data.split(' ')
    for idx, cell in enumerate(data):
        if cell != '_':
            pred, _ = predictions[str(cell)+'.jpg']
            ex.append(pred)
        else:
            ex.append(0)
    return ex, label


def process_structured_test_examples(csv_file):
    X_test = []
    y_test = []

    def process_row(r):
        ex = []
        board_str = r[0]
        label = r[1]
        for idx, cell in enumerate(board_str.split(' ')):
            ex.append(int(cell))
        if label == 'valid':
            new_label = 0
        else:
            new_label = 1
        return ex, new_label

    for row in csv_file.values:
        x, y = process_row(row)
        X_test.append(x)
        y_test.append(y)
    return X_test, y_test


def run(cmd_args):
    repeats = cmd_args['repeats']
    noise_pcts = cmd_args['noise_pcts']
    datasets = cmd_args['datasets']
    non_perturbed_dataset = cmd_args['non_perturbed_dataset']
    baseline_data_sizes = cmd_args['baseline_data_sizes']

    # Load structured and unstructured test examples
    # Same test set both data sizes
    structured_X_test, structured_y_test = process_structured_test_examples(
        pd.read_csv(data_dir+'/structured_data/small/test.csv'))
    unstructured_test_data = pd.read_csv('../data/unstructured_data/small/test.csv')

    # Load neural network predictions
    standard_preds = json.loads(open(cache_dir+'/digit_predictions/softmax/standard_test_set.json').read())
    rotated_preds = json.loads(open(cache_dir+'/digit_predictions/softmax/rotated_test_set.json').read())

    for data_size in baseline_data_sizes:
        # Load sudoku datasets
        train_files = {}
        for tf in repeats:
            tf = str(tf)
            train_files[tf] = pd.read_csv(data_dir+'/unstructured_data/'+data_size+'/train_'+tf+'.csv')

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

                    print('Running dataset: '+d+' noise pct: '+str(noise_pct))

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

                        # Perform Hyper-parameter tuning
                        # Cache to save running every time
                        # best_params = rf_hyp_tuning(X_train_files_processed[tf], y_train_files_processed[tf])
                        # print(best_params)
                        best_params = {'n_estimators': 100}

                        # Train Model
                        rfc = RandomForestClassifier(n_estimators=best_params['n_estimators'], random_state=random_seed)
                        start_time = time.time()
                        rfc.fit(X_train_files_processed[tf], y_train_files_processed[tf])
                        finish_time = time.time()
                        noise_pct_results['learning_time'].append(finish_time - start_time)

                        # Score
                        structured_score = rfc.score(structured_X_test, structured_y_test)
                        unstructured_score = rfc.score(unstruc_X_test_examples, unstruc_y_test_examples)
                        noise_pct_results['structured_data_accuracy'].append(structured_score)
                        noise_pct_results['unstructured_data_accuracy'].append(unstructured_score)

                        # Score probabilistic accuracy
                        # unstruc_prob_score = 0
                        # rf_probs = rfc.predict_proba(unstruc_X_test_examples)
                        # for pr_idx, rf_prob in enumerate(rf_probs):
                        #     unstruc_prob_score += rf_prob[unstruc_y_test_examples[pr_idx]]
                        # unstruc_prob_acc = unstruc_prob_score / len(rf_probs)
                        # noise_pct_results['unstructured_data_prob_accuracy'].append(unstruc_prob_acc)

                        # Score brier score
                        rf_probs = rfc.predict_proba(unstruc_X_test_examples)
                        gt_one_h = []
                        for y in unstruc_y_test_examples:
                            oh = [0, 0]
                            oh[y] = 1
                            gt_one_h.append(oh)
                        noise_pct_results['unstructured_data_brier_score'].append(brier_multi(gt_one_h, rf_probs))

                        # Interpretability - Use first tree in the forest
                        columns = []
                        for c in range(16):
                            columns.append('cell_{0}'.format(c+1))
                        columns.append('label')
                        total_predicates, num_rules = get_tree_info(rfc.estimators_[0], columns)
                        noise_pct_results['interpretability']['num_predicates'].append(total_predicates)
                        noise_pct_results['interpretability']['num_rules'].append(num_rules)

                        # Save first tree
                        # tree_dir = results_dir+'/'+data_size+'/trees/'+d
                        # if d == non_perturbed_dataset:
                        #     tree_name = 'train_{0}_tree'.format(train_idx+1)
                        # else:
                        #     tree_name = 'train_{0}_noise_pct_{1}_tree'.format(train_idx+1, noise_pct)
                        #
                        # export_graphviz(rfc.estimators_[0], out_file=tree_dir+'/'+tree_name+'.dot',
                        #                 feature_names=columns[:-1],
                        #                 class_names=['valid', 'invalid'],
                        #                 rounded=True,
                        #                 proportion=False,
                        #                 precision=2,
                        #                 filled=True)
                        #
                        # # Convert to png and save
                        # call(['dot', '-Tpng', tree_dir+'/'+tree_name+'.dot', '-o',
                        #       tree_dir+'/'+tree_name+'.png', '-Gdpi=600'])
                    # Save final results
                    if d == non_perturbed_dataset:
                        res_key = 'noise_pct_0'
                    else:
                        res_key = 'noise_pct_'+str(noise_pct)
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
            print('Finished Dataset: ' + d + ', size: ' + data_size + '. Results: ')
            print(dataset_results)
            with open(results_dir + '/' + data_size + '/' + d + '.json', 'w') as outf:
                outf.write(json.dumps(dataset_results))


if __name__ == '__main__':
    cmd_args = add_cmd_line_args(desc='Random Forest Baseline Sudoku 4x4 task.', custom_args=custom_args)
    cmd_args = process_custom_args(cmd_args)

    print('Calling with command line args:')
    print(cmd_args)
    run(cmd_args)
