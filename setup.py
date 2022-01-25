from pathlib import Path
import os
from os.path import join, dirname

networks = ['softmax', 'edl_gen']
const_networks = ['constant_softmax', 'constant_edl_gen']
decks = ['standard', 'batman_joker', 'captain_america', 'adversarial_standard', 'adversarial_batman_joker',
         'adversarial_captain_america']
digit_datasets = ['standard', 'rotated']


def create_fs_tree(cache_path, top_name, constant=True):
    lt_path = join(cache_path, top_name)
    Path(lt_path).mkdir(parents=True, exist_ok=True)
    if constant:
        nwrks = networks + const_networks
    else:
        nwrks = networks
    for n in nwrks:
        Path(join(lt_path, n)).mkdir(parents=True, exist_ok=True)
        for d in decks:
            Path(join(join(lt_path, n), d)).mkdir(parents=True, exist_ok=True)


def create_nsl_res_dir(nsl, top_name, constant=False):
    tp = join(nsl, top_name)
    Path(tp).mkdir(parents=True, exist_ok=True)
    if constant:
        nwrks = networks + const_networks
    else:
        nwrks = networks
    for n in nwrks:
        Path(join(tp, n)).mkdir(parents=True, exist_ok=True)


def create_sudoku_dirs(sud_name):
    sud_cache_path = join('examples', join(sud_name, 'cache'))
    Path(fs_cache_path).mkdir(parents=True, exist_ok=True)
    cp_path = join(fs_cache_path, 'digit_predictions')
    Path(cp_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(cp_path, n)).mkdir(parents=True, exist_ok=True)

    create_fs_tree(sud_cache_path, 'learned_rules')
    create_fs_tree(sud_cache_path, 'learning_tasks')
    create_fs_tree(sud_cache_path, 'test_examples_for_clingo')
    create_fs_tree(sud_cache_path, 'test_unstructured_examples_for_clingo', constant=False)

    res_path = join('examples', join(sud_name, 'results'))
    Path(res_path).mkdir(parents=True, exist_ok=True)
    baselines = ['cnn_lstm', 'rf', 'rf_with_knowledge']
    for b in baselines:
        Path(join(res_path, b)).mkdir(parents=True, exist_ok=True)
        Path(join(join(res_path, b), 'small')).mkdir(parents=True, exist_ok=True)
        Path(join(join(res_path, b), 'large')).mkdir(parents=True, exist_ok=True)

    wpr_path = join(res_path, 'weight_penalty_ratios')
    Path(wpr_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(wpr_path, n)).mkdir(parents=True, exist_ok=True)

    ilp_path = join(res_path, 'incorrect_ILP_example_analysis')
    Path(ilp_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(ilp_path, n)).mkdir(parents=True, exist_ok=True)

    nsl_path = join(res_path, 'nsl')
    Path(nsl_path).mkdir(parents=True, exist_ok=True)
    create_nsl_res_dir(nsl_path, 'network_acc')
    create_nsl_res_dir(nsl_path, 'structured_test_data', constant=True)
    unstd = join(nsl_path, 'unstructured_test_data')
    Path(unstd).mkdir(parents=True, exist_ok=True)
    unstd_wp = join(unstd, 'without_problog')
    Path(unstd_wp).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(unstd_wp, n)).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    # Make directories
    Path('_tmp_ILP_working_dir').mkdir(parents=True, exist_ok=True)

    # Follow suit winner
    fs_cache_path = join('examples', join('follow_suit', 'cache'))
    Path(fs_cache_path).mkdir(parents=True, exist_ok=True)
    cp_path = join(fs_cache_path, 'card_predictions')
    Path(cp_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(cp_path, n)).mkdir(parents=True, exist_ok=True)

    create_fs_tree(fs_cache_path, 'learned_rules')
    create_fs_tree(fs_cache_path, 'learning_tasks')
    create_fs_tree(fs_cache_path, 'test_examples_for_clingo')
    create_fs_tree(fs_cache_path, 'test_unstructured_examples_for_clingo', constant=False)

    res_path = join('examples', join('follow_suit', 'results'))
    Path(res_path).mkdir(parents=True, exist_ok=True)
    baselines = ['fcn', 'rf']
    for b in baselines:
        Path(join(res_path, b)).mkdir(parents=True, exist_ok=True)
        Path(join(join(res_path, b), 'small')).mkdir(parents=True, exist_ok=True)
        Path(join(join(res_path, b), 'large')).mkdir(parents=True, exist_ok=True)

    wpr_path = join(res_path, 'weight_penalty_ratios')
    Path(wpr_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(wpr_path, n)).mkdir(parents=True, exist_ok=True)

    ilp_path = join(res_path, 'incorrect_ILP_example_analysis')
    Path(ilp_path).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(ilp_path, n)).mkdir(parents=True, exist_ok=True)

    nsl_path = join(res_path, 'nsl')
    Path(nsl_path).mkdir(parents=True, exist_ok=True)
    create_nsl_res_dir(nsl_path, 'higher_ranked_cards')
    create_nsl_res_dir(nsl_path, 'learned_rule_breakdown', constant=True)
    create_nsl_res_dir(nsl_path, 'network_acc')
    create_nsl_res_dir(nsl_path, 'network_predictions')
    create_nsl_res_dir(nsl_path, 'structured_test_data', constant=True)
    create_nsl_res_dir(nsl_path, 'higher_ranked_cards')
    unstd = join(nsl_path, 'unstructured_test_data')
    Path(unstd).mkdir(parents=True, exist_ok=True)
    unstd_wp = join(unstd, 'without_problog')
    Path(unstd_wp).mkdir(parents=True, exist_ok=True)
    for n in networks:
        Path(join(unstd_wp, n)).mkdir(parents=True, exist_ok=True)






