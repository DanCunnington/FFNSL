# 0) To pre-train both softmax and edl_gen neural networks
```
cd feature_extractor
python train.py
cd edl_gen
python train.py
```

For EDL_GEN, generate the neural network predictions for learning and inference
`cd feature_extractor/edl_gen && python generate_predictions.py`

# 1) Rule learning
Note if this is the first time running you will need to generate the softmax neural network predictions:
`python train.py --networks='["softmax", "edl_gen"]' --perform_feature_extraction`
Otherwise:
`python train.py --networks='["softmax", "edl_gen"]'`

# 2) Run structured test data evaluation - Accuracy, interpretability and learning time
`python test_structured_data.py --networks='["softmax", "edl_gen"]'`

# 3) Inference evaluation
Note if this is the first time running, you will need to generate the softmax neural network predictions:
`python test_unstructured_data.py --networks='["softmax", "edl_gen"]' --perform_feature_extraction`
`python test_unstructured_data_without_problog.py`

# 4) Percentage of incorrect ILP examples
`python analyse_incorrect_ILP_examples.py --networks='["softmax", "edl_gen"]'`
`python analyse_incorrect_ILP_examples.py --networks='["softmax", "edl_gen"]' --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]' --save_file_ext='_more_repeats'`
Graph handled in the jupyter notebook `common_graphs/Percentage of incorrect ILP examples.ipynb`.

# 5) 50 repeats high percentages of distributional shifts
`python train.py --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]'`

# 6) Evaluate 50 repeats
`python test_structured_data.py --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]' --save_file_ext='_more_repeats'`

# 7) ILP Example weight penalty ratios
`python analyse_weight_penalties.py --networks='["softmax", "edl_gen"]'`
`python analyse_weight_penalties.py --networks='["softmax", "edl_gen"]' --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]' --save_file_ext='_more_repeats'`

# 8) Baselines
`cd baselines`
`python rf.py`
`python fcn.py`

# 9) To generate accuracy and confidence score analysis
`python analyse_network_accuracy_and_confidence_under_dist_shift_train_data.py`
`python analyse_network_accuracy_under_dist_shift_test_data.py`

# 10) Graphs
All handled with jupyter notebooks located within `paper_results/graphs`


# 11) Optional - Further Analysis
 * Investigate the learned rules and return the percentage of correct `rank_higher` and `suit` rules:
    
    `python investigate_learned_rules.py --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]' --save_file_ext='_more_repeats'`
 
 * Return the percentage of ILP examples that contain a higher ranked card for the ground-truth winning player:
 
    `python investigate_higher_rank_cards.py --repeats='range(1,51)' --noise_pcts='[95,96,97,98,99,100]' --save_file_ext='_more_repeats'`
    