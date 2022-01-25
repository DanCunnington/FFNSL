# 0) To pre-train both softmax and edl_gen neural networks
```
cd feature_extractor
python train.py
cd edl_gen
python train.py
```

For EDL_GEN, generate the neural network predictions for learning and inference
`cd feature_extractor/edl_gen && python generate_predictions.py`

# 1) Accuracy and learning time for rule learning
Note if this is the first time running you will need to generate the softmax neural network predictions:
`python train.py --networks='["softmax", "edl_gen", "reduced_background_knowledge"]' --perform_feature_extraction`
Otherwise:
`python train.py --networks='["softmax", "edl_gen", "reduced_background_knowledge"]'`

# 2) Run structured test data evaluation
`python test_structured_data.py --networks='["softmax", "edl_gen", "reduced_background_knowledge"]'`

# 3) Generate modified interpretability results to account for reduced background knowledge
`python generate_nsl_interpretability_results.py --networks='["softmax", "edl_gen", "reduced_background_knowledge"]'`

# 4) Inference evaluation
Note if this is the first time running, you will need to generate the softmax neural network predictions:
`python test_unstructured_data.py --networks='["softmax", "edl_gen", "reduced_background_knowledge"]' --perform_feature_extraction`
`python test_unstructured_data_without_problog.py`

# 5) Percentage of incorrect ILP examples
`python analyse_incorrect_ILP_examples.py --networks='["softmax", "edl_gen"]'`
`python analyse_incorrect_ILP_examples.py --networks='["softmax", "edl_gen"]' --datasets='["rotated"]' --repeats='range(1,51)' --noise_pcts='[80,85,90,95,96]' --save_file_ext='_more_repeats'`
Graph handled in the jupyter notebook `common_graphs/Percentage of incorrect ILP examples.ipynb`.

# 6) 50 repeats high percentages of distributional shifts
`python train.py --repeats='range(1,51)' --datasets='["rotated"]' --noise_pcts='[80,85,90,95,96]'`

# 7) Evaluate 50 repeats
`python test_structured_data.py --repeats='range(1,51)' --datasets='["rotated"]' --noise_pcts='[80,85,90,95,96]' --save_file_ext='_more_repeats'`

# 8) ILP Example weight penalty ratios
`python analyse_weight_penalties.py --networks='["softmax", "edl_gen"]'`
`python analyse_weight_penalties.py --networks='["softmax", "edl_gen"]' --datasets='["rotated"]' --repeats='range(1,51)' --noise_pcts='[80,85,90,95,96]' --save_file_ext='_more_repeats'`

# 9) Baselines
`cd baselines`
`python rf.py`
`python cnn_lstm.py`

# 10) To generate accuracy and confidence score analysis
`python analyse_network_accuracy_and_confidence_under_dist_shift_train_data.py`
`python analyse_network_accuracy_under_dist_shift_test_data.py`

# 11) Graphs
All handled with jupyter notebooks located within `paper_results/graphs`
