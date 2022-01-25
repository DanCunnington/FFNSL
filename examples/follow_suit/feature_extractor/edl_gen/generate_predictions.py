import tensorflow as tf
import sys
import numpy as np
import json
import torch
import pandas as pd
from tensorflow.saved_model import tag_constants
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
from dataset import load_data

# Load datasets
train_batch_size = 32
test_batch_size = 32
_, standard_test_loader = load_data(root_dir='../', deck='standard',
                                    train_batch_size=train_batch_size,
                                    test_batch_size=test_batch_size)

_, batman_joker_test_loader = load_data(root_dir='../', deck='batman_joker',
                                        train_batch_size=train_batch_size,
                                        test_batch_size=test_batch_size)

_, captain_america_test_loader = load_data(root_dir='../', deck='captain_america',
                                           train_batch_size=train_batch_size,
                                           test_batch_size=test_batch_size)

_, adversarial_standard_test_loader = load_data(root_dir='../', deck='adversarial_standard',
                                                train_batch_size=train_batch_size,
                                                test_batch_size=test_batch_size)

_, adversarial_batman_joker_test_loader = load_data(root_dir='../', deck='adversarial_batman_joker',
                                                    train_batch_size=train_batch_size,
                                                    test_batch_size=test_batch_size)

_, adversarial_captain_america_test_loader = load_data(root_dir='../', deck='adversarial_captain_america',
                                                       train_batch_size=train_batch_size,
                                                       test_batch_size=test_batch_size)

test_loaders = {
    "standard": standard_test_loader,
    "batman_joker": batman_joker_test_loader,
    "captain_america": captain_america_test_loader,
    "adversarial_standard": adversarial_standard_test_loader,
    "adversarial_batman_joker": adversarial_batman_joker_test_loader,
    "adversarial_captain_america": adversarial_captain_america_test_loader
}

cache_dir = '../../cache/card_predictions/edl_gen'

g2 = tf.Graph()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


with g2.as_default():
    with tf.Session(graph=g2) as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'saved_model'
        )
        for ts in test_loaders:
            preds = {}
            preds_for_problog = {}
            test_data_csv_file = pd.read_csv('../data/'+ts+'/test.csv')
            card_mapping = test_loaders[ts].dataset.mapping
            image_ids = test_loaders[ts].dataset.playing_cards

            for batch_idx, (data, target) in enumerate(test_loaders[ts]):
                X = g2.get_tensor_by_name('X:0')
                u = g2.get_tensor_by_name('uncertainty_out:0')
                prob = g2.get_tensor_by_name('prob_out:0')
                evidence = g2.get_tensor_by_name('evidence_out:0')
                flattened_data = torch.flatten(data, start_dim=1)
                feed_dict = {X: flattened_data}
                output = sess.run([u, prob, evidence], feed_dict=feed_dict)
                u = output[0]
                prob = output[1]
                evidence = output[2]

                start_num_samples = test_loaders[ts].batch_size * batch_idx
                batch_image_ids = image_ids.loc[start_num_samples:start_num_samples + len(data) - 1]['img'].values

                for idx, img_id in enumerate(batch_image_ids):
                    preds[img_id] = (card_mapping[np.argmax(prob[idx])], np.max(prob[idx]))
                    _all_preds_this_image = []
                    for pred_idx in range(52):
                        if prob[idx][pred_idx] > 0.00001:
                            _all_preds_this_image.append((card_mapping[pred_idx], prob[idx][pred_idx]))
                    preds_for_problog[img_id] = _all_preds_this_image

            print('Finished Deck: ', ts)
            # Save predictions to cache
            with open(cache_dir + '/' + ts + '_test_set.json', 'w') as cache_out:
                cache_out.write(json.dumps(preds, cls=NpEncoder))
            with open(cache_dir + '/' + ts + '_test_set_for_problog.json', 'w') as cache_out:
                cache_out.write(json.dumps(preds_for_problog, cls=NpEncoder))

