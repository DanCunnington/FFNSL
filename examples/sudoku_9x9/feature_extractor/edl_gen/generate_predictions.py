import tensorflow as tf
import sys
import numpy as np
import json
from tensorflow.saved_model import tag_constants
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
from dataset import load_data

_, rtl = load_data(root_dir='../', data_type='rotated')
_, std_test = load_data(root_dir='../')
test_loaders = {
    "standard": std_test,
    "rotated": rtl,
}
cache_dir = '../../cache/digit_predictions/edl_gen'

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
            for batch_idx, (data, target) in enumerate(test_loaders[ts]):
                X = g2.get_tensor_by_name('X:0')
                u = g2.get_tensor_by_name('uncertainty_out:0')
                prob = g2.get_tensor_by_name('prob_out:0')
                evidence = g2.get_tensor_by_name('evidence_out:0')
                feed_dict = {X: data.reshape(1, -1)}
                output = sess.run([u, prob, evidence], feed_dict=feed_dict)
                u = output[0]
                prob = output[1]
                evidence = output[2]

                preds[str(batch_idx)+'.jpg'] = (np.argmax(prob)+1, np.max(prob))
                _all_preds_this_image = []
                for pred_idx in range(9):
                    if prob[0][pred_idx] > 0.00001:
                        _all_preds_this_image.append((pred_idx+1, prob[0][pred_idx]))
                preds_for_problog[str(batch_idx)+'.jpg'] = _all_preds_this_image

            print('Finished Dataset: ', ts)
            # Save predictions to cache
            with open(cache_dir + '/' + ts + '_test_set.json', 'w') as cache_out:
                cache_out.write(json.dumps(preds, cls=NpEncoder))
            with open(cache_dir + '/' + ts + '_test_set_for_problog.json', 'w') as cache_out:
                cache_out.write(json.dumps(preds_for_problog, cls=NpEncoder))

