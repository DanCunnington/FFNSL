import json
import sys
custom_args = [{
    "flag": "--datasets",
    "type": str,
    "default": '["standard","rotated"]',
    "help": 'which MNIST datasets to use as input to the neural networks'
}, {
    "flag": "--non_perturbed_dataset",
    "type": str,
    "default": "standard",
    "help": 'which dataset is within the training distribution'
}]


def process_custom_args(a):
    # Process custom args
    try:
        datasets_arg = json.loads(a['datasets'])
    except json.decoder.JSONDecodeError as e:
        print("Error decoding decks argument. Stack trace: ")
        print(e)
        sys.exit(1)
    a['datasets'] = datasets_arg
    return a
