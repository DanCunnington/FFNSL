import json
import sys
custom_args = [{
    "flag": "--decks",
    "type": str,
    "default": '["standard","batman_joker","captain_america",'
               '"adversarial_standard","adversarial_batman_joker","adversarial_captain_america"]',
    "help": 'which card decks to use as input to the neural networks'
}, {
    "flag": "--non_perturbed_deck",
    "type": str,
    "default": "standard",
    "help": 'which card deck is within the training distribution'
}]


def process_custom_args(a):
    # Process custom args
    try:
        decks_arg = json.loads(a['decks'])
    except json.decoder.JSONDecodeError as e:
        print("Error decoding decks argument. Stack trace: ")
        print(e)
        sys.exit(1)
    a['decks'] = decks_arg
    return a
