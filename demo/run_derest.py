import argparse
import os
from argparse import RawTextHelpFormatter
from itertools import product

MAX_ITER = 5000

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

parser.add_argument("-n", "--network",
                    help="Derest will be ran on chosen kind of network. "
                         "Default option is \"lenet\".",
                    choices=["lenet", "alexnet", "googlenet"],
                    default="lenet")

parser.add_argument("-e", "--examples", type=int,
                    help="Chooses how many different fractions of weights "
                         "will be deleted (the higher the number,"
                         "the more thorough the plot "
                         "and the more time it takes)."
                         "Default is 4",
                    default=4)


args = parser.parse_args()


options = {
    "i": ["derest"],
    "b": [250],
    "c": ["max", "length"],
    "a": ["none", "length", "max_value"],
    "r": ["none", "length", "max_value"],
    "t": ["all", "conv", "fully-connected"]
}

path = os.path.dirname(__file__)
if path:
    path += "/"

for i, values in zip(xrange(MAX_ITER), product(*options.itervalues())):
    a = " -n " + args.network + " -e " + str(args.examples) + " -p -l "
    for k, v in zip(options, values):
        a += " -" + k + " " + str(v)

    print "RUN " + str(i + 1) + " : " + a

    os.system("python " + path + "indicators.py" + a)

