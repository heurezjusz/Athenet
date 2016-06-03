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
    "i" : ["derest"],
    "b" : [250],
    "c" : ["sum_max", "sum_lenght", "max_lenght"],
    "a" : ["none", "lenght", "max_value"],
    "r" : ["none", "lenght", "max_value"],
    "t" : ["all", "conv", "fully-connected"]
}

for i, values in zip(xrange(MAX_ITER), product(*options.itervalues())):
    a = " -n " + args.network + " -e " + str(args.examples)
    for k, v in zip(options, values):
        a += " -" + k + " " + str(v)

    print "RUN " + str(i + 1) + " : " + a

    os.system("python " + os.path.dirname(__file__) + "/indicators.py" + a)

