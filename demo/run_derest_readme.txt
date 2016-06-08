run_derest should be executed from 'Athenet/' directory, not from 'Athenet/demo'. Example run:

$ python demo/run_derest.py -n lenet -e 10

for more options execute:

$ python demo/run_derest.py -h

If any other program like Dropbox observes files in Athenet/tmp, turn it off
for correct behavior of run_derest. run_derest should be able to delete
files in this directory.
