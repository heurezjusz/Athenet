How to run AlexNet:

All required files can be downloaded from:
https://drive.google.com/folderview?id=0Bw3yOJZsDOJMeDl3R05mYzFJOFE&usp=sharing_eid&ts=565f926e&tid=0B3GIboTT8-hNZC14QmViUUV2MEE

GoogLeNet model can be created using athenet.models.googlenet(trained) function.
'trained' parameter specifies whether to load trained weights from file.
In case of trained=True, the weights by default will be loaded from
bin/googlenet_weights.pkl.gz file, which must be accesible.
Alternatively, trained=False can be used, in which case the weights will
be set randomly.

For testing accuracy there are needed:
1. Validation images in the bin/ILSVRC<year>_img_val_googlenet folder.
2. Ground truth for validation images in the
   data/ILSVRC<year>_img_val.txt file.
Example validation data for GoogleNet and ground truth from ImageNet 2012 can be
downloaded using link above.
