# LRCN
Long-termRecurrentNeuralNetworks

To train(You mush have data folder in the repository):

julia lrcn.jl --fast

To generate:

julia lrcn.jl img_path --generate 100

This example implements the Long-term recurrent convolutional network model from

Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description."
Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

* Paper url: https://arxiv.org/pdf/1411.4389.pdf
* Project page: https://github.com/ekinakyurek/lrcn
