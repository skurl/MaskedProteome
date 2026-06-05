# Secondary Structure Prediction

Im trying to learn how to encode transformer-based ML models.

## Authors

- Maciej Szczesny

## Notes
[2026-06-01]:
Basically the other notebook is a suboptimal to run on a computer. [MOST OF THE TRANSFORMER CODE FROM HERE](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb).

Length size of 72 was used because 1. It contains 700+ seq. 2.

This model is extremely inefficient as ive since learned. I decided to use a encoder-dcoder architecture, since I approached it as a "translation" problem, but after I built it I realsed it only requires new embeddings per residue.

[2026-06-03]: This architecture needs changing! Current encoder-decoder model is inefficient, and could be better solved using an encoder-only architecture. `Test Loss: 0.4879, Test Accuracy: 82.17%`

[2026-06-04]: After changing the architecture, we're getting *marginally* better results: `Test Loss: 0.8457, Test Accuracy: 82.06%`
Changing DNN ReLU to GELU did marginally improve results.

Introducing a local convolution layer further improved our results: `Test Loss: 0.6680 Test Accuracy: 83.41%`

[2026-06-05]: Changed the embedding from 1-20 per amino acid and 0-7 for sst8 to one-hot embeddings. It did result in a drop in performance, but it is conceptually better. `Test Loss: 0.8760
Test Accuracy: 79.66%`

