Goal: digit recognition network for Kaggle submission

Learning rate sweep experiment:

From a 15-epoch MLP run at `lr=0.1`, there are multiple epochs where the loss increases before continuing to decrease, and at the last few epochs the training accuracy is still increasing while validation is only slightly behind. As such, it seems that this learning rate overshoots the local minimum and requires correcting. As such, I conduct an experiment, measuring validation accuracy for learning rates of `0.01`, `0.03`, `0.1,` `0.3`, `0.5`, and `1.0` and plot validation accuracy against epoch for all four. From the initial run, `0.1` seems to be too high of a learning rate and results in wobbling around the bottom of the valley, so a lower learning rate would be better. However, the training accuracy was still increasing yet, so I doubt a 90% slower learning rate would reach a meaningful plateau. As such, I predict that a learning rate of `0.03` will result in the highest validation accuracy.
