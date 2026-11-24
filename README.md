# MNIST from scratch: a two-layer neural network in NumPy

A handwritten-digit classifier built with no ML libraries: every gradient derived by hand and checked numerically. **0.973 on Kaggle's Digit Recognizer**, up from 0.908 for a softmax baseline.

## Results

| Model | Settings | Validation | Kaggle |
|---|---|---|---|
| Softmax regression (baseline) | lr 0.1, 10 epochs | 0.9068 | 0.90803 |
| MLP, 128 hidden units | lr 0.1, 15 epochs | 0.9660 | 0.96400 |
| MLP, 128 hidden units | lr 0.3, 15 epochs | 0.9738 | 0.97275 |

Gradient check: backpropagation matches finite differences to a maximum relative
error of 3.3 × 10⁻⁹.

## How it works

- **Model:** 784 → 128 (ReLU) → 10 (softmax), He initialization, mini-batch
  gradient descent (batch 128), cross-entropy loss.
- **Backpropagation:**
- **Full derivation:**
- **Gradient check:**
- **Tests:**

## Experiment: learning-rate sweep

**Question.**

**Prediction.** From a 15-epoch MLP run at `lr=0.1`, there are multiple epochs where the loss increases before continuing to decrease, and at the last few epochs the training accuracy is still increasing while validation is only slightly behind. As such, it seems that this learning rate overshoots the local minimum and requires correcting. As such, I conduct an experiment, measuring validation accuracy for learning rates of `0.01`, `0.03`, `0.1,` `0.3`, `0.5`, and `1.0` and plot validation accuracy against epoch for all six. From the initial run, `0.1` seems to be too high of a learning rate and results in wobbling around the bottom of the valley, so a lower learning rate would be better. However, the training accuracy was still increasing yet, so I doubt a 90% slower learning rate would reach a meaningful plateau. As such, I predict that a learning rate of `0.03` will result in the highest validation accuracy.

**Setup.** Two-layer MLP (784 → 128 ReLU → 10), 15 epochs, batch size 128, all seeds 0.
Only the learning rate changes: 0.01, 0.03, 0.1, 0.3, 0.5, 1.0.
Runs from commit `bcceb44`; raw numbers in `results/lr_sweep/`.

![Validation accuracy for each learning rate, as six small panels](figures/lr_sweep_accuracy.png)

![Training loss for all six learning rates on a log scale](figures/lr_sweep_loss.png)

| Learning rate | Validation (epoch 15) | Training (epoch 15) |
|---|---|---|
| 0.01 | 0.9164 | 0.9236 |
| 0.03 | 0.9430 | 0.9519 |
| 0.1 | 0.9660 | 0.9801 |
| 0.3 | 0.9738 | 0.9962 |
| 0.5 | 0.9760 | 0.9990 |
| 1.0 | 0.9720 | 0.9990 |

**What happened.**

**What I chose, and why.** lr 0.3: within 11 images of the best, and it never collapsed.
Kaggle: 0.97275.

## Limitations

## How to run

## Project structure
