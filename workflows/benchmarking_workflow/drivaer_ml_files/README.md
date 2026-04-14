# DrivAerML Dataset Files

The DrivAerML dataset consists of 500 parametrically varied geometries.
However, only 484 of these geometries are usable for training and validation.
Since the [publication](https://arxiv.org/pdf/2408.11969) does not recommend
any training-validation split, we propose a split in this repository to enable
a fair and consistent benchmark.

The dataset is split as 90% for training and 10% for validation. For selecting
the validation set, the entire dataset is sorted by drag force. Then 10% of the
validation set is chosen from the top of the sorted dataset, and another
10% of the validation set is chosen from the bottom of the sorted dataset.
The remaining 80% of the validation set is chosen randomly from the remaining
dataset.

This allows us to include some out-of-distribution data in the validation set,
which is important for the generalizability of the models.

The figure below shows the distribution of the training-validation sets.

![Design trend split](design_trend_split_0.9_0.1.png)
