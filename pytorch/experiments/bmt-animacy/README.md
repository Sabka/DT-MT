# BMT experiment - animate vs. inanimate objects (different data portions)

Code is adapted from https://github.com/iSarmad/MeanTeacher-SNTG-HybridNet, which contain MT model architecture and training setup. I changed MT model to BMT model, keep some parts of training setup, and changed task to binary classification of animate and inanimate objects.

## How to create dataset for binary classification of animate vs. inanimate objects ?
- rename `0data-local` to `data-local`
- in folder data-local/bin run `python3 unpack_cifar10.py . .`, this downloads data
- rename `data-local` to `data-local-10-labels` and create empty `data-local`
- `data-local-10-labels` now contain dataset devided to 10 classes, now we create binary dataset from it. Run command `python3 prepare_labels.py`.
- after previous step, data are ready for BMT training


## Training
Before training, you can update model and training parameters in parameters.py (for example number of epochs in line 47). If the machine has cuda available, it automaticaly runs on cuda. Training write outputs on the console.

### Baseline
Baseline for this experiment is only student network, without teacher.
In baseline.py, we can set labeled_portion parameter, which means, how many labeled samples from 1 class are used. Other samples are unlabeled.
We can then run baseline by command `python3 baseline.py`.

### BMT
In main.py, we can also set labeled_portion parameter.
We can then run BMT model by command `python3 main.py`.


## Setup virtual environment
I used conda environment in our machines. Steps for setuping env are in the root folder in README.


# main changes of Sarmads code
- async -> remove async
- .cuda() -> .to(args.device)
- .view(-1) -> .reshape(-1)
