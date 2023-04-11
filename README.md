# DT-MT

## Neptun - priprava venv

bolo treba stiahnut, overit a nainstalovat condu do mojho homu https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html potom `conda init`

Vytvorenie prostredia s potrebnymi libkami:

`conda create --name torch_cuda`

`conda install -n torch_cuda  pytorch-gpu torchvision cudatoolkit=11.1 -c conda-forge`

`conda install -n mt tqdm`

`conda install -c anaconda scikit-learn quicksom`

`conda install -c conda-forge matplotlib`

or just

`conda install -n mt2  pytorch-gpu torchvision matplotlib tqdm scikit-learn cudatoolkit=11.1 -c conda-forgee`

`conda activate torch_cuda`

## bugs
- async -> remove async
- .cuda() -> .to(args.device)
- IndexError: invalid index of a 0-dim tensor. Use `tensor.item() -> ??
- .view(-1) -> .reshape(-1)


How to use screen on Neptun?

`screen` 

run command to be execute

`ctrl + A` and `ctrl + D`

to return to screen 

`screen -ls` - list of all screens, find id

`screen -r [id/nothing]`

to terminate screen - `ctrl + D`
