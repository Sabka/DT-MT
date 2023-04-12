# DT-MT

## Neptun - priprava venv

bolo treba stiahnut, overit a nainstalovat condu do mojho homu https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html potom `conda init`

Vytvorenie prostredia s potrebnymi libkami:


`conda create -n mt2`

`conda install -n mt2  pytorch-gpu torchvision matplotlib tqdm scikit-learn scipy=1.8 pandas cudatoolkit=11.1 -c conda-forge`

`conda activate mt2`

`pip install quicksom`

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
