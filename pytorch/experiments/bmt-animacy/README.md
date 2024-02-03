# DT-mean-teacher

Kod je adaptovany z https://github.com/iSarmad/MeanTeacher-SNTG-HybridNet, poodstranovala som casti kt. nie su 
potrebne pre mean teachera a tie, ktore pouzivaju cudu, lebo ju nemám a hádzali erorry.

Chýba ešte priečinok data-local so samotným datasetom, kt. je ale veľmi veľký 500MB.

## Ako si vytvoriť dataset pre binary mean teachera?
- premenujeme `0data-local` na `data-local`
- v priečinku data-local/bin spustíme `python3 unpack_cifar10.py . .`, tým sa dáta stiahnu
- premenujeme si `data-local` na `data-local-10-labels` a vytvoríme prázdny `data-local`
- pôvodný `data-local` obsahoval dataset rozdelený na 10 labelov, ak chceme binárny, použijeme skript 
  `prepare_dataset_to_binMT.py`, spustíme ako `python3 prepare_dataset_to_binMT.py` 
   ( kde ak sme vynechali predošlý krok upravíme hodnoty `old = 'data-local-10-labels'` a `cur = 'data-local'  ` na cesty k priečinkom so starým datasetom a prázdnemu priečinku kde bude nový dataset, inak len spustíme)
- po spustení by mali byť dáta pripravené pre trénovanie binary mean teachera

Pred trenovanim treba este v parameters.py upravit pocet epoch na riadku 47 napr. na 200 a zrejme aj odkomentovat vsetky `#.cuda()` ktore som zakomentovala, lebo to moj pocitac nepodporuje, ale stroj s GPU by mohol.

Trénovanie vypisuje na konzolu, možno bude dobré posielať výstup niekam do súboru.


## Neptun - priprava venv

bolo treba stiahnut, overit a nainstalovat condu do mojho homu https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html potom `conda init`

Vytvorenie prostredia s potrebnymi libkami:

`conda create --name torch_cuda`

`conda install -n torch_cuda  pytorch-gpu torchvision cudatoolkit=11.1 -c conda-forge`

`conda install -n mt tqdm`

`conda install -c anaconda scikit-learn`

`conda install -c conda-forge matplotlib`

or just

`conda install -n mt2  pytorch-gpu torchvision matplotlib tqdm scikit-learn cudatoolkit=11.1 -c conda-forgee`

`conda activate torch_cuda`

# bugs
- async -> remove async
- .cuda() -> .to(args.device)
- IndexError: invalid index of a 0-dim tensor. Use `tensor.item() -> ??
- .view(-1) -> .reshape(-1)
