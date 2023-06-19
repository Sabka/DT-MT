# DT-MT

How to use screen on Neptun?

`screen` 

run command to be execute

`ctrl + A` and `ctrl + D`

to return to screen 

`screen -ls` - list of all screens, find id

`screen -r [id/nothing]`

to terminate screen - `ctrl + D`



# RUN TRAINING w LOGGING TO FILE
`python main.py     --dataset cifar10     --labels data-local/labels/cifar10/1000_balanced_labels/00.txt     --arch cifar_sarmad     --consistency 100.0     --consistency-rampup 5     --labeled-batch-size 62     --epochs 180     --lr-rampdown-epochs 210 2>&1 | tee curlog.txt`
