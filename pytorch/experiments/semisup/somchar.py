from som_from_fv import SOM
import torch

som = torch.load("fv_som_results/pretrained_som-79ep.pt")
dists = [[0 for j in range(som.n*som.m)] for i in range(som.m*som.n)]

for i in range(som.n * som.m):
	for j in range(som.n * som.m):
		dists[i][j] = float(torch.sum(torch.pow(som.weights[i] - som.weights[j], 2)))
		
max_value = dists[0][0], (0,0)
for i, row in enumerate(dists):
    for j, num in enumerate(row):
        if num > max_value[0]:
            max_value = num, (i,j)
            
min_value = dists[0][1], (0,0)
for i, row in enumerate(dists):
    for j, num in enumerate(row):
        if i == j: continue
        if num < min_value[0]:
            min_value = num, (i,j)
		
print(max_value, min_value)
