import json
import os, sys

p = []

dir = "base-exp-250eps-10hid"
for file_name in os.listdir(dir):
	if "json" in file_name:
		p.append(file_name)

# p = [ "20models100eps0k.json", "20models100eps0.8k.json", "20models100eps0.7k.json", "20models500eps0k.json", "20models500eps0.7k.json", "20models500eps0.5k.json", "20models500eps0.8k.json", "20models50eps-base.json", "20models50eps.json", "10models50eps0.8k.json", "10models50eps0.5k.json", "10models50eps0.2k.json", "10models50eps0.1k.json", "10models50eps0.01k.json", "10models50eps0.001k.json", "20models50eps0.85k.json", "20models50eps0.82k.json", "20models50eps0.78k.json", "20models50eps0.75k.json", "20models50eps0.65k.json", "20models50eps0.9k.json", "20models50eps0.7k.json", "20models50eps0.75k.json", ""]

import json

class MS:
    def __init__(self, models, eps):
        self.means = [0 for _ in range(eps)]
        self.stds = [0 for _ in range(eps)]
        self.models = models
        self.eps = eps

    def fill(self):

        for model in self.models:
            for epoch in range(self.eps):
                self.means[epoch] += self.models[str(model)][str(epoch)]
        self.means = [i/len(self.models) for i in self.means]

        for model in self.models:
            for epoch in range(self.eps):
                self.stds[epoch] += (self.means[epoch] - self.models[str(model)][str(epoch)])**2
        self.stds = [(i/len(self.models))**0.5 for i in self.stds]

for i in sorted(p):
	f = open(dir + "/" + i)
	data = json.load(f)
	f.close()
	
	kappa = i.split("eps")
	kappa = kappa[1].split("k")
	kappa = kappa[0]
	
	
	EPS = 250
	train = MS(data['train_loss'], EPS)
	train.fill()

	
	test = MS(data['test_acc'], EPS)
	test.fill()


	train_acc = MS({ "0": data['train_acc']}, EPS)
	train.fill()

	print(f"${kappa}$ &   ${round(train.means[-1],3)}	\pm {round(train.stds[-1],3)} $  &  ${round(test.means[-1], 2)}	\pm {round(test.stds[-1], 2)}$  \\\ \\hline")		


"""
for i in sorted(p):
	f = open(i)
	data = json.load(f)
	f.close()

	avg = 0
	num = 0
	for j in data['test_acc']:
		if "100" in i:
			avg += data['test_acc'][j]["99"]
		elif "500" in i:
			avg += data['test_acc'][j]["499"]
		elif "50" in i:
			avg += data['test_acc'][j]["49"]
		num += 1
	avg /= num
	
	
	print(f"{i}\t{avg}")"""


