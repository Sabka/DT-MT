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

    def write(self, file):
        with open(file, "w") as w:
            w.write(f"x\ty\terr\n")
            for i in range(self.eps):
                w.write(f"{i+1}\t{round(self.means[i], 4)}\t{round(self.stds[i], 4)}\n")



### MODEL

f = open('raw_data/m2-20models50eps.json')
data = json.load(f)
f.close()


EPS = 50
train = MS(data['train_loss'], EPS)
train.fill()
train.write("model_train_loss.txt")
test = MS(data['test_acc'], EPS)
test.fill()
test.write("model_test_acc.txt")

#print(data["som"]["qe"][-1])
#print(data["som"]["wd"][-1])
#print(data["som"]["e"][-1])


### BASELINE
f = open('raw_data/b2-20models50eps.json')
data = json.load(f)
f.close()

EPS = 50
train = MS(data['train_loss'], EPS)
train.fill()
train.write("baseline_train_loss.txt")
test = MS(data['test_acc'], EPS)
test.fill()
test.write("baseline_test_acc.txt")



