import  os
import shutil

cur = 'data-local/labels'
os.mkdir(cur + '/custom')

anim = {'bird', 'frog', 'cat', 'horse', 'dog', 'deer'}
inanim = {'ship', 'truck', 'automobile', 'airplane'}


for i in anim:
    with open(cur + '/custom/'+i+".txt", "w") as w:
        w.write("")

for i in inanim:
    with open(cur + '/custom/' + i + ".txt", "w") as w:
        w.write("")


d = {}

for i in range(20):
    pth = "data-local-10-labels/labels/cifar10/4000_balanced_labels/" + ("0" if i <10 else "") + str(i) + ".txt"
    with open(pth, "r") as r:
        for line in r:
            img, lab = line.split()
            if lab not in d: d[lab] = set()
            d[lab].add(img)


# labels we have : [3805, 3828, 3792, 3810, 3802, 3791, 3782, 3798, 3795, 3798] 38001
# print([len(i) for i in d.values()], sum([len(i) for i in d.values()]))

for i in d:
    with open(cur + '/custom/'+ i +".txt", "a") as w:
        for name in d[i]:
            w.write(name + " " + ("animate" if i in anim else "inanimate") + "\n")
