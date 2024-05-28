import  os
import shutil

old = 'data-local-10-labels'
cur = 'data-local'


os.mkdir(cur + '/labels')
os.mkdir(cur + '/labels/cifar10')
os.mkdir(cur + '/labels/cifar10/1000_balanced_labels')
os.mkdir(cur + '/labels/cifar10/4000_balanced_labels')
os.mkdir(cur + '/bin')
os.mkdir(cur + '/bin/train')
os.mkdir(cur + '/bin/val')

anim = {'bird', 'frog', 'cat', 'horse', 'dog', 'deer'}
inanim = {'ship', 'truck', 'automobile', 'airplane'}

os.mkdir(cur + '/bin/train/inanimate')
os.mkdir(cur + '/bin/train/animate')


for file in os.listdir(old + '/bin/train+val'):
    if file in inanim:
        for image in os.listdir(old + '/bin/train+val/' + file):
            shutil.copy(old + '/bin/train+val/' + file + '/' + image, cur + '/bin/train/inanimate/' + image)
    else:
        for image in os.listdir(old + '/bin/train+val/' + file):
            shutil.copy(old + '/bin/train+val/' + file + '/' + image,
                        cur + '/bin/train/animate/' + image)

os.mkdir(cur + '/bin/val/inanimate')
os.mkdir(cur + '/bin/val/animate')

for file in os.listdir(old + '/bin/test'):

    if file in inanim:
        for image in os.listdir(old + '/bin/test/' + file):
            shutil.copy(old + '/bin/test/' + file + '/' + image, cur + '/bin/val/inanimate/' + image)

    else:
        for image in os.listdir(old + '/bin/test/' + file):
            shutil.copy(old + '/bin/test/' + file + '/' + image,
                        cur + '/bin/val/animate/' + image)

for file in os.listdir(old + '/labels/cifar10/4000_balanced_labels'):

    with open(old + '/labels/cifar10/4000_balanced_labels/' + file, 'r') as r:
        with open(cur + '/labels/cifar10/4000_balanced_labels/' + file, 'w') as w:

            for line in r:

                image, label = line.strip().split()
                if label in inanim:
                    label = 'inanimate'
                else:
                    label = 'animate'

                w.write(f"{image} {label}\n")

for file in os.listdir(old + '/labels/cifar10/1000_balanced_labels'):

    with open(old + '/labels/cifar10/1000_balanced_labels/' + file, 'r') as r:
        with open(cur + '/labels/cifar10/1000_balanced_labels/' + file, 'w') as w:

            for line in r:

                image, label = line.strip().split()
                if label in inanim:
                    label = 'inanimate'
                else:
                    label = 'animate'

                w.write(f"{image} {label}\n")

