import os
import json
import matplotlib.pyplot as plt

baseline = "200eps0k0psi1.json"

mt_som = ["200eps0k0psi2.json",
"200eps10k1psi1.json", 
"200eps10k10psi1.json", 
"200eps0k0psi2.json", 
"200eps10k1psi2.json", 
"200eps5k1psi1.json", 
"200eps5k10psi1.json", 
"200eps10k1psi2.json",
"200eps10k10psi2.json",
"200eps20k1psi1.json",
"200eps5k1psi2.json",
"200eps5k10psi2.json",
"200eps20k1psi2.json"
]


plt.style.use('bmh')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))


with open(baseline, 'r') as r:
    data = json.load(r)
print(baseline, max(data['test_acc']['teacher']))
l = len(data['test_acc']['teacher'])

axs[0].plot(range(0,l,1), data['train_loss']['total'][::1], label="Baseline")
axs[1].plot(range(0,l,1), data['train_loss']['sup'][::1])
axs[2].plot(range(0,l,1), data['train_loss']['som'][::1])


for model_i in mt_som:
    with open(model_i, 'r') as r:
        data = json.load(r)
    print(model_i, max(data['test_acc']['teacher']))
    l = len(data['test_acc']['teacher'])

    axs[0].plot(range(0,l,1), data['train_loss']['total'][::1], label="MT-SOM (kappa=5, psi=10)")
    axs[1].plot(range(0,l,1), data['train_loss']['sup'][::1])
    axs[2].plot(range(0,l,1), data['train_loss']['som'][::1])


axs[0].set_title('Training loss - total')
axs[1].set_title('Training loss - supervised')
axs[2].set_title('Training loss - consistency')

# axs[0].set_ylim([0, 10])

#axs[4].set_ylim([55, 68])


fig.legend(loc='lower right',  bbox_to_anchor=(0.98, 0.78),
           fancybox=True, shadow=True)

# Adjust layout
plt.tight_layout()

# Save plot as image
plt.savefig('fv-som-loss.png')






###############################################

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

with open(baseline, 'r') as r:
    data = json.load(r)

l = len(data['test_acc']['teacher'])

axs[0].plot(range(0,l,1), data['test_acc']['student'][::1], label="Baseline")
axs[1].plot(range(0,l,1), data['test_acc']['teacher'][::1])


for model_i in mt_som:
    with open(model_i, 'r') as r:
        data = json.load(r)
		
    axs[0].plot(range(0,l,1), data['test_acc']['student'][::1], label="MT-SOM (kappa=5, psi=10)")
    axs[1].plot(range(0,l,1), data['test_acc']['teacher'][::1])


axs[0].set_title('Test accuracy - student')
axs[1].set_title('Test accuracy - teacher')

axs[0].set_ylim([30, 85])

axs[1].set_ylim([70, 85])


fig.legend(loc='lower right', bbox_to_anchor=(0.98, 0.1),
           fancybox=True, title="SOM size", shadow=True)

# Adjust layout
plt.tight_layout()

# Save plot as image
plt.savefig('fv-som-acc.png')

