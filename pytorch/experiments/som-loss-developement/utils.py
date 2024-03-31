import json
from matplotlib import pyplot as plt
import seaborn as sns


def show_conf_matrix(confusion, class_labels):
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.2)  # Adjust the font size for clarity

    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


f = open('model_results/zoo-exp-200ep-10hid/10models200eps0.2k.json')
data = json.load(f)

epoch = 200-1
for model_num in range(20):
    # show_conf_matrix(data['conf_mats'][str(epoch)], ['Mammal', 'Bird', 'Reptile',
    # 'Fish', 'Amphibian', 'Bug', 'Invertebrate'])
    # print(data['conf_mats'][str(model_num)][str(epoch)])
    mat_data = data['conf_mats'][str(model_num)][str(epoch)]
    for i in range(len(mat_data)):
        mat_data[i] = mat_data[i]
    print(mat_data)
    show_conf_matrix(mat_data, ['0', 'Mammal', 'Bird', 'Reptile',
                                'Fish', 'Amphibian', 'Bug', 'Invertebrate'])
