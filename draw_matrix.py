import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def draw_confusion_matrix(label_true, label_pre, label_name, title='Confusion Matrix', dpi=100, pdf_save_path=None):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pre, normalize='true')
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel('Predict label')
    plt.ylabel('Truth label')
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1,1,1) if i==j else (0, 0, 0)
            # color = (0, 0, 0)
            value = float(format('%.2f'%cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

if __name__ == '__main__':
    pass