"""
t-SNE对手写数字进行可视化
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import os


def get_data():
    color = {'0': 0.1245,
             '1': 0.2546,
             '2': 0.3846,
             '3': 0.5556,
             '4': 0.7006,
             '5': 0.8365,
             '6': 0.9456}
    label_test = []
    color_test = []
    data_test = []
    for file_name in os.listdir('./tsne/time_shuffle/'):
        data_test.append(np.load('./tsne/time_shuffle/' + file_name))
    data_test = np.concatenate(
        (data_test[0], data_test[1], data_test[2], data_test[3], data_test[4], data_test[5], data_test[6],
         data_test[7], data_test[8], data_test[9], data_test[10], data_test[11], data_test[12], data_test[13],
         data_test[14], data_test[15], data_test[16], data_test[17], data_test[18], data_test[19], data_test[20],
         data_test[21], data_test[22], data_test[23], data_test[24], data_test[25], data_test[26], data_test[27],
         data_test[28], data_test[29], data_test[30], data_test[31], data_test[32], data_test[33], data_test[34],
         data_test[35],
         ))
    '''data_test[36], data_test[37], data_test[38], data_test[39], data_test[40], data_test[41], data_test[42],
         data_test[43], data_test[44], data_test[45], data_test[46], data_test[47], data_test[48], data_test[49],
         data_test[50], data_test[51], data_test[52], data_test[53], data_test[54], data_test[55], data_test[56],
         data_test[57], data_test[58], data_test[59], data_test[60], data_test[61], data_test[62], data_test[63],
         data_test[64], data_test[65], data_test[66], data_test[67], data_test[68], data_test[69], data_test[70],
         data_test[71],
         data_test[72], data_test[73], data_test[74], data_test[75], data_test[76], data_test[77], data_test[78],
         data_test[79], data_test[80], data_test[81], data_test[82], data_test[83], data_test[84], data_test[85],
         data_test[86], data_test[87], data_test[88], data_test[89], data_test[90], data_test[91], data_test[92],
         data_test[93], data_test[94], data_test[95], data_test[96], data_test[97], data_test[98], data_test[99],
         data_test[100], data_test[101], data_test[102], data_test[103], data_test[104], data_test[105], data_test[106],
         data_test[107],
         data_test[108], data_test[109], data_test[110], data_test[111], data_test[112], data_test[113], data_test[114],
         data_test[115], data_test[116], data_test[117], data_test[118], data_test[119], data_test[120], data_test[121],
         data_test[122], data_test[123], data_test[124], data_test[125], data_test[126], data_test[127], data_test[128],
         data_test[129], data_test[130], data_test[131], data_test[132], data_test[133], data_test[134], data_test[135],
         data_test[136], data_test[137], data_test[138], data_test[139], data_test[140], data_test[141], data_test[142],
         data_test[143],
         data_test[144], data_test[145], data_test[146], data_test[147], data_test[148], data_test[149], data_test[150],
         data_test[151], data_test[152], data_test[153], data_test[154], data_test[155], data_test[156], data_test[157],
         data_test[158], data_test[159], data_test[160], data_test[161], data_test[162], data_test[163], data_test[164],
         data_test[165], data_test[166], data_test[167], data_test[168], data_test[169], data_test[170], data_test[171],
         data_test[172], data_test[173], data_test[174], data_test[175], data_test[176], data_test[177], data_test[178],
         data_test[179],
         data_test[180], data_test[181], data_test[182], data_test[183], data_test[184], data_test[185], data_test[186],
         data_test[187], data_test[188], data_test[189], data_test[190], data_test[191], data_test[192], data_test[193],
         data_test[194], data_test[195], data_test[196], data_test[197], data_test[198], data_test[199], data_test[200],
         data_test[201], data_test[202], data_test[203], data_test[204], data_test[205], data_test[206], data_test[207],
         data_test[208], data_test[209], data_test[210], data_test[211], data_test[212], data_test[213], data_test[214],
         data_test[215],
         data_test[216], data_test[217], data_test[218], data_test[219], data_test[220], data_test[221], data_test[222],
         data_test[223], data_test[224], data_test[225], data_test[226], data_test[227], data_test[228], data_test[229],
         data_test[230], data_test[231], data_test[232], data_test[233], data_test[234], data_test[235], data_test[236],
         data_test[237], data_test[238], data_test[239], data_test[240], data_test[241], data_test[242], data_test[243],
         data_test[244], data_test[245], data_test[246], data_test[247], data_test[248], data_test[249], data_test[250],
         data_test[251],
         data_test[252], data_test[253], data_test[254], data_test[255], data_test[256], data_test[257], data_test[258],
         data_test[259], data_test[260], data_test[261], data_test[262], data_test[263], data_test[264], data_test[265],
         data_test[266], data_test[267], data_test[268], data_test[269], data_test[270], data_test[271], data_test[272],
         data_test[273], data_test[274], data_test[275], data_test[276], data_test[277], data_test[278], data_test[279],
         data_test[280], data_test[281], data_test[282], data_test[283], data_test[284], data_test[285], data_test[286],
         data_test[287],
         data_test[288], data_test[289], data_test[290], data_test[291], data_test[292], data_test[293], data_test[294],
         data_test[295], data_test[296], data_test[297], data_test[298], data_test[299], data_test[300], data_test[301],
         data_test[302], data_test[303], data_test[304], data_test[305], data_test[306], data_test[307], data_test[308],
         data_test[309], data_test[310], data_test[311], data_test[312], data_test[313], data_test[314], data_test[315],
         data_test[316], data_test[317], data_test[318], data_test[319], data_test[320], data_test[321], data_test[322],
         data_test[323]data_test[324], data_test[325], data_test[326], data_test[327], data_test[328], data_test[329], data_test[330],
         data_test[331], data_test[332], data_test[333], data_test[334], data_test[335], data_test[336], data_test[337],
         data_test[338], data_test[339], data_test[340], data_test[341], data_test[342], data_test[343], data_test[344],
         data_test[345], data_test[346], data_test[347], data_test[348], data_test[349], data_test[350], data_test[351],
         data_test[352], data_test[353], data_test[354], data_test[355], data_test[356], data_test[357], data_test[358],
         data_test[359]'''
    for data in data_test:
        label_test.append(np.argmax(data))
        color_test.append(color[str(np.argmax(data))])

    return data_test, label_test, color_test


def plot_embedding(data, label, color, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()

    ax = plt.subplot(111)
    ax = plt.scatter(data[:, 0], data[:, 1], c=color)
    plt.legend(handles=ax.legend_elements()[0],labels=['0', '1', '2', '3', '4', '5', '6'],loc='upper right')
    '''cbar = plt.colorbar(ticks=range(7))
    cbar.set_label(label='digit value', fontdict=font)'''
    '''for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})'''
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig('./time_shuffle_tsne.png')
    return fig


def main():
    data, label, color = get_data()
    print('Computing t-SNE embedding')
    # 降到2维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label, color,
                   't-SNE embedding of the digits (time %.2fs)'
                   % (time() - t0))
    plt.show()


if __name__ == '__main__':
    main()
