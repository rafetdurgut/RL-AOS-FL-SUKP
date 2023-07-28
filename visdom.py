#%%
import csv
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-bright')
operator_size = 3
alpha = 0.5
gama = 0.3
run = 0
w = 25
pmin=0.1
lmode = 0
reward = 'extreme'
# pName = 'sukp 400_385_0.15_0.85.txt'
pName='sukp 100_100_0.10_0.75.txt'

def get_data(fileName):
    datas = ([[] for _ in range(operator_size)])
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            if len(row) > 0:
            
                datas[line_count % operator_size].append(row)
                line_count += 1
        # for i in range(3):
        #     for j in range(30):
        #         datas[i][j] = datas[i][j][0:-1]
    return datas
def get_array_data(fileName ,ind):
    datas = ([[] for _ in range(operator_size)])
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if len(row) > 0:
                if line_count<3:
                    for c in row:
                        inner_list = [(elt.strip()) for elt in c.replace('[','').replace(']','').split(' ') if elt!=""]
                        datas[line_count % operator_size].append((float(inner_list[0]), float(inner_list[ind])))
                line_count += 1
        
    return datas
def get_conv_data(fileName):
    datas = []
    cg = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        lastFE = 0
        for row in csv_reader:
            if len(row) > 0:
                if lastFE > row[0]:
                    datas.append(cg)
                    cg = []
                else:
                    cg.append(row)
                lastFE = row[0]
                line_count += 1
        datas.append(cg)
    return datas
import math
import matplotlib.gridspec as gridspec
def draw_data_triple(value, title,y_label, equal=1):
    maxY = 0
    fig2 = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(4, 4,figure=fig2)
    axs = []
    for ind, o in enumerate(aos):
        print(o)
        if o == "CLRL":
            file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-{learning}-{pName}.csv"
        else:
            file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}.csv"
        credits = get_data(file_name)
        x = int(math.floor(ind / 2))
        y = int(ind % 2)

        if x == 0:
            ax = plt.subplot(gs[x:x+2, y*2:y*2 + 2])
        else:
            ax = plt.subplot(gs[x * 2:(x * 2) + 2, 1:3])
        for i in range(operator_size):
            print(i)
            ortalama = np.mean(credits[i],axis=0)
            maxY = max(maxY,max(ortalama))
            ax.plot(ortalama[0:-1], label=operators[i])
            ax.set_title(o)
            ax.set_xlabel('Iterasyon')
            ax.set_ylabel(y_label)
        axs.append(ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, loc='upper right', ncol=1)
    fig2.suptitle(title)

    if equal:
        plt.setp(axs, ylim=(0, maxY*1.1))
    plt.savefig(title,dpi=600,transparent=True)
    plt.show()
def draw_data(value, title,y_label, equal=1):
    fig, axs = plt.subplots(2, 2)
    maxY = 0
    for ind, o in enumerate(aos):
        print(o)
        if o == "CLRL":
            file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}.csv"
        else:
            file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}.csv"
        credits = get_data(file_name)
        for i in range(operator_size):
            print(i)
            ortalama = np.mean(credits[i],axis=0)
            maxY = max(maxY,max(ortalama))
            x = int(math.floor(ind / 2))
            y = int(ind % 2)
            axs[x, y].plot(ortalama, label=operators[i])
            axs[x, y].set_title(o)

    handles, labels = axs[x, y].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=operator_size)
    fig.suptitle(title)
    if equal:
        plt.setp(axs, ylim=(0,maxY))

    plt.savefig(title, dpi=600, transparent=True)
    plt.show()

def draw_data_unique(value, o, title,y_label, equal=1):
    fig, axs = plt.subplots()
    maxY = 0
    print(o)
    if o == "CLRL":
        file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-{lmode}-{pName}.csv"
    else:
        file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}.csv"
    credits = get_data(file_name)
    
    for i in range(operator_size):
        print(i)
        ortalama = np.mean(credits[i], axis=0)
        maxY = max(maxY, max(ortalama))
        axs.plot(ortalama, label=operators[i])
        #axs.set_title(o)

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=operator_size)
    fig.suptitle(title)
    if equal:
        plt.setp(axs, ylim=(0, maxY))
    #axs.set_xlabel('Iteration')
    axs.set_ylabel('Credit Value')
    plt.savefig(title, dpi=600, transparent=True)
    plt.show()
a=""
def draw_data_unique_filtered(value, o, title,y_label, equal=1, ind=1):
    fig, axs = plt.subplots()
    maxY = 0
    print(o)
    file_name = f"results/{value}-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-{lmode}-{pName}.csv"

    
    credits = get_array_data(file_name, ind)
    a = credits.copy()
    for i in range(operator_size):
        print(i)
        ortalama = credits[i]
        # maxY = max(maxY, max(ortalama))
        axs.plot(*zip(*ortalama), label=operators[i], marker='*')
        
        #axs.set_title(o)

    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, frameon=False, loc='lower center', ncol=operator_size)
    # fig.suptitle(title)
    # if equal:
    #     plt.setp(axs, ylim=(0, maxY))
    # #axs.set_xlabel('Iteration')
    # axs.set_ylabel('Credit Value')
    # plt.savefig(title, dpi=600, transparent=True)
    plt.show()

aos = ["CLRL", "PM", "UCB","CLRL"]
operators = ["disABC", "ibinABC", "binABC"]

for i in range(1,6):
    draw_data_unique_filtered('cluster_history',"CLRL", 'Cluster Values', 'Credit Values',1,i)

# for i in range(1,6):
#     draw_data_unique_filtered('ncl',"CLRL", 'Cluster Values', 'Credit Values',1,i)
    
# draw_data_unique('credits',"CLRL", 'Credit Values', 'Credit Values',0)
# draw_data_unique('rewards',"CLRL", 'Reward Values', 'Kredi Değeri',0)
# draw_data_unique('usage',"CLRL", 'Usage Counts', 'Kredi Değeri',0)
# draw_data_unique('success',"CLRL", 'Success Counts', 'Kredi Değeri',0)

# file_name = f"results/cg-CLRL-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-0-{pName}.csv"
# fig, ax = plt.subplots()

# convergence = get_conv_data(file_name)
# x = [c[0] for c in convergence[run][:]]
# y = [c[1] for c in convergence[run][:]]
# ax.plot(x, y, label="RLABC")

# file_name = f"results/cg-CLRL-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-1-{pName}.csv"

# convergence = get_conv_data(file_name)
# x = [c[0] for c in convergence[run][:]]
# y = [c[1] for c in convergence[run][:]]
# ax.plot(x, y, label="RLABC-T")

# file_name = f"results/cg-CLRL-{operator_size}-{reward}-{pmin}-{w}-{alpha}-{gama}-2-{pName}.csv"

# convergence = get_conv_data(file_name)
# x = [c[0] for c in convergence[run][:]]
# y = [c[1] for c in convergence[run][:]]
# ax.plot(x, y, label="RLABC-TC")

# ax.set_xlabel('Iteration')
# ax.set_ylabel('Objective Value')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, frameon=False, loc='lower right', ncol=1)
# fig.savefig('convergence', dpi=600)
# plt.show()
#
# draw_data('credits', 'Kredi Grafiği', 'Kredi Değeri',1)
# draw_data('rewards', 'Ödül Grafiği', 'Ödül Değeri',1)
# draw_data('usage', 'Operatör Kullanım Sayıları', 'Kullanım',1)
# draw_data('success', 'Operatör Başarılı Güncelleme Sayıları', 'Başarı sayısı',1)

#draw_data('rewards', 'Rewards',1)
#draw_data('usage', 'Usage',1)
#draw_data('success', 'Success',1)

# fig, ax = plt.subplots()
# for ind, o in enumerate(aos):
#     print(ind)
#     convergence = get_conv_data(f"results/cg-{o}-{operator_size}-{reward}-{pmin}-{w}-{alpha}.csv")
#     x = [c[0] for c in convergence[run][:]]
#     y = [c[1] for c in convergence[run][:]]
#     ax.plot(x, y, label=o)
#     ax.set_xlabel('Iterasyon')
#     ax.set_ylabel('En iyi değer')

# print(convergence)