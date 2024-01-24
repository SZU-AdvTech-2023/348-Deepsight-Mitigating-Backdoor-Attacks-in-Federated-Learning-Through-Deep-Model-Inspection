import matplotlib.pyplot as plt

data1 = []
with open('main_acc_test.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data1.append(value)
data2 = []
with open('bd_acc_test.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data2.append(value)
data3 = []
with open('ds_acc_main.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data3.append(value)
data4 = []
with open('ds_acc_bd.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data4.append(value)
plt.plot(data1, label='Main Accuary')
plt.plot(data2, label='Backdoor Accuary')
plt.plot(data3, label='DeepSight Main Accuary')
plt.plot(data4, label='DeepSight Backdoor Accuary')
plt.legend()
plt.xlabel('X',)
plt.ylabel('Y')
plt.title('Training Accuary')
plt.savefig('trainingacc.png')
plt.show()
