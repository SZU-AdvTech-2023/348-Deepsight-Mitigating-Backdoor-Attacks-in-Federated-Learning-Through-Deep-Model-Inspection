import matplotlib.pyplot as plt

data1 = []
with open('ds_loss_main.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data1.append(value)
data2 = []
with open('bd_loss.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data2.append(value)
data3 = []
with open('ds_loss_bd.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data3.append(value)

data4 = []
with open('main_loss.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data4.append(value)

data5 = []
with open('ds_loss_NA.txt', 'r') as f:
    for line in f:
        value = float(line.strip())
        data5.append(value)
# 绘制折线图
plt.plot(data1, label='DeepSight Main Loss')
plt.plot(data2, label='Backdoor Loss')
plt.plot(data3, label='DeepSight Backdoor Loss')
plt.plot(data4, label='Main Loss')
plt.plot(data5, label='DeepSight No Attack Main Loss')
plt.legend()
plt.xlabel('X',)
plt.ylabel('Y')
plt.title('Training Loss')
plt.savefig('trainingloss.png')
plt.show()

