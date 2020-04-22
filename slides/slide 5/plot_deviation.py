from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# import data
iris = {}
with open('iris.txt', 'r') as fin:
    for line in fin:
        line = line.strip().split(',')
        case = map(float, line[0:-1])
        if line[-1] in iris:
            iris[line[-1]].append(case)
        else:
            iris[line[-1]] = [case]

# scale attributes to guassian distribution (0, 1)
scaled_iris = {}
for category in iris:
    scaled_iris[category] = preprocessing.scale(iris[category],axis=0)

# figure
xlabels=['sepal length', 'sepal width', 'petal length', 'petal width']
labels = []
data = []
for category in scaled_iris:
    labels.append(category)
    data.extend(scaled_iris[category])

data = np.array(data)
repeated = data.repeat(50, axis=1)

plt.figure(figsize=(16, 10))
cax = plt.matshow(repeated, fignum=1, cmap='jet')
plt.xticks([50, 100, 150], ['', '', ''])
plt.yticks([50, 100, 150])
plt.ylabel('             '.join(x[5:-1] for x in labels), fontsize=24)
plt.xlabel('     '.join(x for x in xlabels), fontsize=24)
plt.colorbar(cax)
plt.show()
plt.savefig('iris matrix.pdf')