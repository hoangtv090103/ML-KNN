## KNN Algorithm

KNN is a supervised machine learning algorithm that can be used to solve both classification and regression problems. It is a lazy learning algorithm because it doesn't have a specialized training phase and uses all data for training while classification. KNN is a non-parametric algorithm because it doesn't make any assumptions on the underlying data distribution.

## How does the KNN algorithm work?

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph.

## How do we choose the factor K?

The number of neighbors (K) is the most important factor in KNN. K is generally an odd number if the number of classes is 2. Choosing K=1 is not a good idea because a noisy data point will have a very strong impact. Choosing a higher value of K will be a good idea because it reduces the effect of noise on the classification, but there is no guarantee that the overall accuracy will increase. The best way to find the optimal value of K is to try different values of K and check the accuracy for each value of K.

## How do we calculate the distance between two points?

The distance between two points is calculated using Euclidean distance. All the points are plotted on a graph and distance is calculated between two points using the Euclidean distance formula.

**Euclidean distance formula:**

$$
d(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}
$$

where $p$ and $q$ are two points, $p_1$ and $q_1$ are the $x$ coordinates, $p_2$ and $q_2$ are the $y$ coordinates, and so on.

### Import library

```python
import csv
import numpy as np
import math
```

### Load data

```python
def load_data(path='') -> tuple:
    f = open(path, "r")
    data = csv.reader(f)  # csv format
    data = np.array(list(data))  # covert to matrix
    data = np.delete(data, 0, 0)  # delete header
    data = np.delete(data, 0, 1)  # delete index
    np.random.shuffle(data)  # shuffle data
    f.close()
    data_length = len(data)
    trainSet = data[:int(0.8 * data_length)]  # 80% is training data
    testSet = data[int(0.8 * data_length):]  # 20% is testing data
    return trainSet, testSet
```

### Calculate distance

```python
def calcDistance(pointA, pointB, numOfFeature=4) -> float:
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2  # Euclidean distance
    return math.sqrt(tmp)
```

### Find k nearest neighbor

```python
def kNearestNeighbor(trainSet, point, k) -> list:
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "distance": calcDistance(point, item[:-1])
        })
    distances.sort(key=lambda x: x["distance"])
    labels = [item["label"] for item in distances]
    return labels[:k]
```

### Find most occur label

```python
def findMostOccur(arr) -> str:
    labels = set(arr)  # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans
```

### Main

```python
if __name__ == "__main__":
    trainSet, testSet = load_data("iris.csv")
    correct = 0
    for item in testSet:
        knn = kNearestNeighbor(trainSet, item, 5)  # k = 5: 5 nearest neighbor
        answer = findMostOccur(knn)
        print("label: {} -> predicted: {}".format(item[-1], answer))
        correct += item[-1] == answer

    print("Accuracy: {}%".format(correct / len(testSet) * 100))
```

## Result

```
Examples output:
label: Iris-virginica -> predicted: Iris-virginica
label: Iris-versicolor -> predicted: Iris-versicolor
label: Iris-versicolor -> predicted: Iris-versicolor
label: Iris-versicolor -> predicted: Iris-versicolor
label: Iris-versicolor -> predicted: Iris-versicolor
```

## Accuracy

```
Accuracy above 90%
```
