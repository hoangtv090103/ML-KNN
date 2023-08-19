import csv
import numpy as np
import math


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


def calcDistance(pointA, pointB, numOfFeature=4) -> float:
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2  # Euclidean distance
    return math.sqrt(tmp)


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


if __name__ == "__main__":
    trainSet, testSet = load_data("iris.csv")
    correct = 0
    for item in testSet:
        knn = kNearestNeighbor(trainSet, item, 5)  # k = 5: 5 nearest neighbor
        answer = findMostOccur(knn)
        print("label: {} -> predicted: {}".format(item[-1], answer))
        correct += item[-1] == answer

    print("Accuracy: {}%".format(correct / len(testSet) * 100))
