## Thuật toán KNN

KNN là một thuật toán học máy có giám sát có thể được sử dụng để giải quyết cả vấn đề phân loại và hồi quy. Nó là một thuật toán học tập lười biếng vì nó không có giai đoạn đào tạo chuyên biệt và sử dụng tất cả dữ liệu để đào tạo trong khi phân loại. KNN là một thuật toán phi tham số vì nó không đưa ra bất kỳ giả định nào về phân phối dữ liệu bên dưới.

## Thuật toán KNN hoạt động như thế nào?

Thuật toán KNN giả định rằng các thứ tương tự tồn tại gần nhau. Nói cách khác, những thứ tương tự gần nhau. KNN nắm bắt ý tưởng về sự tương tự (đôi khi được gọi là khoảng cách, tính gần gũi hoặc sự gần gũi) với một số toán học mà chúng ta có thể đã học được khi còn nhỏ — tính toán khoảng cách giữa các điểm trên một đồ thị.

## Chúng ta chọn yếu tố K như thế nào?

Số lượng hàng xóm (K) là yếu tố quan trọng nhất trong KNN. K thường là số lẻ nếu số lớp là 2. Chọn K=1 không phải là ý tưởng hay vì một điểm dữ liệu nhiễu sẽ có tác động rất lớn. Chọn giá trị K cao hơn sẽ là một ý tưởng hay vì nó làm giảm tác động của nhiễu trên phân loại, nhưng không có gì đảm bảo rằng độ chính xác tổng thể sẽ tăng lên. Cách tốt nhất để tìm giá trị K tối ưu là thử các giá trị K khác nhau và kiểm tra độ chính xác cho mỗi giá trị K.

## Cách để tìm giá trị K tối ưu?

Chúng ta có thể tìm giá trị K tối ưu bằng cách sử dụng phương pháp kiểm tra chéo. Phương pháp kiểm tra chéo là một phương pháp đánh giá hiệu suất mô hình. Nó được sử dụng để đánh giá hiệu suất của một mô hình học máy trên dữ liệu không được sử dụng để đào tạo mô hình. Phương pháp kiểm tra chéo là một phương pháp đánh giá hiệu suất mô hình. Nó được sử dụng để đánh giá hiệu suất của một mô hình học máy trên dữ liệu không được sử dụng để đào tạo mô hình. Phương pháp kiểm tra chéo là một phương pháp đánh giá hiệu suất mô hình. Nó được sử dụng để đánh giá hiệu suất của một mô hình học máy trên dữ liệu không được sử dụng để đào tạo mô hình.

## Chúng ta tính khoảng cách giữa hai điểm như thế nào?

Khoảng cách giữa hai điểm được tính bằng khoảng cách Euclide. Tất cả các điểm được vẽ trên một đồ thị và khoảng cách được tính giữa hai điểm bằng cách sử dụng công thức khoảng cách Euclide.

**Công thức khoảng cách Euclide**

$$
d(p,q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + ... + (p_n - q_n)^2}
$$

trong đó p và q là hai điểm, p1 và q1 là tọa độ x, p2 và q2 là tọa độ y, và v.v.

## Thư viện sử dụng

```python
import csv
import numpy as np
import math
```

### Tải dữ liệu

```python
def load_data(path='') -> tuple:
f = open(path, "r")
data = csv.reader(f) # định dạng csv
data = np.array(list(data)) # chuyển đổi thành ma trận
data = np.delete(data, 0, 0) # xóa header
data = np.delete(data, 0, 1) # xóa chỉ số
np.random.shuffle(data) # xáo trộn dữ liệu
f.close()
data_length = len(data)
trainSet = data[:int(0.8 * data_length)] # 80% là dữ liệu đào tạo
testSet = data[int(0.8 * data_length):] # 20% là dữ liệu kiểm tra
return trainSet, testSet
```

## Tính khoảng cách

```python
def calcDistance(pointA, pointB, numOfFeature=4) -> float:
tmp = 0
for i in range(numOfFeature):
tmp += (float(pointA[i]) - float(pointB[i])) ** 2 # khoảng cách Euclide
return math.sqrt(tmp)
```

## Tìm k hàng xóm gần nhất

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

## Tìm nhãn xuất hiện nhiều nhất

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

## Hàm main

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

## Kết quả

```
Ví dụ:
label: Iris-setosa -> predicted: Iris-setosa
label: Iris-setosa -> predicted: Iris-setosa
label: Iris-setosa -> predicted: Iris-setosa
label: Iris-setosa -> predicted: Iris-setosa
label: Iris-setosa -> predicted: Iris-setosa
```

## Độ chính xác

```
Độ chính xác tối thiểu: 90%
```
