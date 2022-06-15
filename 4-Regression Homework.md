# Regression Homework

> Lazy re-implementation.
> 并没有做到看过代码之后自己从零写出来，在 preprocessing 的 indexing 和 adagrad 那里有点反应不过来，导致需要重新看一下源码；submit 部分抄的代码；没有进行任何优化不过让不少参数变成可调节了，设置了 cross validation。

**reference**: [ML 2020 Spring - Regression](https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C)

[hw1_data.zip](https://drive.google.com/uc?id=1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm)

## Load the dataset

### Unzip data

```python
import os
os.system('unzip hw1_data.zip')
```

```plain
Archive:  hw1_data.zip
    inflating: test.csv
    inflating: train.csv

0
```

### Read csv

```python
import pandas as pd
pd.set_option('display.max_columns', None)
pd.read_csv('train.csv', nrows=18, encoding='big5')
```

<div>
<style scoped>
.dataframe tbody tr th:only-of-type {
    vertical-align: middle;
}

.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>日期</th>
      <th>測站</th>
      <th>測項</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>AMB_TEMP</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>13</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>15</td>
      <td>17</td>
      <td>20</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>21</td>
      <td>19</td>
      <td>17</td>
      <td>16</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>CH4</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>CO</td>
      <td>0.51</td>
      <td>0.41</td>
      <td>0.39</td>
      <td>0.37</td>
      <td>0.35</td>
      <td>0.3</td>
      <td>0.37</td>
      <td>0.47</td>
      <td>0.78</td>
      <td>0.74</td>
      <td>0.59</td>
      <td>0.52</td>
      <td>0.41</td>
      <td>0.4</td>
      <td>0.37</td>
      <td>0.37</td>
      <td>0.47</td>
      <td>0.69</td>
      <td>0.56</td>
      <td>0.45</td>
      <td>0.38</td>
      <td>0.35</td>
      <td>0.36</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>NMHC</td>
      <td>0.2</td>
      <td>0.15</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>0.11</td>
      <td>0.06</td>
      <td>0.1</td>
      <td>0.13</td>
      <td>0.26</td>
      <td>0.23</td>
      <td>0.2</td>
      <td>0.18</td>
      <td>0.12</td>
      <td>0.11</td>
      <td>0.1</td>
      <td>0.13</td>
      <td>0.14</td>
      <td>0.23</td>
      <td>0.18</td>
      <td>0.12</td>
      <td>0.1</td>
      <td>0.09</td>
      <td>0.1</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>NO</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>1.7</td>
      <td>1.8</td>
      <td>1.5</td>
      <td>1.9</td>
      <td>2.2</td>
      <td>6.6</td>
      <td>7.9</td>
      <td>4.2</td>
      <td>2.9</td>
      <td>3.4</td>
      <td>3</td>
      <td>2.5</td>
      <td>2.2</td>
      <td>2.5</td>
      <td>2.3</td>
      <td>2.1</td>
      <td>1.9</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>1.8</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>NO2</td>
      <td>16</td>
      <td>9.2</td>
      <td>8.2</td>
      <td>6.9</td>
      <td>6.8</td>
      <td>3.8</td>
      <td>6.9</td>
      <td>7.8</td>
      <td>15</td>
      <td>21</td>
      <td>14</td>
      <td>11</td>
      <td>14</td>
      <td>12</td>
      <td>11</td>
      <td>11</td>
      <td>22</td>
      <td>28</td>
      <td>19</td>
      <td>12</td>
      <td>8.1</td>
      <td>7</td>
      <td>6.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>NOx</td>
      <td>17</td>
      <td>9.8</td>
      <td>8.7</td>
      <td>8.6</td>
      <td>8.5</td>
      <td>5.3</td>
      <td>8.8</td>
      <td>9.9</td>
      <td>22</td>
      <td>29</td>
      <td>18</td>
      <td>14</td>
      <td>17</td>
      <td>15</td>
      <td>14</td>
      <td>13</td>
      <td>25</td>
      <td>30</td>
      <td>21</td>
      <td>13</td>
      <td>9.7</td>
      <td>8.6</td>
      <td>8.7</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>O3</td>
      <td>16</td>
      <td>30</td>
      <td>27</td>
      <td>23</td>
      <td>24</td>
      <td>28</td>
      <td>24</td>
      <td>22</td>
      <td>21</td>
      <td>29</td>
      <td>44</td>
      <td>58</td>
      <td>50</td>
      <td>57</td>
      <td>65</td>
      <td>64</td>
      <td>51</td>
      <td>34</td>
      <td>33</td>
      <td>34</td>
      <td>37</td>
      <td>38</td>
      <td>38</td>
      <td>36</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>PM10</td>
      <td>56</td>
      <td>50</td>
      <td>48</td>
      <td>35</td>
      <td>25</td>
      <td>12</td>
      <td>4</td>
      <td>2</td>
      <td>11</td>
      <td>38</td>
      <td>56</td>
      <td>64</td>
      <td>56</td>
      <td>57</td>
      <td>52</td>
      <td>51</td>
      <td>66</td>
      <td>85</td>
      <td>85</td>
      <td>63</td>
      <td>46</td>
      <td>36</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>PM2.5</td>
      <td>26</td>
      <td>39</td>
      <td>36</td>
      <td>35</td>
      <td>31</td>
      <td>28</td>
      <td>25</td>
      <td>20</td>
      <td>19</td>
      <td>30</td>
      <td>41</td>
      <td>44</td>
      <td>33</td>
      <td>37</td>
      <td>36</td>
      <td>45</td>
      <td>42</td>
      <td>49</td>
      <td>45</td>
      <td>44</td>
      <td>41</td>
      <td>30</td>
      <td>24</td>
      <td>13</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>RAINFALL</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
      <td>NR</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>RH</td>
      <td>77</td>
      <td>68</td>
      <td>67</td>
      <td>74</td>
      <td>72</td>
      <td>73</td>
      <td>74</td>
      <td>73</td>
      <td>66</td>
      <td>56</td>
      <td>45</td>
      <td>37</td>
      <td>40</td>
      <td>42</td>
      <td>47</td>
      <td>49</td>
      <td>56</td>
      <td>67</td>
      <td>72</td>
      <td>69</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>SO2</td>
      <td>1.8</td>
      <td>2</td>
      <td>1.7</td>
      <td>1.6</td>
      <td>1.9</td>
      <td>1.4</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>5.1</td>
      <td>15</td>
      <td>4.5</td>
      <td>2.7</td>
      <td>3.5</td>
      <td>3.6</td>
      <td>3.9</td>
      <td>4.4</td>
      <td>9.9</td>
      <td>5.1</td>
      <td>3.4</td>
      <td>2.3</td>
      <td>2</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>THC</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.8</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>2.1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>2.1</td>
      <td>2</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>WD_HR</td>
      <td>37</td>
      <td>80</td>
      <td>57</td>
      <td>76</td>
      <td>110</td>
      <td>106</td>
      <td>101</td>
      <td>104</td>
      <td>124</td>
      <td>46</td>
      <td>241</td>
      <td>280</td>
      <td>297</td>
      <td>305</td>
      <td>307</td>
      <td>304</td>
      <td>307</td>
      <td>124</td>
      <td>118</td>
      <td>121</td>
      <td>113</td>
      <td>112</td>
      <td>106</td>
      <td>110</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>WIND_DIREC</td>
      <td>35</td>
      <td>79</td>
      <td>2.4</td>
      <td>55</td>
      <td>94</td>
      <td>116</td>
      <td>106</td>
      <td>94</td>
      <td>232</td>
      <td>153</td>
      <td>283</td>
      <td>269</td>
      <td>290</td>
      <td>316</td>
      <td>313</td>
      <td>305</td>
      <td>291</td>
      <td>124</td>
      <td>119</td>
      <td>118</td>
      <td>114</td>
      <td>108</td>
      <td>102</td>
      <td>111</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>WIND_SPEED</td>
      <td>1.4</td>
      <td>1.8</td>
      <td>1</td>
      <td>0.6</td>
      <td>1.7</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>2</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>1.6</td>
      <td>1.9</td>
      <td>2.1</td>
      <td>3.3</td>
      <td>2.5</td>
      <td>2.2</td>
      <td>1.4</td>
      <td>2.2</td>
      <td>2.8</td>
      <td>3</td>
      <td>2.6</td>
      <td>2.7</td>
      <td>2.1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2014/1/1</td>
      <td>豐原</td>
      <td>WS_HR</td>
      <td>0.5</td>
      <td>0.9</td>
      <td>0.6</td>
      <td>0.3</td>
      <td>0.6</td>
      <td>1.9</td>
      <td>2</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>1.2</td>
      <td>2</td>
      <td>2.6</td>
      <td>2.1</td>
      <td>2.1</td>
      <td>1.9</td>
      <td>1</td>
      <td>2.5</td>
      <td>2.5</td>
      <td>2.8</td>
      <td>2.6</td>
      <td>2.4</td>
      <td>2.3</td>
    </tr>
  </tbody>
</table>
</div>

### Preprocessing

```python
# preprocessing

pd.options.mode.chained_assignment = None
raw_data = pd.read_csv('train.csv', encoding='big5')
data = raw_data.iloc[:, 3:] # remove the first three columns
data[data == 'NR'] = 0      # the rainfall is 'NR' (not rain) or numbers (amount), convert to zero
data = data.to_numpy()
data.shape
```

```plain
(4320, 24)
```

## Extract features

### Declare a 18 dim vector

![fig1](https://drive.google.com/uc?id=1LyaqD4ojX07oe5oDzPO99l9ts5NRyArH)

![fig2](https://drive.google.com/uc?id=1ZroBarcnlsr85gibeqEF-MtY13xJTG47)

“將原始 4320 \* 18 的資料依照每個月分重組成 12 個 18 (features) \* 480 (hours) 的資料。” 这里有误，应为 4320 * 24

### Declare x for previous 9-hr data and y for the 10th hr pm2.5

![fig3](https://drive.google.com/uc?id=1wKoPuaRHoX682LMiBgIoOP4PDyNKsJLK)

```python
# declare a 18 dim vector
# (18*20*12, 24) to 12*(18, 20*24)

import numpy as np

month_data = []

for month in range(12):
    sample = np.zeros((18, 480))

    for day in range(20):   # 20 days in a month
        sample[:, day * 24 : (day + 1) * 24] = data[(month * 20 + day) * 18 : (month * 20 + day + 1) * 18, :]

    month_data.append(sample)

# declare x for previous 9-hr data and y for the 10th hr pm2.5
# 12*(18, 20*24) to (12*(480-9), 18*9) and (12*(480-9), 1)

previous_hours = 9
x, y = np.zeros((12*(480-previous_hours), 18*previous_hours)), np.zeros((12*(480-previous_hours), 1))

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour >= 24-previous_hours:
                continue

            x[month * 471 + day * 24 + hour, :] = \
                month_data[month][:, day * 24 + hour : day * 24 + hour + previous_hours].reshape(1, -1) # (18, 9) to (1, 18*9)
            y[month * 471 + day * 24 + hour, 0] = \
                month_data[month][9, day * 24 + hour + previous_hours]  # pm2.5 is the 10th feature

print(x)
print(y)

```

```plain
[[14.  14.  14.  ...  2.   2.   0.5]
[14.  14.  13.  ...  2.   0.5  0.3]
[14.  13.  12.  ...  0.5  0.3  0.8]
...
[17.  18.  19.  ...  1.1  1.4  1.3]
[18.  19.  18.  ...  1.4  1.3  1.6]
[19.  18.  17.  ...  1.3  1.6  1.8]]
[[30.]
[41.]
[44.]
...
[17.]
[24.]
[29.]]
```

## Normalize

```python
# normalize

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
eps = 1e-8

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x_std[j] == 0:
            x[i, j] = (x[i, j] - x_mean[j]) / (x_std[j] + eps)
        else:
            x[i, j] = (x[i, j] - x_mean[j]) / x_std[j]

print(x)
```

```plain
[[-1.35825331 -1.35883937 -1.359222   ...  0.26650729  0.2656797
-1.14082131]
[-1.35825331 -1.35883937 -1.51819928 ...  0.26650729 -1.13963133
-1.32832904]
[-1.35825331 -1.51789368 -1.67717656 ... -1.13923451 -1.32700613
-0.85955971]
...
[-0.88092053 -0.72262212 -0.56433559 ... -0.57693779 -0.29644471
-0.39079039]
[-0.7218096  -0.56356781 -0.72331287 ... -0.29578943 -0.39013211
-0.1095288 ]
[-0.56269867 -0.72262212 -0.88229015 ... -0.38950555 -0.10906991
0.07797893]]
```

## Cross Validation

```python
# cross validation

import math

portion = 0.8

x_train_set = x[: math.floor(len(x) * portion), :]
y_train_set = y[: math.floor(len(y) * portion), :]

# train

dim = 18 * previous_hours + 1
w = np.random.rand(dim, 1)  # including bias
x_train_set = np.concatenate((np.ones((x_train_set.shape[0], 1)), x_train_set), axis=1).astype(float)
epoch = 1000
lr = 1
eps = 1e-10
adagrad = np.zeros((dim, 1))

for e in range(epoch):
    loss = np.sqrt(np.sum((np.dot(x_train_set, w) - y_train_set) ** 2) / x.shape[0])    # rmse
    if (e % 100 == 0):
        print(str(e) + ':' + str(loss))
    gradient = 2 * np.dot(-1 * x_train_set.transpose(), y_train_set - np.dot(x_train_set, w))
    adagrad += gradient ** 2
    w -= lr * gradient / np.sqrt(adagrad + eps)

print('training loss: ' + str(loss))
```

```plain
0:26.7311611433129
100:8.286660338939111
200:6.302492077078951
300:5.735051517334312
400:5.5352739869566765
500:5.438448973594182
600:5.377568807111021
700:5.33358918971557
800:5.29966642679888
900:5.272580388877657
training loss: 5.250678162201367
```

```python
# validation

x_validation = x[math.floor(len(x) * portion): , :]
y_validation = y[math.floor(len(y) * portion): , :]

x_validation = np.concatenate((np.ones((x_validation.shape[0], 1)), x_validation), axis=1).astype(float)

loss = np.sqrt(np.sum((np.dot(x_validation, w) - y_validation) ** 2) / x_validation.shape[0])
print('validation loss: ' + str(loss))
```

```plain
validation loss: 5.8774181106481915
```

## Test

```python
# preprocessing

testdata = pd.read_csv('test.csv', header = None, encoding = 'big5')
raw_test_data = testdata.iloc[:, 2:]
raw_test_data[raw_test_data == 'NR'] = 0
test_data = raw_test_data.to_numpy()
x_test = np.zeros((240, 18*previous_hours), dtype = float)

for i in range(240):
    x_test[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
    
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        if x_std[j] == 0:
            x_test[i, j] = (x_test[i, j] - x_mean[j]) / (x_std[j] + eps)
        else:
            x_test[i, j] = (x_test[i, j] - x_mean[j]) / x_std[j]

x_test = np.concatenate((np.ones((240, 1)), x_test), axis = 1).astype(float)

# prediction

y_test = np.dot(x_test, w)
print(y_test)

# save prediction

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), y_test[i][0]]
        csv_writer.writerow(row)
        print(row)
```

```plain
    [[  5.22083623]
     [ 17.18939023]
     [ 24.6340404 ]
     [  7.77080339]
     [ 28.45372693]
     [ 22.03379111]
     [ 23.3233716 ]
     [ 31.74891464]
     [ 17.66334952]
     [ 58.80552944]
     [ 14.69956879]
     [  9.97226546]
     [ 61.10181171]
     [ 52.79759841]
     [ 21.76444391]
     [ 11.67242285]
     [ 31.69340543]
     [ 67.09161642]
     [ -1.41238928]
     [ 16.47365557]
     [ 42.35219554]
     [ 71.31900323]
     [  9.32399642]
     [ 17.82534034]
     [ 15.3417354 ]
     [ 38.18578205]
     [ 13.77034656]
     [ 73.11448987]
     [  7.0693704 ]
     [ 55.5771634 ]
     [ 22.86338816]
     [  9.02904154]
     [  3.3439134 ]
     [ 21.6560789 ]
     [ 31.1640635 ]
     [ 36.50803296]
     [ 42.73709168]
     [ 30.63894184]
     [ 43.38058156]
     [ 35.99856664]
     [  6.49048489]
     [ 40.10116521]
     [ 31.75192751]
     [ 50.41115144]
     [ 16.44502043]
     [ 35.27003235]
     [ 24.25212151]
     [  9.82245518]
     [ 23.41532802]
     [ 32.06349548]
     [ 22.44311073]
     [  9.2624354 ]
     [ 22.19708462]
     [ 53.62480562]
     [ 17.02233431]
     [ 35.19712746]
     [ 31.55457457]
     [ 23.39244416]
     [ 58.78042478]
     [ 21.08618755]
     [ 16.28759516]
     [ 40.77887035]
     [ 12.39271934]
     [ 49.38695891]
     [ 13.95883275]
     [ 15.49247984]
     [ 14.18444168]
     [ -0.18608462]
     [ 44.14407896]
     [ 32.51579667]
     [ 19.58247975]
     [ 37.83350646]
     [ 59.81036101]
     [  4.94281069]
     [ 16.53075214]
     [  6.16768154]
     [ 40.55223223]
     [ 15.41976965]
     [ 23.61552198]
     [ 23.05300507]
     [ 25.19647212]
     [ 38.90492748]
     [ 23.87705432]
     [ 87.76223558]
     [ 36.83738573]
     [ 27.25792508]
     [ 24.35255732]
     [ 29.70032728]
     [ 25.28887055]
     [ 21.03376927]
     [ 30.02405639]
     [ 40.21235053]
     [  4.45529579]
     [ 37.50175629]
     [ 45.82055949]
     [ 15.85998189]
     [ 34.90395718]
     [ 11.86165552]
     [ 25.03068123]
     [  5.19468176]
     [ 18.43402343]
     [ 28.94634995]
     [ 11.85216098]
     [ 16.17679073]
     [ 24.53453828]
     [ 38.46081252]
     [ 31.07511063]
     [  6.56614875]
     [  8.53942711]
     [ 77.49361463]
     [ 45.29508984]
     [ 15.33449179]
     [ 29.12462764]
     [ 16.25483331]
     [ 14.14354779]
     [ 25.36864651]
     [ 24.34035039]
     [ 10.44334821]
     [ 17.29241471]
     [ 19.28581748]
     [ 81.32635954]
     [ 25.70198565]
     [ 34.44566265]
     [ 25.14874744]
     [  8.39327328]
     [ 41.61581212]
     [ 11.20641757]
     [ 19.06299625]
     [ 28.54087966]
     [ 62.61854955]
     [ 23.07182843]
     [ 22.12422579]
     [ 59.22690277]
     [ 14.35237811]
     [ 14.51467893]
     [  2.92777894]
     [ 11.54232962]
     [ 58.59016509]
     [ 19.09885984]
     [  5.53874094]
     [ 27.47685845]
     [ 25.3992135 ]
     [ 42.80453346]
     [ 31.73622946]
     [ 17.31521826]
     [ 26.48092401]
     [ 10.90505147]
     [ 51.95578888]
     [ 22.3849984 ]
     [ 38.3643975 ]
     [  9.19143359]
     [  7.17402375]
     [ 25.0834489 ]
     [  6.68393928]
     [ 14.81149764]
     [ 41.70695633]
     [  8.66508213]
     [ 36.71385752]
     [ 11.23925161]
     [ 19.42896198]
     [ 41.28724254]
     [ 18.66546503]
     [ 11.46132244]
     [  7.61655283]
     [ 52.74863916]
     [ 29.60602591]
     [ -1.21287924]
     [ 15.94699833]
     [ 63.13295959]
     [ 13.48367878]
     [ 65.08139766]
     [ 40.38705338]
     [ 25.40122016]
     [ 20.2608441 ]
     [ 62.21928836]
     [ 24.06844173]
     [ 20.62921461]
     [ 36.73433253]
     [ 11.95546524]
     [ 30.05657859]
     [ 16.79762899]
     [ 11.25443691]
     [ 55.06169132]
     [ 45.29349129]
     [ 17.7181506 ]
     [ 34.99191726]
     [ 26.47871195]
     [ 69.00465922]
     [ 10.1747062 ]
     [ 57.46654017]
     [ 37.51641094]
     [ 15.36091264]
     [ 29.39876037]
     [ -0.20063475]
     [ 19.57695115]
     [  1.15851532]
     [ 34.24207356]
     [ 10.51058106]
     [ 18.55705234]
     [ 62.1632704 ]
     [ 24.37271459]
     [ 23.02129029]
     [ 64.45178774]
     [ 10.85508409]
     [  9.49103805]
     [ 12.0218634 ]
     [  8.49890593]
     [  2.78954282]
     [120.8717655 ]
     [ 19.16415755]
     [ 15.38505493]
     [ 14.47074532]
     [ 35.36058463]
     [ 35.76924178]
     [ 20.31689683]
     [ 33.90473963]
     [ 77.19993641]
     [  0.80263064]
     [ 13.28909497]
     [ 33.8406981 ]
     [ 16.86785802]
     [ 12.16421738]
     [114.60500533]
     [ 12.31766703]
     [ 17.38521072]
     [ 60.7350164 ]
     [ 16.25570204]
     [ 19.59167051]
     [ 10.16550837]
     [  5.56606802]
     [ 44.24173353]
     [ 14.75659642]
     [ 50.48929527]
     [ 43.57079178]
     [ 23.45659246]
     [ 41.94965951]
     [ 67.70078732]
     [ 39.46580444]
     [ 15.10506988]
     [ 16.37495049]]
    ['id', 'value']
    ['id_0', 5.220836225084875]
    ['id_1', 17.189390232156867]
    ['id_2', 24.6340404031951]
    ['id_3', 7.770803390803701]
    ['id_4', 28.453726926916463]
    ['id_5', 22.033791107509067]
    ['id_6', 23.323371601375392]
    ['id_7', 31.748914635949443]
    ['id_8', 17.663349520018897]
    ['id_9', 58.80552943799073]
    ['id_10', 14.699568794482264]
    ['id_11', 9.97226545528485]
    ['id_12', 61.101811708565165]
    ['id_13', 52.79759841287686]
    ['id_14', 21.764443914885316]
    ['id_15', 11.672422851765424]
    ['id_16', 31.69340542507091]
    ['id_17', 67.09161642102211]
    ['id_18', -1.41238927831305]
    ['id_19', 16.473655565237003]
    ['id_20', 42.35219553811542]
    ['id_21', 71.31900323247741]
    ['id_22', 9.323996420383011]
    ['id_23', 17.82534034089438]
    ['id_24', 15.341735400673008]
    ['id_25', 38.18578205008971]
    ['id_26', 13.770346555865428]
    ['id_27', 73.11448987313514]
    ['id_28', 7.069370403520663]
    ['id_29', 55.577163396990116]
    ['id_30', 22.86338815856776]
    ['id_31', 9.029041543549951]
    ['id_32', 3.3439134002497433]
    ['id_33', 21.656078902833432]
    ['id_34', 31.164063501969444]
    ['id_35', 36.508032962274285]
    ['id_36', 42.73709167807854]
    ['id_37', 30.638941840103158]
    ['id_38', 43.380581555533894]
    ['id_39', 35.99856663928361]
    ['id_40', 6.490484889853747]
    ['id_41', 40.10116520569568]
    ['id_42', 31.751927513322094]
    ['id_43', 50.411151442062746]
    ['id_44', 16.445020425518596]
    ['id_45', 35.270032347372876]
    ['id_46', 24.25212150741824]
    ['id_47', 9.822455180146566]
    ['id_48', 23.41532801501123]
    ['id_49', 32.06349547550335]
    ['id_50', 22.443110725342088]
    ['id_51', 9.262435403666966]
    ['id_52', 22.197084618155074]
    ['id_53', 53.624805622452776]
    ['id_54', 17.022334310887906]
    ['id_55', 35.19712746454358]
    ['id_56', 31.554574571835353]
    ['id_57', 23.39244415540127]
    ['id_58', 58.780424777625186]
    ['id_59', 21.086187549556207]
    ['id_60', 16.28759515840642]
    ['id_61', 40.77887035044823]
    ['id_62', 12.392719338900426]
    ['id_63', 49.38695890741114]
    ['id_64', 13.95883275257688]
    ['id_65', 15.492479844172022]
    ['id_66', 14.184441677143177]
    ['id_67', -0.18608461513939645]
    ['id_68', 44.14407896295079]
    ['id_69', 32.51579667081624]
    ['id_70', 19.582479747599873]
    ['id_71', 37.833506459185756]
    ['id_72', 59.810361009545844]
    ['id_73', 4.942810690515607]
    ['id_74', 16.530752137865388]
    ['id_75', 6.167681541400569]
    ['id_76', 40.55223223380302]
    ['id_77', 15.419769652392162]
    ['id_78', 23.615521979092783]
    ['id_79', 23.053005066023076]
    ['id_80', 25.19647212365758]
    ['id_81', 38.90492748383238]
    ['id_82', 23.877054315989383]
    ['id_83', 87.76223557917392]
    ['id_84', 36.83738573096214]
    ['id_85', 27.25792508250214]
    ['id_86', 24.35255732168399]
    ['id_87', 29.700327284708344]
    ['id_88', 25.288870545172415]
    ['id_89', 21.033769268801446]
    ['id_90', 30.02405639289306]
    ['id_91', 40.21235052892103]
    ['id_92', 4.455295789543474]
    ['id_93', 37.50175628540148]
    ['id_94', 45.82055948557037]
    ['id_95', 15.859981892232032]
    ['id_96', 34.903957182935414]
    ['id_97', 11.861655520318017]
    ['id_98', 25.030681226594638]
    ['id_99', 5.194681763997508]
    ['id_100', 18.43402343426718]
    ['id_101', 28.946349945407963]
    ['id_102', 11.852160977954657]
    ['id_103', 16.17679073486417]
    ['id_104', 24.53453827948239]
    ['id_105', 38.46081251878485]
    ['id_106', 31.075110626858216]
    ['id_107', 6.566148750747422]
    ['id_108', 8.539427111716801]
    ['id_109', 77.49361463181552]
    ['id_110', 45.29508984155545]
    ['id_111', 15.334491788586254]
    ['id_112', 29.124627644966655]
    ['id_113', 16.254833312630748]
    ['id_114', 14.143547792641211]
    ['id_115', 25.36864651065947]
    ['id_116', 24.340350390443078]
    ['id_117', 10.443348214452048]
    ['id_118', 17.29241470844039]
    ['id_119', 19.28581747625071]
    ['id_120', 81.3263595385092]
    ['id_121', 25.70198564956458]
    ['id_122', 34.44566265282955]
    ['id_123', 25.14874744094629]
    ['id_124', 8.393273282593944]
    ['id_125', 41.615812124540994]
    ['id_126', 11.206417574040074]
    ['id_127', 19.062996254892802]
    ['id_128', 28.54087966077174]
    ['id_129', 62.618549548303065]
    ['id_130', 23.07182843166408]
    ['id_131', 22.124225790689863]
    ['id_132', 59.22690276979412]
    ['id_133', 14.35237811440567]
    ['id_134', 14.514678926579876]
    ['id_135', 2.9277789437793125]
    ['id_136', 11.542329620603514]
    ['id_137', 58.59016508931779]
    ['id_138', 19.09885984180488]
    ['id_139', 5.5387409433122095]
    ['id_140', 27.476858454871078]
    ['id_141', 25.39921350314863]
    ['id_142', 42.80453346059244]
    ['id_143', 31.736229457877975]
    ['id_144', 17.315218261809846]
    ['id_145', 26.480924013856637]
    ['id_146', 10.905051467351537]
    ['id_147', 51.955788881088694]
    ['id_148', 22.38499839870123]
    ['id_149', 38.3643974951984]
    ['id_150', 9.191433588936984]
    ['id_151', 7.174023751042224]
    ['id_152', 25.083448898163226]
    ['id_153', 6.683939275285034]
    ['id_154', 14.811497644958182]
    ['id_155', 41.7069563276906]
    ['id_156', 8.66508213386911]
    ['id_157', 36.71385752178815]
    ['id_158', 11.239251609587019]
    ['id_159', 19.428961977826294]
    ['id_160', 41.2872425426047]
    ['id_161', 18.665465032039762]
    ['id_162', 11.461322442139526]
    ['id_163', 7.616552829855996]
    ['id_164', 52.74863915517254]
    ['id_165', 29.606025910472475]
    ['id_166', -1.2128792436493363]
    ['id_167', 15.946998329983]
    ['id_168', 63.13295959192602]
    ['id_169', 13.48367878387209]
    ['id_170', 65.08139765886857]
    ['id_171', 40.38705337585746]
    ['id_172', 25.401220159758292]
    ['id_173', 20.260844097503377]
    ['id_174', 62.2192883631234]
    ['id_175', 24.068441730435573]
    ['id_176', 20.62921461350205]
    ['id_177', 36.734332527566245]
    ['id_178', 11.955465238246246]
    ['id_179', 30.05657858504548]
    ['id_180', 16.797628990357495]
    ['id_181', 11.254436914749611]
    ['id_182', 55.06169131851283]
    ['id_183', 45.2934912917776]
    ['id_184', 17.71815059951079]
    ['id_185', 34.99191725770809]
    ['id_186', 26.478711946842]
    ['id_187', 69.00465921879001]
    ['id_188', 10.174706198837121]
    ['id_189', 57.4665401686308]
    ['id_190', 37.51641093953077]
    ['id_191', 15.360912636504505]
    ['id_192', 29.398760365624675]
    ['id_193', -0.20063474503901446]
    ['id_194', 19.57695115118182]
    ['id_195', 1.1585153178104441]
    ['id_196', 34.242073555902245]
    ['id_197', 10.51058106022673]
    ['id_198', 18.557052340343933]
    ['id_199', 62.163270402881395]
    ['id_200', 24.372714589341292]
    ['id_201', 23.021290285073558]
    ['id_202', 64.45178773843335]
    ['id_203', 10.855084093914549]
    ['id_204', 9.49103804839963]
    ['id_205', 12.02186340278562]
    ['id_206', 8.498905933975186]
    ['id_207', 2.789542824973708]
    ['id_208', 120.87176549853748]
    ['id_209', 19.164157552178374]
    ['id_210', 15.385054929962596]
    ['id_211', 14.470745318635787]
    ['id_212', 35.36058462887554]
    ['id_213', 35.76924177924023]
    ['id_214', 20.31689683223928]
    ['id_215', 33.90473962670855]
    ['id_216', 77.19993641108492]
    ['id_217', 0.802630643061157]
    ['id_218', 13.289094965179583]
    ['id_219', 33.84069810344862]
    ['id_220', 16.867858015593864]
    ['id_221', 12.16421737503118]
    ['id_222', 114.60500533433057]
    ['id_223', 12.317667033330206]
    ['id_224', 17.385210722434273]
    ['id_225', 60.73501640112582]
    ['id_226', 16.25570203873363]
    ['id_227', 19.59167051201742]
    ['id_228', 10.165508373351287]
    ['id_229', 5.5660680221605965]
    ['id_230', 44.24173352735038]
    ['id_231', 14.756596424230388]
    ['id_232', 50.48929527387388]
    ['id_233', 43.57079177635173]
    ['id_234', 23.45659246425452]
    ['id_235', 41.94965951286996]
    ['id_236', 67.70078732289717]
    ['id_237', 39.46580444434839]
    ['id_238', 15.105069882684894]
    ['id_239', 16.374950492779245]
```