# Classification

> Lazy re-implementation. 有关 generative model 部分是抄的
> TODO：手写 Adam

**reference**: [ML 2020 Spring - Classification](https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C)

[data.tar.gz](https://drive.google.com/uc?id=1KSFIRh0-_Vr7SdiSCZP1ItV7bXPxMD92)

下载后需要 `tar -zxvf data.tar.gz`

学到了几点：

- [numpy 的 *, np.dot, np.matmul, np.multiply, @](https://blog.csdn.net/FrankieHello/article/details/103510118)
- [numpy 的 np.random.shuffle 是对传入的 index 直接操作, return None](https://blog.csdn.net/jasonzzj/article/details/53932645)
- [python 风格规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)
- [numpy 的 broadcast](https://zhuanlan.zhihu.com/p/60365398)
- cross entropy 的乘法是 np.dot，求 gradient 用 error 乘 X 算梯度因为要对每个 feature 求偏导，所以乘法是 np.multiply or \*

## Logistic regression

### Load the dataset

```python
import numpy as np

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_raw_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_raw_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_raw_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

# 54256 training size, 27622 test size, 510 feature dimensions
print(X_raw_train.shape)
print(Y_raw_train.shape)
print(X_raw_test.shape)
```

    (54256, 510)
    (54256,)
    (27622, 510)

### Data preprocessing

- normalization
- train_dev_split

```python
class Args(object):

    def __init__(self):

        self.dev_ratio = 0.1
        self.max_iter = 10
        self.batch_size = 8
        self.learning_rate = 0.2
        self.mode = "adaptive_lr"
        self.seed = 0


args = Args()
np.random.seed(args.seed)
```

```python
class ProcessDataset(object):
    """Data preprocessing

    reference: https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C#scrollTo=T3QjToT_Sq9J

    Attributes:
        X_train:
            numpy training set with size (train_size, feature_dimension)
        Y_train:
            numpy training label with size (train_size,)
        X_test:
            numpy testing set with size (test_size, feature_dimension)
        dev_ratio:
            development set ratio. If 'None', there will be no dev set
        specified_columns:
            indexes of the columns that will be normalized. Default: all columns. 
            If 'None', no columns will be normalized.
    """
    
    def __init__(self, X_train, Y_train, X_test, dev_ratio, specified_columns=np.arange(X_train.shape[1])):
        """Initialize dataset with normalization and dev split.
        """

        self.X_train, self.Y_train, self.X_test = X_train, Y_train, X_test
        self.dev_ratio = dev_ratio
        self.specified_columns = specified_columns


    def _normalization(self):
        """Normalization
        """

        specified_columns = self.specified_columns

        if specified_columns is None:
            return self.X_train, self.X_test
        
        # Training normalization
        X_mean = np.mean(self.X_train[:, specified_columns], axis=0)
        X_std = np.std(self.X_train[:, specified_columns], axis=0)

        X_train = (self.X_train - X_mean) / (X_std + 1e-8)
        
        # Testing normalization
        X_test = (self.X_test - X_mean) / (X_std + 1e-8)

        return X_train, X_test

    
    def _train_dev_split(self, X_train, Y_train):
        """Train dev split.
        """

        if self.dev_ratio == None:
            return X_train, Y_train, None, None

        train_size = int(X_train.shape[0]*(1-self.dev_ratio))

        return X_train[:train_size], Y_train[:train_size], X_train[train_size:], Y_train[train_size:]


    def process_dataset(self):

        X_train, X_test = self._normalization()
        Y_train = self.Y_train
        X_train, Y_train, X_dev, Y_dev = self._train_dev_split(X_train, Y_train)
        return X_train, X_dev, Y_train, Y_dev, X_test


dataset = ProcessDataset(X_raw_train, Y_raw_train, X_raw_test, dev_ratio=args.dev_ratio)
X_train, X_dev, Y_train, Y_dev, X_test = dataset.process_dataset()

# check correctness. It should be 48830, 5426, 27622, 510
train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))
```

    Size of training set: 48830
    Size of development set: 5426
    Size of testing set: 27622
    Dimension of data: 510

### Train

```python
class LogisticRegression(object):
    """Logistic regression

    reference: https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C#scrollTo=T3QjToT_Sq9J

    Attributes:
        max_iter:
            max iteration for training; default=10
        batch_size:
            mini batch size; default=8
        learning_rate:
            default=0.2
        mode:
            gradient descent mode; "adaptive_lr" and "Adam" are available;
            default="adaptive_lr"
        w:
            weight parameter, shape = [data dimension, ]
        b:
            bias parameter, scalar
    """

    def __init__(self, max_iter=10, batch_size=8, learning_rate=0.2, mode="adaptive_lr"):
        
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mode = mode
        self.w = None
        self.b = None


    def _shuffle(self, X, Y):
        """This function shuffles two equal-length list/array, X and Y, together."""

        index = np.arange(X.shape[0])
        np.random.shuffle(index)

        return X[index], Y[index]


    def _sigmoid(self, z):
        """Sigmoid function can be used to calculate probability.
        To avoid overflow, minimum/maximum output value is set.
        """

        return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1 - 1e-8)


    def _f(self, X, w, b):
        """This is the logistic regression function, parameterized by w and b

        Args:
            X: input data, shape = [batch_size, data_dimension]
            w: weight vector, shape = [data_dimension, ]
            b: bias, scalar

        Return:
            predicted probability of each row of X being positively labeled,
            shape = [batch_size, ]
        """

        return self._sigmoid(X @ w + b) # np.matmul is recommended instead of np.dot


    def _predict(self, X, w, b):
        """This function returns a truth value prediction for each row of X 
        by rounding the result of logistic regression function.

            Args:
                X: input data, shape = [batch_size, data_dimension]
                w: weight vector, shape = [data_dimension, ]
                b: bias, scalar

            Return:
                predicted label of each row of X; "1" means being positively labeled,
                shape = [batch_size, ]
        """

        return np.round(self._f(X, w, b)).astype(int)


    def _accuracy(self, Y_pred, Y_label):
        """This function calculates prediction accuracy"""

        return len(Y_pred[Y_pred == Y_label]) / len(Y_pred)


    def _cross_entropy_loss(self, Y_pred, Y_label):
        """This function computes the cross entropy.
        
        Args:
            Y_pred: probabilistic predictions, float vector
            Y_label: ground truth labels, bool vector

        Return:
            cross_entropy: cross entropy, scalar
        """

        return -1 * (np.dot(Y_label, np.log(Y_pred)) + np.dot(1-Y_label, np.log(1-Y_pred))) / Y_pred.shape[0]


    def _gradient(self, X, Y_label, w, b):
        """This function computes the gradient of cross entropy loss with respect to weight w and bias b.
        
        Args:
            X: input data, shape = [batch_size, data_dimension]
            Y_label: predict label, shape = [batch_size, ]
            w: weight vector, shape = [data_dimension, ]
            b: bias, scalar

        Return:
            w_grad: weight vector, shape = [data_dimension, ]
            b_grad: bias, scalar
        """

        error = Y_label - self._f(X, w, b)
        w_grad = -np.sum(error * X.T, axis=1)
        b_grad = -np.sum(error)

        return w_grad, b_grad


    def train(self, X_train, Y_train, X_dev, Y_dev):
        
        self.w = np.zeros((X_train.shape[1]))
        self.b = np.zeros((1))

        # Keep the loss and accuracy at every iteration for plotting
        train_loss = []
        dev_loss = []
        train_acc = []
        dev_acc = []
        step = 1

        for epoch in range(self.max_iter):
            # Random shuffle at the begging of each epoch
            X_train, Y_train = self._shuffle(X_train, Y_train)

            # Split mini batches
            for batch_idx in range(int(np.ceil(X_train.shape[0] / self.batch_size))):
                # The last batch may be smaller than the batch size
                if (batch_idx+1)*self.batch_size > X_train.shape[0]:
                    X, Y = X_train[batch_idx*self.batch_size:], Y_train[batch_idx*self.batch_size:]
                else:
                    X, Y = \
                        X_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size],\
                        Y_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                
                w_grad, b_grad = self._gradient(X, Y, self.w, self.b)

                if self.mode == "adaptive_lr":
                    self.w -= self.learning_rate * w_grad / np.sqrt(step)
                    self.b -= self.learning_rate * b_grad / np.sqrt(step)
                    step += 1

            # Compute accuracy and loss
            Y_train_pred, Y_dev_pred = self._predict(X_train, self.w, self.b), self._predict(X_dev, self.w, self.b)
            Y_train_prob, Y_dev_prob = self._f(X_train, self.w, self.b), self._f(X_dev, self.w, self.b)
            train_acc.append(self._accuracy(Y_train_pred, Y_train))
            train_loss.append(self._cross_entropy_loss(Y_train_prob, Y_train))
            dev_acc.append(self._accuracy(Y_dev_pred, Y_dev))
            dev_loss.append(self._cross_entropy_loss(Y_dev_prob, Y_dev))

        print('Training loss: {}'.format(train_loss[-1]))
        print('Development loss: {}'.format(dev_loss[-1]))
        print('Training accuracy: {}'.format(train_acc[-1]))
        print('Development accuracy: {}'.format(dev_acc[-1]))

        return train_loss, dev_loss, train_acc, dev_acc


    def plot_loss_curve(self, train_loss, dev_loss, train_acc, dev_acc):
        
        import matplotlib.pyplot as plt

        # Loss curve
        plt.plot(train_loss)
        plt.plot(dev_loss)
        plt.title('Loss')
        plt.legend(['train', 'dev'])
        plt.savefig('loss.png')
        plt.show()

        # Accuracy curve
        plt.plot(train_acc)
        plt.plot(dev_acc)
        plt.title('Accuracy')
        plt.legend(['train', 'dev'])
        plt.savefig('acc.png')
        plt.show()


    def test(self, X_test):
        # Predict testing labels
        predictions = self._predict(X_test, self.w, self.b)
        with open(output_fpath.format('logistic'), 'w') as f:
            f.write('id,label\n')
            for i, label in  enumerate(predictions):
                f.write('{},{}\n'.format(i, label))

        # Print out the most significant weights
        ind = np.argsort(np.abs(self.w))[::-1]
        with open(X_test_fpath) as f:
            content = f.readline().strip('\n').split(',')
        features = np.array(content)
        for i in ind[0:10]:
            print(features[i], self.w[i])


trainer = LogisticRegression(max_iter=args.max_iter,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    mode=args.mode
)
train_loss, dev_loss, train_acc, dev_acc = trainer.train(X_train, Y_train, X_dev, Y_dev)
```

    Training loss: 0.2717616051600512
    Development loss: 0.2965783188452641
    Training accuracy: 0.8836575875486381
    Development accuracy: 0.8724659049023221

### Plot loss curve

```python
trainer.plot_loss_curve(train_loss, dev_loss, train_acc, dev_acc)
```

![loss](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202206152215293.png)

![acc](https://raw.githubusercontent.com/Wind2375like/I-m_Ghost/main/img/202206152215565.png)

### Predict

```python
trainer.test(X_test)
```

     Wisconsin -1.9992005630443568
     Vietnam -1.3208115845688664
     Italy -1.2867251505441304
     Unemployed full-time 1.146077283557539
    num persons worked for employer 1.036322413897671
     Vietnam -0.985582361462992
     Grandchild 18+ never marr RP of subfamily -0.9529355508692217
     Child under 18 ever married -0.9426567124047583
     Trinadad&Tobago -0.8801246750567079
     1 0.8390511513114712

## Generative Model

### Data preprocessing

```python
dataset = ProcessDataset(X_raw_train, Y_raw_train, X_raw_test, dev_ratio=None)
X_train, _, Y_train, _, X_test = dataset.process_dataset()
```

### Directly generate model parameters

```python
class GenerativeModel(object):

    def __init__(self):

        self.w = None
        self.b = None

    def _compute_mean_and_var(self, X_train, Y_train):

        # Compute in-class mean
        X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
        X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

        mean_0 = np.mean(X_train_0, axis = 0)
        mean_1 = np.mean(X_train_1, axis = 0)  

        # Compute in-class covariance
        cov_0 = np.zeros((data_dim, data_dim))
        cov_1 = np.zeros((data_dim, data_dim))

        for x in X_train_0:
            cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
        for x in X_train_1:
            cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

        # Shared covariance is taken as a weighted average of individual in-class covariance.
        cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

        return X_train_0, X_train_1, mean_0, mean_1, cov


    def _sigmoid(self, z):
        """Sigmoid function can be used to calculate probability.
        To avoid overflow, minimum/maximum output value is set.
        """

        return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1 - 1e-8)


    def _f(self, X, w, b):
        """This is the logistic regression function, parameterized by w and b

        Args:
            X: input data, shape = [batch_size, data_dimension]
            w: weight vector, shape = [data_dimension, ]
            b: bias, scalar

        Return:
            predicted probability of each row of X being positively labeled,
            shape = [batch_size, ]
        """

        return self._sigmoid(X @ w + b) # np.matmul is recommended instead of np.dot


    def _predict(self, X, w, b):
        """This function returns a truth value prediction for each row of X 
        by rounding the result of logistic regression function.

            Args:
                X: input data, shape = [batch_size, data_dimension]
                w: weight vector, shape = [data_dimension, ]
                b: bias, scalar

            Return:
                predicted label of each row of X; "1" means being positively labeled,
                shape = [batch_size, ]
        """

        return np.round(self._f(X, w, b)).astype(int)


    def _accuracy(self, Y_pred, Y_label):
        """This function calculates prediction accuracy"""

        return len(Y_pred[Y_pred == Y_label]) / len(Y_pred)


    def train(self, X_train, Y_train):

        X_train_0, X_train_1, mean_0, mean_1, cov = self._compute_mean_and_var(X_train, Y_train)

        # Compute inverse of covariance matrix.
        # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
        # Via SVD decomposition, one can get matrix inverse efficiently and accurately.      
        u, s, v = np.linalg.svd(cov, full_matrices=False)
        inv = np.matmul(v.T * 1 / s, u.T)

        # Directly compute weights and bias
        self.w = np.dot(inv, mean_0 - mean_1)
        self.b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
            + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

        # Compute accuracy on training set
        Y_train_pred = 1 - self._predict(X_train, self.w, self.b)
        print('Training accuracy: {}'.format(self._accuracy(Y_train_pred, Y_train)))


    def test(self, X_test):

        # Predict testing labels
        predictions = 1 - self._predict(X_test, self.w, self.b)
        with open(output_fpath.format('generative'), 'w') as f:
            f.write('id,label\n')
            for i, label in  enumerate(predictions):
                f.write('{},{}\n'.format(i, label))

        # Print out the most significant weights
        ind = np.argsort(np.abs(self.w))[::-1]
        with open(X_test_fpath) as f:
            content = f.readline().strip('\n').split(',')
        features = np.array(content)
        for i in ind[0:10]:
            print(features[i], self.w[i])


trainer = GenerativeModel()
trainer.train(X_train, Y_train)
```

    Training accuracy: 0.8721984665290475

### Predict

```python
trainer.test(X_test)
```

     Retail trade 8.0625
     34 -6.02734375
     37 -5.60546875
     Other service -5.24609375
     Other Rel 18+ never marr RP of subfamily -4.921875
     Abroad -4.0
     3 4.0
     3 -3.6875
     Abroad 3.59375
     Tennessee 3.0234375
