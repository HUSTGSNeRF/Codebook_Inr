---
******noteId**: "6e337840e17a11ed8c01ed051df43427"
tags: [实验]

---
用sign激活函数

|      |  LR  |       -1        |     -2      |       -3        |       -4        |
| :--: | :--: | :-------------: | :---------: | :-------------: | :-------------: |
| ReLU | sgd  |   0.868977778   | 0.686222222 |     0.5424      |   0.525777778   |
|      | adam |   0.529333333   | 0.796666667 |      0.866      |   0.716755556   |
| Sin  | sgd  |   0.522755556   | 0.680755556 | **0.830888889** |   0.680711111   |
|      | adam |   0.513822222   | 0.524088889 |     0.6188      | **0.687111111** |
| Tanh | sgd  | **0.834044444** | 0.682444444 |   0.673333333   |   0.555822222   |



ReLU

#### 最后一层的激活函数

ReLU作为激活函数，使用softsign

会更低

test_acc=86.73%, train_loss=0.00373

sign 0.843466666666666



lr=0.5，sgd精度达到89%



D:\codebook_inr\run_saved\exp_10

#### 新数据集最高精度

D:\codebook_inr\run_saved\exp_71



#### 线性层加BN

只在最后一层加效果最好

D:\codebook_inr\run_saved



### 仿真实验：两个数据集

##### 消融实验 

PE

Act

Enc



##### model based对比试验

### 实测测实验







### class torch.nn.CrossEntropyLoss(weight=None, size_average=True)[[source\]](http://pytorch.org/docs/_modules/torch/nn/modules/loss.html#CrossEntropyLoss)

此标准将`LogSoftMax`和`NLLLoss`集成到一个类中。

当训练一个多类分类器的时候，这个方法是十分有用的。

- weight(tensor): `1-D` tensor，`n`个元素，分别代表`n`类的权重，如果你的训练样本很不均衡的话，是非常有用的。默认值为None。

调用时参数：

- input : 包含每个类的得分，`2-D` tensor,`shape`为 `batch*n`
- target: 大小为 `n` 的 `1—D` `tensor`，包含类别的索引(`0到 n-1`)。

Loss可以表述为以下形式：
$$
\begin{aligned}
loss(x, class) &= -\text{log}\frac{exp(x[class])}{\sum_j exp(x[j]))}\
               &= -x[class] + log(\sum_j exp(x[j]))
\end{aligned}
$$


当`weight`参数被指定的时候，`loss`的计算公式变为：
$$
loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))
$$
计算出的`loss`对`mini-batch`的大小取了平均。

形状(`shape`)：

- Input: (N,C) `C` 是类别的数量
- Target: (N) `N`是`mini-batch`的大小，0 <= targets[i] <= C-1

### class torch.nn.BCELoss(weight=None, size_average=True)[[source\]](http://pytorch.org/docs/_modules/torch/nn/modules/loss.html#BCELoss)

==`nn.functional.binary_cross_entropy_with_logits`

计算 `target` 与 `output` 之间的二进制交叉熵。 
$$
loss(o,t)=-\frac{1}{n}\sum_i(t[i] *log(o[i])+(1-t[i])* log(1-o[i]))
$$
如果`weight`被指定 ：
$$
loss(o,t)=-\frac{1}{n}\sum_iweights[i] *(t[i]* log(o[i])+(1-t[i])* log(1-o[i]))
$$
这个用于计算 `auto-encoder` 的 `reconstruction error`。注意 0<=target[i]<=1。

默认情况下，loss会基于`element`平均，如果`size_average=False`的话，`loss`会被累加。



### class torch.nn.MultiLabelMarginLoss(size_average=True)[[source\]](http://pytorch.org/docs/_modules/torch/nn/modules/loss.html#MultiLabelMarginLoss)

计算多标签分类的 `hinge loss`(`margin-based loss`) ，计算`loss`时需要两个输入： input x(`2-D mini-batch Tensor`)，和 output y(`2-D tensor`表示mini-batch中样本类别的索引)。
$$
loss(x, y) = \frac{1}{x.size(0)}\sum_{i=0,j=0}^{I,J}(max(0, 1 - (x[y[j]] - x[i])))
$$


其中 `I=x.size(0),J=y.size(0)`。对于所有的 `i`和 `j`，满足 $y[j]\neq0, i \neq y[j]$，`x` 和 `y` 必须具有同样的 `size`。

这个标准仅考虑了第一个非零 `y[j] targets` 此标准允许了，对于每个样本来说，可以有多个类别。

### class torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)[[source\]](http://pytorch.org/docs/_modules/torch/nn/modules/loss.html#MultiLabelSoftMarginLoss)

创建一个标准，基于输入x和目标y的 `max-entropy`，优化多标签 `one-versus-all` 的损失。`x`:2-D mini-batch Tensor;`y`:binary 2D Tensor。对每个mini-batch中的样本，对应的loss为：
$$
loss(x, y) = - \frac{1}{x.nElement()}\sum_{i=0}^I y[i]\text{log}\frac{exp(x[i])}{(1 + exp(x[i])}
                      + (1-y[i])\text{log}\frac{1}{1+exp(x[i])}
$$
其中 `I=x.nElement()-1`, $y[i] \in {0,1}$，`y` 和 `x`必须要有同样`size`
