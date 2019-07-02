<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
## 梯度下降参数更行规则
### 1.权重学习规则<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;\omega\rightarrow&space;\omega&space;-&space;\eta\frac{\partial&space;C}{\partial&space;\omega}" title="\large \omega\rightarrow \omega - \eta\frac{\partial C}{\partial \omega}" /><br>
### 2.偏置学习规则<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;b\rightarrow&space;b-&space;\eta\frac{\partial&space;C}{\partial&space;b}" title="\large b\rightarrow b- \eta\frac{\partial C}{\partial b}" /><br>
## 反向传播核心方程

### 1. 输出层误差向量的每个元素如下：
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\delta^L_j&space;=\frac{\partial&space;C}{\partial&space;a^L_j}&space;{\sigma}'(z^L_j)" title="\LARGE \delta^L_j =\frac{\partial C}{\partial a^L_j} {\sigma}'(z^L_j)" /><br>
#### 向量形式：
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\delta^L&space;=\triangledown_a&space;C&space;\odot&space;{\sigma}'(z^L)" title="\LARGE \delta^L =\triangledown_a C \odot {\sigma}'(z^L)" /><br>
### 2. 使用下一层误差表示当前层误差
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\delta^l&space;=((\omega&space;^{l&plus;1})^T)\delta&space;^{l&plus;1})\odot&space;\sigma'(z^l)" title="\LARGE \delta^l =((\omega ^{l+1})^T)\delta ^{l+1})\odot \sigma'(z^l)" /><br>
### 3. 代价函数关于偏置改变率：
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\frac{\partial&space;C}{\partial&space;b^l_j}=\delta&space;^l_j" title="\LARGE \frac{\partial C}{\partial b^l_j}=\delta ^l_j" /><br>
### 4. 代价函数关于权重改变率：
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\frac{\partial&space;C}{\partial&space;w&space;^l_{jk}}=a^{l-1}_k&space;\delta&space;^l_j" title="\LARGE \frac{\partial C}{\partial \omega ^l_{jk}}=a^{l-1}_k \delta ^l_j" /><br>
##
### 5. 交叉熵代价函数：<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;C&space;=&space;-\frac{1}{n}\sum_{x}[y\,&space;lna&plus;(1-y)\,ln(1-a)]" title="\large C = -\frac{1}{n}\sum_{x}[y\, lna+(1-y)\,ln(1-a)]" /><br>
### 6. 柔性最大值：<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{200}&space;\bg_white&space;\large&space;a^L_j=\frac{e^{z^L_j}}{\sum_{k}e^{z^L_j}}" title="\large a^L_j=\frac{e^{z^L_j}}{\sum_{k}e^{z^L_j}}" /><br>
### 7. 规范化代价函数：<br>
#### L2规范化（权重衰减）<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;C=C_0&space;&plus;&space;\frac{\lambda&space;}{2n}\sum_{\omega&space;}\omega^2" title="\large C=C_0 + \frac{\lambda }{2n}\sum_{w}w^2" /><br>
##### 此时权重学习规则变为(偏置不变)：<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;\bg_white&space;\large&space;\omega\rightarrow&space;\left&space;(&space;1-&space;\frac{\eta\lambda}{n}&space;\right&space;)\omega&space;-&space;\eta\frac{\partial&space;C}{\partial&space;\omega}" title="\large \omega\rightarrow \left ( 1- \frac{\eta\lambda}{n} \right )\omega - \eta\frac{\partial C}{\partial \omega}" /><br>
#### L1规范化<br>
<img src="https://latex.codecogs.com/png.latex?\dpi{150}&space;\bg_white&space;\large&space;C=C_0&plus;\frac{\lambda}{n}\sum_{w}|w|" title="\large C=C_0+\frac{\lambda}{n}\sum_{w}|w|" /><br>
