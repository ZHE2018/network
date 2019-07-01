<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
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
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\LARGE&space;\frac{\partial&space;C}{\partial&space;\omega&space;^l_{jk}}=a^{l-1}_k&space;\delta&space;^l_j" title="\LARGE \frac{\partial C}{\partial \omega ^l_{jk}}=a^{l-1}_k \delta ^l_j" /><br>
##
### 证明：
#### 1：
输出层误差定义：<br>
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\large&space;\delta&space;^L_j=\frac{\partial&space;C}{\partial&space;z^L_j}" title="\large \delta ^L_j=\frac{\partial C}{\partial z^L_j}" /><br>
<img src="https://latex.codecogs.com/png.latex?\bg_white&space;\large&space;\delta&space;^L_j=\frac{\partial&space;C}{\partial&space;z^L_j}=\sum_k&space;\frac{\partial&space;C}{\partial&space;a^L_k}\frac{\partial&space;a^L_k}{\partial&space;z^L_j}&space;=\frac{\partial&space;C}{\partial&space;a^L_j}\frac{\partial&space;a^L_j}{\partial&space;z^L_j}=\frac{\partial&space;C}{\partial&space;a^L_j}\sigma&space;'(z^L_j)" title="\large \delta ^L_j=\frac{\partial C}{\partial z^L_j}=\sum_k \frac{\partial C}{\partial a^L_k}\frac{\partial a^L_k}{\partial z^L_j} =\frac{\partial C}{\partial a^L_j}\frac{\partial a^L_j}{\partial z^L_j}=\frac{\partial C}{\partial a^L_j}\sigma '(z^L_j)" />
