假设输入向量为 $ \mathbf{x} \in \mathbb{R}^{input\_{\dim}} $，则该网络可以被表示为一系列连续的函数映射：

1. **线性层**：$ \mathbf{h}_i = \mathbf{W} _{i} \mathbf{x} + \mathbf{b} _{i} $
   其中，$ \mathbf{W} _{i} \in \mathbb{R}^{output _{\dim} \times input_{\dim}} $ 是权重矩阵，$ \mathbf{b} _{i} \in \mathbb{R}^{output _{\dim}} $ 是偏置向量。
2. **Leaky ReLU 激活**：$ \mathbf{h}_{i} = \text{LeakyReLU}(\mathbf{h}_{i}) $
   Leaky ReLU 定义为 $ f(x) = \max(ax, x) $，其中$a=0.01$对于所有元素应用此操作。
3. **批量归一化**：$ \mathbf{h}_{i} = \text{BatchNorm1d}(\mathbf{h}_{i}) $
   批量归一化通过对 mini-batch 中的数据进行标准化来稳定中间层的分布。
   规定$m=\text{batch}$,$\gamma$与$\beta$为可学习参数。
   批量归一化公式如下：
$$ \mu _{\text{batch}} = \frac{1}{m} \sum _{i=1}^{m} x _i $$
$$ \sigma _{\text{batch}}^2 = \frac{1}{m} \sum _{i=1}^{m} (x_i - \mu _{\text{batch}})^2 $$
$$ \hat{x}_i = \frac{x_i - \mu_{\text{batch}}}{\sqrt{{\sigma}_{\text{batch}}^2 + \epsilon}} $$
$$ y _{i} = \gamma \hat{x} _{i} + \beta $$

$ y = \mathbf{W}_{3} \left( \text{BatchNorm1d}\left( \text{LeakyReLU}\left( \mathbf{W}_{2} \left( \text{BatchNorm1d}\left( \text{LeakyReLU}\left( \mathbf{W}_{1} \mathbf{x} + \mathbf{b}_{1} \right) \right) \right) + \mathbf{b}_{2} \right) \right) \right) + \mathbf{b}_{3} $
