## Chinese Intent Match 2019-3

#### 1.preprocess

clean() 去除停用词，替换同音、同义词，prepare() 打乱后划分训练、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

add_flag() 分别添加 bos、sep，merge() 连接两句后填充为相同长度

#### 4.build

通过 dnn 的 trm 构建匹配模型、bos 代表整句特征，对编码器词特征 x 多头

线性映射得到 q、k、v，使用点积注意力得到语境向量 c、再线性映射进行降维

#### 6.match

predict() 实时交互，输入单句、经过清洗后预测，输出所有类别的概率
