## Chinese Intent Match 2019-3

#### 1.preprocess

clean() 去除停用词，替换同音、同义词，prepare() 打乱后划分训练、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.represent

embed() 建立词索引到词向量的映射，align() 将序列填充为相同长度

#### 4.build

train 80% / dev 20% 划分，通过 dnn 的 trm 构建匹配模型

#### 6.match

predict() 实时交互，输入单句、经过清洗后预测，输出所有类别的概率
