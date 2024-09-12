import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./covid_train.csv")
# print(df.describe)
df_test = pd.read_csv("./covid_test.csv")
# print(df_test.describe)

scaler = MinMaxScaler()
columns_to_normalize = df.columns[1:]
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df.to_csv("./covid_train_norm.csv")
columns = df.columns

# 遍历每一对列并计算皮尔逊相关系数
# for i in range(len(columns)):
#     for j in range(i + 1, len(columns)):
#         col1 = columns[i]
#         col2 = columns[j]
#         corr, _ = pearsonr(df[col1], df[col2])
#         print(f"皮尔逊相关系数 ({col1}, {col2}): {corr:.4f}")

column_iwannatest = "tested_positive"
corr_values = []
# 遍历每一列并计算相关系数
for i in range(len(columns)):
    col1 = columns[i]
    corr, _ = pearsonr(df[column_iwannatest], df[col1])
    if abs(corr) >= 0.75:
        corr_values.append((col1, corr, i))
        print(
            f"皮尔逊相关系数 ({column_iwannatest}, {col1}): {corr:.4f} 以及这个id为:{i}"
        )
        # print(f"{i},")

# 绘制相关系数的条形图
if corr_values:
    labels, values, ids = zip(*corr_values)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="blue")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x位置：条形中间
            bar.get_height() + 0.01,  # y位置：条形顶部之上一点
            f"{value:.2f}",  # 显示的文本
            ha="center",  # 水平对齐方式
            va="bottom",
        )  # 垂直对齐方式

    plt.xlabel("Columns")
    plt.ylabel("Pearson Correlation")
    plt.title(f"{column_iwannatest} vs. Different Columns")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    # 保存图表
    plt.savefig("./heatmap.jpg")

else:
    print("没有找到相关系数绝对值大于等于 0.75 的变量。")
# corr_matrix = df.corr(method="pearson")
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
# plt.title("皮尔逊相关系数热力图")
# plt.show()
# plt.savefig("./heatmap.jpg")
