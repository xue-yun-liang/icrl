import torch

def sample_index_from_2d_array(array):
    sampled_indices = []
    for sub_array in array:
        probabilities = torch.tensor(sub_array, dtype=torch.float)
        probabilities /= torch.sum(probabilities)  # 将概率归一化为和为1
        sampled_index = torch.multinomial(probabilities, 1).item()  # 使用multinomial进行抽样
        sampled_indices.append(sampled_index)
    return sampled_indices

# 示例数组
array = [[1, 3], [1, 2, 3, 4]]

# 抽样得到索引
result = sample_index_from_2d_array(array)
print(result)
