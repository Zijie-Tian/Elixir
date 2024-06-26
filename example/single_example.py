'''
Author: Zijie Tian tzj21@mails.tsinghua.edu.cn
Date: 2024-06-25 12:02:29
LastEditors: Zijie Tian tzj21@mails.tsinghua.edu.cn
LastEditTime: 2024-06-25 12:02:39
FilePath: /Elixir/Elixir/example/single_example.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from elixir.search import minimum_waste_search
from elixir.wrapper import ElixirModule, ElixirOptimizer
from transformers import BertForSequenceClassification
import torch


world_size = 8
world_group = torch.distributed.new_group(list(range(world_size)))

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

sr = minimum_waste_search(model, world_size)
model = ElixirModule(model, sr, world_group)
optimizer = ElixirOptimizer(model, optimizer)