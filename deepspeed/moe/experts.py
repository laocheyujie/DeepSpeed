# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
from typing import List, Optional

import torch
from torch import nn


class Experts(nn.Module):

    # NOTE: 定义一个 MoE 层上所有的 expert
    def __init__(self, expert: nn.Module, num_local_experts: int = 1, expert_group_name: Optional[str] = None) -> None:
        super(Experts, self).__init__()

        # NOTE: 每块 gpu 上共 num_local_experts 个 expert
        self.deepspeed_experts = nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for param in expert.parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # NOTE: 将 input 做切分后，将分块分别喂给该 gpu 上维护的若干个 expert
        # inputs 尺寸: (G, e, C, M)
        #           E = 该层 expert 总数
        #           G = ep_world_size
        #           e = num_local_experts，满足 E = G * e
        #           C = expert capacity
        #           M = token embedding
        # chunk_input: 沿着 e 维度切分 inputs，方便各块 input 喂给该 gpu 上对应的各个 expert
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs: List[torch.Tensor] = []

        for chunk, expert in zip(chunks, self.deepspeed_experts):
            # out 尺寸: (G, C, M)
            out = expert(chunk)
            if isinstance(out, tuple):
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]
        
        # NOTE: concat 后最终 out 尺寸: (G, e, C, M)
        return torch.cat(expert_outputs, dim=1)
