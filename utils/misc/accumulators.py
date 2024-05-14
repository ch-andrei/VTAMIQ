# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from copy import deepcopy


class Mean:
    """
    Running average of the values that are 'add'ed
    """
    def __init__(self, update_weight=1):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        _value = value.detach() if isinstance(value, torch.Tensor) else value
        if self.average is None:
            self.average = 0 + _value
        else:
            delta = _value - self.average
            self.average += delta * self.update_weight * weight / (self.counter + self.update_weight - 1)

    def value(self):
        """Access the current running average"""
        return self.average

    def reset(self):
        self.__init__(self.update_weight)


class Max:
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    def __init__(self):
        self.max = None

    def add(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            if isinstance(value, torch.Tensor):
                value = value.detach()
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current max"""
        return self.max

    def reset(self):
        self.__init__()