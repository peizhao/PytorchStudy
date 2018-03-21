import torch
from torch.autograd import Variable
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

word_to_ix = {"hello":0, "world":1}
embeds = nn.Embedding(2,5)
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(Variable(lookup_tensor))
print(hello_embed)