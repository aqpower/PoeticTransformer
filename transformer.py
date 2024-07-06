#!/usr/bin/env python
# coding: utf-8

# In[124]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import json
import random
import matplotlib.pyplot as plt
from torchinfo import summary


# In[125]:


seed = 9
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

batch_size = 96
num_epochs = 99
context_len = 128
initial_lr = 0.001
data_path = "./data/chinese-poetry/唐诗"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("now using", device)


# In[126]:


with open("poems.json", "r", encoding="utf-8") as f:
    poems = json.load(f)

with open("vocab.json", "r", encoding="utf-8") as f:
    word_to_index = json.load(f)

index_to_word = {index: word for word, index in word_to_index.items()}

vocab_size = len(word_to_index)

print("VOCAB_SIZE:", vocab_size)
print("data_size", len(poems))


# 将句子转换为列表形式，并添加结束符
poems = [list(poem) + ["<EOP>"] for poem in poems]
index_tensors = {
    word: torch.LongTensor([word_to_index[word]]) for word in word_to_index
}


# In[127]:


def generate_sample(poem):

    inputs = [index_tensors[poem[i - 1]] for i in range(1, len(poem))]
    outputs = [index_tensors[poem[i]] for i in range(1, len(poem))]

    # 将输入和输出列表合并为张量
    encoded_inputs = torch.cat(inputs)
    encoded_outputs = torch.cat(outputs)

    return encoded_inputs, encoded_outputs


class PoetryDataset(Dataset):
    def __init__(self, poems, transform=None):
        self.poems = poems
        self.transform = transform

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, index):
        poem = self.poems[index]
        input_data, output_data = generate_sample(poem)
        if self.transform:
            input_data = self.transform(input_data)
        return input_data, output_data


def custom_collate_fn(batch):
    inputs, outputs = zip(*batch)
    # 统一长度以进行批处理
    padded_inputs = nn.utils.rnn.pad_sequence(
        inputs, batch_first=True, padding_value=word_to_index["<START>"]
    )
    padded_outputs = nn.utils.rnn.pad_sequence(
        outputs, batch_first=True, padding_value=word_to_index["<START>"]
    )
    return padded_inputs, padded_outputs


dataset = PoetryDataset(poems)
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
)


# ![](../multi-head.png)
# 
# ![](../self-attention.png)
# 

# In[128]:


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads=4, mask=False):
        super(SelfAttention, self).__init__()

        assert embed_size % num_heads == 0, "Embedding size 必须是 heads 的整数倍"
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # 计算所有 heads 的 query, key 和 value
        self.query_projection = nn.Linear(embed_size, embed_size, bias=False)
        self.key_projection = nn.Linear(embed_size, embed_size, bias=False)
        self.value_projection = nn.Linear(embed_size, embed_size, bias=False)

        # 在 multi-head self-attention 操作后应用
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.mask = mask

    def forward(self, x):
        batch_size, seq_length, embed_size = x.size()

        # 将输入 x 分别通过线性层投影到 query, key 和 value 向量
        # 只用三次 k×k 矩阵乘法就能实现 multi-head 功能
        # 唯一需要的额外操作是将生成的 output vector 重新按块排序
        queries = self.query_projection(x).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        keys = self.key_projection(x).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        values = self.value_projection(x).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )

        # print(queries.size())
        # torch.Size([1, 5, 4, 64])

        # 将 tensor 重新排列，以适应 multi-head attention
        queries = (
            queries.transpose(1, 2)
            .reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        )
        keys = (
            keys.transpose(1, 2)
            .reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        )
        values = (
            values.transpose(1, 2)
            .reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        )

        # print(queries.size())
        # torch.Size([4, 5, 64])

        # 计算 Scaled dot-product attention 点积相关度矩阵
        dot_product = torch.bmm(queries, keys.transpose(1, 2))
        # print(dot_product.size())
        # torch.Size([4, 5, 5])

        # softmax 函数对非常大的输入值敏感。
        # 这些 input 会梯度消失，学习变慢甚至完全停止。
        # 由于点积的平均值随着嵌入维度 k 的增加而增大
        # 因此点积送到 softmax 之前进行缩放有助于缓解这个问题。
        scaled_dot_product = dot_product / (self.embed_size**0.5)

        # 如果启用了 mask，则对未来的 token 进行屏蔽
        if self.mask:
            # torch.triu(..., diagonal=1)：保留上三角部分
            # 指定 diagonal=1 表示从第一个对角线开始（即排除主对角线），其余部分设为零
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask = mask.to(device)
            scaled_dot_product.masked_fill_(mask, float("-inf"))

        attention = F.softmax(scaled_dot_product, dim=2)

        # 将 self-attention 应用于 values
        # print(torch.bmm(attention, values).size())
        # torch.Size([4, 5, 64])
        out = torch.bmm(attention, values).reshape(
            batch_size, self.num_heads, seq_length, self.head_dim
        )
        # print(out.size())
        # torch.Size([1, 4, 5, 64])
        out = (
            out.transpose(1, 2)
            .reshape(batch_size, seq_length, self.embed_size)
        )
        # print(out.size())
        # torch.Size([1, 5, 256])
        return self.fc_out(out)


# ![](../transformer-block.png)
# 

# In[129]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, mask=False):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, num_heads=num_heads, mask=mask)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Self-attention 和残差连接
        attended = self.attention(x)
        x = self.norm1(attended + x)

        # 前馈神经网络和残差连接
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


# ![](../transformer-architecture.png)

# In[130]:


class Transformer(nn.Module):
    def __init__(
        self,
        embed_size,
        num_heads,
        num_layers,
        context_len,
        num_tokens,
        num_classes,
        mask=False,
    ):
        super(Transformer, self).__init__()

        self.token_emb = nn.Embedding(num_tokens, embed_size)
        self.pos_emb = nn.Embedding(context_len, embed_size)

        self.layers = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, mask) for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_size, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        batch_size, seq_length = x.size()

        # 生成 token 嵌入
        tokens = self.token_emb(x)
        # 生成位置嵌入
        positions = torch.arange(seq_length).to(x.device)
        # print(positions.size())
        # torch.Size([5])
        # print(self.pos_emb(positions).size())
        # torch.Size([5, 256])

        positions = self.pos_emb(positions).expand(batch_size, seq_length, -1)
        # print(positions.size())
        # torch.Size([1, 5, 256])
        

        # 将 token 嵌入和位置嵌入相加
        x = tokens + positions

        # 通过所有 Transformer 层
        x = self.layers(x)

        # 最后映射到类概率
        x = self.fc_out(x)

        return x


# In[131]:


def train(model, data_loader, num_epochs, device, optimizer, criterion, scheduler, model_name):
    log_dict = {
        "train_loss_per_epoch": [],
        "train_perplexity_per_epoch": [],
        "model_name": model_name,
    }
    start_time = time.time()
    model = model.to(device)
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d} | Current Learning Rate: {current_lr:.6f}"
        )
        total_loss = 0
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

            if not batch_idx % 5:
                print(
                    f"Epoch: {epoch + 1:03d}/{num_epochs:03d} | Batch {batch_idx:04d}/{len(data_loader):04d} | Loss: {loss:.6f}"
                )

        avg_loss = total_loss / len(data_loader.dataset)
        scheduler.step(avg_loss)
        perplexity = torch.exp(torch.tensor(avg_loss))
        log_dict["train_loss_per_epoch"].append(avg_loss)
        log_dict["train_perplexity_per_epoch"].append(perplexity)

        print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} min")

    torch.save(model.state_dict(), f"{model_name}_model_state_dict.pth")
    print(f"Total Training Time: {(time.time() - start_time)/ 60:.2f} min")
    return log_dict


# In[132]:


def plot_training_stats(log_dict):
    model_name = log_dict["model_name"]

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(log_dict["train_loss_per_epoch"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}_Training Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(log_dict["train_perplexity_per_epoch"], label="Training Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.grid(True)
    # plt.yscale("log")
    plt.title(f"{model_name}_Training Perplexity")
    plt.savefig(f"{model_name}_training_stats.svg")
    plt.show()


mask = True
model = Transformer(
    embed_size=256,
    num_heads=8,
    num_layers=8,
    context_len=context_len,
    num_tokens=vocab_size,
    num_classes=vocab_size,
    mask=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=9, verbose=True
)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_index["<START>"])
# log_dict = train(
#     model, data_loader, num_epochs, device, optimizer, criterion, scheduler, "transformer"
# )
# plot_training_stats(log_dict)
model.load_state_dict(torch.load("model_state_dict.pth"))
model.to(device)

inputs = torch.tensor([[1]]).to(device) 
summary(model, input_data=inputs, depth=4)


# ![](./training_stats.svg)

# In[152]:


def generate_text(start_word, top_k=1, temperature=0.7, log=False):
    generated_text = ""
    index_tensors_list = []
    for word in start_word:
        index_tensors_list.append(index_tensors[word].unsqueeze(0))
        generated_text += word
    model.eval()
    with torch.no_grad():
        for _ in range(context_len - len(generated_text)):
            input_tensor = torch.tensor(index_tensors_list).unsqueeze(0).to(device)
            # print(input_tensor.size())
            # torch.Size([1, 5])
            output = model(input_tensor.to(device))

            last_word = output[:, -1, :]
            last_word = last_word.view(-1)

            # 调整温度
            # softmax 函数倾向于增强输入向量中最大值的影响
            scaled_logits = last_word / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)

            probabilities, top_indices = probabilities.data.topk(top_k)
            top_words = [index_to_word[index.item()] for index in top_indices]
            probabilities = probabilities / torch.sum(probabilities)

            probabilities_np = probabilities.cpu().numpy()
            indices_np = top_indices.cpu().numpy()
            if log:
                for word, prob in zip(top_words, probabilities_np):
                    print(f"{word}: {prob:.4f}")

            selected_index = np.random.choice(indices_np, p=probabilities_np)

            next_word = index_to_word[selected_index]
            if next_word == "<EOP>":
                break
            if log:
                print(generated_text)
            index_tensors_list.append(index_tensors[next_word])
            generated_text += next_word

    return generated_text.strip()


print(generate_text("高楼入青天", top_k=1))
print(generate_text("长安一片月", top_k=3))
print(generate_text("长安一片月", top_k=3, temperature=1.2))
for i in range(10):
    print(generate_text("月", top_k=20, temperature=1.1))
for i in range(10):
    print(generate_text("海", top_k=3))
print(generate_text("风", top_k=3, log=True))


# In[142]:


def generate_acrostic(start_word, top_k=1, temperature=0.7, log=False):
    generated_text = ""
    words = []
    for word in start_word:
        words += [word]
    index_tensors_list = []
    index_tensors_list.append(index_tensors[words[0]].unsqueeze(0))
    generated_text += words[0]
    model.eval()
    with torch.no_grad():

        ind = 1
        for _ in range(context_len - len(generated_text)):
            input = torch.tensor(index_tensors_list).unsqueeze(0).to(device)
            output = model(input)
            last_word = output[:, -1, :]
            last_word = last_word.view(-1)

            # 调整温度
            # softmax 函数倾向于增强输入向量中最大值的影响
            scaled_logits = last_word / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)

            probabilities, top_indices = probabilities.data.topk(top_k)
            top_words = [index_to_word[index.item()] for index in top_indices]
            probabilities = probabilities / torch.sum(probabilities)

            probabilities_np = probabilities.cpu().detach().numpy()
            indices_np = top_indices.cpu().detach().numpy()
            if log:
                for word, prob in zip(top_words, probabilities_np):
                    print(f"{word}: {prob:.4f}")

            selected_index = np.random.choice(indices_np, p=probabilities_np)

            next_word = index_to_word[selected_index]
            if next_word == "<EOP>":
                break
            generated_text += next_word

            # 如果遇到句号感叹号等，把藏头的词作为下一个句的输入
            if next_word in ["。"]:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                if ind == len(start_word):
                    break
                # 把藏头的词作为输入，预测下一个词
                index_tensors_list.append(index_tensors[next_word])
                index_tensors_list.append(index_tensors[words[ind]])
                generated_text = generated_text + '\n' + words[ind]
                ind += 1
            else:
                index_tensors_list.append(index_tensors[next_word])

            if log:
                print(generated_text)

    return generated_text.strip()

print(generate_acrostic("不如见一面", top_k=20, temperature=1.2))
print()
print(generate_acrostic("深度学习", top_k=20, temperature=1.2))


# In[135]:


def train_all_resnet_models():
    resnet_models = {
        "Transformer_2_heads": (
            Transformer(
                embed_size=128,
                num_heads=2,
                num_layers=8,
                context_len=context_len,
                num_tokens=vocab_size,
                num_classes=vocab_size,
                mask=True,
            ).to(device),
            50,
            0.001,
        ),
        "Transformer_4_heads": (
            Transformer(
                embed_size=128,
                num_heads=4,
                num_layers=8,
                context_len=context_len,
                num_tokens=vocab_size,
                num_classes=vocab_size,
                mask=True,
            ).to(device),
            50,
            0.001,
        ),
        "Transformer_8_heads": (
            Transformer(
                embed_size=128,
                num_heads=8,
                num_layers=8,
                context_len=context_len,
                num_tokens=vocab_size,
                num_classes=vocab_size,
                mask=True,
            ).to(device),
            50,
            0.001,
        ),
    }

    log_dicts = []

    for model_name, (model, num_epochs, initial_lr) in resnet_models.items():
        print(
            f"Training {model_name} for {num_epochs} epochs with initial learning rate {initial_lr}..."
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=initial_lr, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=9, verbose=True
        )
        criterion = nn.CrossEntropyLoss(ignore_index=word_to_index["<START>"])
        log_dict = train(
            model, data_loader, num_epochs, device, optimizer, criterion, scheduler, model_name
        )

        log_dicts.append(log_dict)
        model.load_state_dict(torch.load(f"{model_name}_model_state_dict.pth"))
        plot_training_stats(log_dict)

    return log_dicts


def plot_compare(log_dicts):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    color_list = ["#6495ED", "#4EEE94", "#EEC900", "#FF6347", "#BA55D3", "#00808C"]
    ind_color = 0
    for log_dict in log_dicts:
        train_loss_per_epoch = log_dict["train_loss_per_epoch"]
        model_name = log_dict["model_name"]
        train_perplexity_per_epoch = log_dict["train_perplexity_per_epoch"]

        axs[0].plot(
            np.arange(1, len(train_perplexity_per_epoch) + 1),
            train_perplexity_per_epoch,
            ".--",
            color=color_list[ind_color],
            label=f"{model_name}",
        )

        axs[1].plot(
            np.arange(1, len(train_loss_per_epoch) + 1),
            train_loss_per_epoch,
            ".--",
            color=color_list[ind_color],
            label=f"{model_name}",
        )
        ind_color += 1

    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("train_perplexity_per_epoch")
    axs[0].set_title(f"Training Perplexity")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("train_loss_per_epoch")
    axs[1].set_title(f"Training Loss")
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    axs[0].grid(True)
    axs[1].grid(True)
    fig.savefig("training_performance.svg", format="svg")
    fig.show()

# log_dicts = train_all_resnet_models()
# plot_compare(log_dicts)

