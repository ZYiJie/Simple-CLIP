# %%
#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import random
import wandb
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import transformers
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, BertModel, AutoConfig
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
transformers.logging.set_verbosity_error()

# %% [markdown]
# ### 参数

# %%
class CFG:
    train_file = './data/IQR_train.tsv'    
    valid_file = './data/IQR_dev.tsv'   
    img_key = 'filepath'
    caption_key = 'title'
    max_text_len=34
    image_size=224
    text_ptm='roberta'
    img_ptm='resnet50'
    output_dir = f'{text_ptm}-{img_ptm}-saved'
    # 训练参数
    dim = 256
    device='cuda:0'
    epochs=10
    learning_rate = 1.5e-4 # 0.5 for large
    batch_size=256
    eval_epoch = 1
    apex = True
    seed=42 
    # scheduler参数
    scheduler='cosine'                   # ['linear', 'cosine'] # lr scheduler 类型
    last_epoch=-1                        # 从第 last_epoch +1 个epoch开始训练
    batch_scheduler=True                 # 是否每个step结束后更新 lr scheduler
    weight_decay=0.01
    num_warmup_steps = 0
    num_cycles=0.5                    # 如果使用 cosine lr scheduler， 该参数决定学习率曲线的形状，0.5代表半个cosine曲线
    
    # log参数
    log_step = 5
    wandb = True
    key_metrics = 'image_to_text_R@10'

ptm = {
    "roberta": '/home/zyj/PTMs/chinese-roberta-wwm-ext/',
    "roberta-large": '/home/zyj/PTMs/chinese-roberta-wwm-ext-large/',
    "resnet50": '/home/zyj/PTMs/resnet-50/',
    "vit": '/home/zyj/PTMs/vit-base-patch16-224/',
    "vit-large": '/home/zyj/PTMs/vit-large-patch16-224/',
}


# %%
#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# %% [markdown]
# ### 数据预处理

# %%
class TrainDataset(Dataset):
    def __init__(self, input_file):
        self.tokenizer = AutoTokenizer.from_pretrained(ptm[CFG.text_ptm])
        data_df = pd.read_csv(input_file, sep='\t')
        self.img_paths = data_df[CFG.img_key].values
        self.texts = data_df[CFG.caption_key].values

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ptm[CFG.img_ptm])
        
        print(f'load data from {input_file} len={len(self.texts)}')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        img_path = self.img_paths[item]
        text_tensor = self.tokenizer(text, 
                                max_length=CFG.max_text_len, 
                                truncation=True, 
                                return_tensors='pt', 
                                padding="max_length",)
        img_tensor = self.feature_extractor(Image.open(img_path).convert("RGB"), return_tensors="pt")
        for k,v in text_tensor.items():
            text_tensor[k] = v.squeeze()
        for k,v in img_tensor.items():
            img_tensor[k] = v.squeeze()
        # print(item, text, img_path, img_tensor.shape)
        return {'text':text_tensor, 'img':img_tensor}

# %% [markdown]
# ### 模型定义

# %% [markdown]
# #### TextEncoder

# %%
class TextEncoder(nn.Module):
    def __init__(self, ptm_name, device, freeze=False):
        super().__init__()
        self.model = BertModel.from_pretrained(ptm_name)
        self.model.to(device)
        self.feat_dim = AutoConfig.from_pretrained(ptm_name).hidden_size

        self.freeze = freeze
        if freeze:
            for name ,param in self.model.named_parameters():
                param.requires_grad = False
        
    def forward(self, inputs):
        if self.freeze:
            self.model.eval()
        last_hidden_state = self.model(**inputs).last_hidden_state # [batch_size, seq_len, hidden_size]
        feature = torch.mean(last_hidden_state, axis=1) # [batch_size, hidden_size]

        feature = F.normalize(feature, dim=-1)
        return feature

# %% [markdown]
# #### ImageEncoder

# %%
class ImageEncoder(nn.Module):
    def __init__(self, ptm_name, device):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(ptm_name)
        self.feat_dim = 1000

    def forward(self, inputs):
        feature = self.model(**inputs).logits # [batch_size, 1000]
        feature = F.normalize(feature, dim=-1)
        return feature


# %% [markdown]
# #### SimpleCLIP

# %%
class SimpleCLIP(nn.Module):
    def __init__(self, dim, text_ptm_name, img_ptm_name, device, freeze=False):
        super().__init__()
        self.device = device
        self.textencoder = TextEncoder(text_ptm_name, device, freeze=freeze)
        self.imgencoder = ImageEncoder(img_ptm_name, device)

        self.text_projection = nn.Parameter(torch.empty(self.textencoder.feat_dim, dim)).to(device)
        self.img_projection = nn.Parameter(torch.empty(self.imgencoder.feat_dim, dim)).to(device)
        self.logit_scale = nn.Parameter(torch.ones([])).to(device)
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.normal_(self.text_projection, std=0.02)
        nn.init.normal_(self.img_projection, std=0.02)

    def loss(self, text_feat, img_feat, logit_scale):
        labels = torch.arange(text_feat.shape[0], device=self.device, dtype=torch.long)

        logits_per_image = logit_scale * img_feat @ text_feat.T   # [batch_size, batch_size]
        logits_per_text = logit_scale * text_feat @ img_feat.T   # [batch_size, batch_size]
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss

    def forward(self, text_inputs, img_inputs, outputLoss=False):
        text_feat = self.textencoder(text_inputs) @ self.text_projection # [batch_size, dim]
        img_feat = self.imgencoder(img_inputs) @ self.img_projection # [batch_size, dim]
        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(text_feat, img_feat, logit_scale)
            return loss, text_feat, img_feat, logit_scale
        else:
            return text_feat, img_feat, logit_scale


# %% [markdown]
# ### 主程序

# %% [markdown]
# #### evaluate

# %%
def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def evaluate(model, valid_dataloader, device):
    model.eval()
    all_text_feat = []
    all_img_feat = []
    tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Dev]")
    total_loss = 0
    for step, batch in tk0:
        for k,v in batch['img'].items():
            batch['img'][k] = v.to(device)
        for k,v in batch['text'].items():
            batch['text'][k] = v.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, text_feat, img_feat, logit_scale = model(batch['text'], 
                                                                batch['img'], 
                                                                outputLoss=True)
        total_loss += loss.item()
        all_text_feat.append(text_feat)
        all_img_feat.append(img_feat)
        
    metrics = get_metrics(image_features=torch.cat(all_img_feat),
                          text_features=torch.cat(all_text_feat),
                          logit_scale=logit_scale)
    metrics['eval_loss'] = total_loss / len(valid_dataloader)
    return metrics


# %% [markdown]
# #### train loop

# %%
def train_eval(model, train_dataloader, valid_dataloader, save_path):
    assert CFG.device.startswith('cuda') or CFG.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(CFG.device)
    best_score = 0
    total_step = 0
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    # 过滤掉冻结的权重
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # 设置权重decay
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": CFG.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    
    num_train_steps = int(len(train_dataloader) * CFG.epochs)
    if CFG.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=CFG.num_warmup_steps, 
                    num_training_steps=num_train_steps, 
                    num_cycles=CFG.num_cycles, 
#                     last_epoch = ((CFG.last_epoch+1)/CFG.epochs)*num_train_steps
                )
    else:
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=CFG.num_warmup_steps, num_training_steps=num_train_steps
            )
    
    for cur_epc in range(int(CFG.epochs)):
        
        if cur_epc % CFG.eval_epoch == 0:
            metrics = evaluate(model, valid_dataloader, device)
            print(f"eval metrics = ")
            pprint(metrics)
            if CFG.wandb:
                wandb.log(metrics, step=total_step)
            if cur_epc > 0 and metrics[CFG.key_metrics] >= best_score:
                best_score = metrics[CFG.key_metrics]
                # model_save_path = os.path.join(save_path,f'epoch{cur_epc}.pt') # 保留所有checkpoint
                model_save_path = os.path.join(save_path,f'best_checkpoint.pt') # 保留最优checkpoint
                torch.save(model.state_dict(), model_save_path)
                print(f'save at {model_save_path}')
        
        training_loss = 0
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc="Epoch: {}".format(cur_epc))
        for step, batch in tk0:
            total_step += 1
            for k,v in batch['img'].items():
                batch['img'][k] = v.to(device)
            for k,v in batch['text'].items():
                batch['text'][k] = v.to(device)
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                loss, _, _, _ = model(batch['text'], batch['img'], outputLoss=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if CFG.batch_scheduler:
                scheduler.step()
            training_loss += loss.item()
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
            if CFG.wandb and (step + 1) % CFG.log_step == 0:
                wandb.log({'train_loss':loss, 'lr':optimizer.param_groups[0]["lr"], 'epoch': cur_epc},
                          step=total_step)
        
    torch.cuda.empty_cache()          

# %% [markdown]
# #### 训练过程

# %%
if __name__ == '__main__':
    seed_everything(seed=42)
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)
    with open(os.path.join(CFG.output_dir, 'config.txt'), 'w') as f:
        for k,v in CFG.__dict__.items():
            f.write(f'{k}: {v}\n')

    # 加载数据
    train_dataset = TrainDataset(CFG.train_file)
    valid_dataset = TrainDataset(CFG.valid_file)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, num_workers=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.batch_size, num_workers=5)
    # 加载模型
    device = torch.device(CFG.device)
    clipModel = SimpleCLIP(CFG.dim, ptm[CFG.text_ptm], ptm[CFG.img_ptm], device, freeze=False)
    
    if CFG.wandb:
        wandb.init(project='SimpleCLIP', name=f'{CFG.text_ptm}-{CFG.img_ptm}-batch{CFG.batch_size}-dim{CFG.dim}')
    
    # 训练
    train_eval(clipModel, train_dataloader, valid_dataloader, CFG.output_dir)


