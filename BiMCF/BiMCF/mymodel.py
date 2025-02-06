import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from PIL import Image
from transformers import BertTokenizer, ViTFeatureExtractor

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        output = self.dense(output)

        return output, attention_weights

class MultiModalTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(MultiModalTransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        attn_output, _ = self.mha(q, k, v, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(q + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class MultiModalTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(MultiModalTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = nn.ModuleList([MultiModalTransformerLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, mask=None):
        q = self.dropout(q)

        for i in range(self.num_layers):
            q = self.enc_layers[i](q, k, v, mask)

        return q

class MTCGateFowrad(nn.Module):
    def __init__(self, d_model):
        super(MTCGateFowrad, self).__init__()
        self.d_model = d_model
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
    

    def forward(self, vision_feature, text_feature, vision_fusion_feature):
        max_vision_feature = torch.max(vision_feature, vision_fusion_feature)
        bs, _, dim_size = vision_feature.shape
        #  b, s, t
        Mf = torch.einsum('bsd,btd->bst', self.linear_1(max_vision_feature), self.linear_2(text_feature))
        # Mf = torch.matmul(self.linear_1(max_vision_feature), self.linear_2(text_feature).T) # b*M*N
        forward_score = F.softmax(Mf.sum(dim=-1), dim=-1)
        vision_forward_featrue = torch.einsum('bs,bsd->bsd', forward_score, max_vision_feature)
        return vision_forward_featrue

class MTCGateBackward(nn.Module):
    def __init__(self, d_model):
        super(MTCGateBackward, self).__init__()
        self.d_model = d_model
        self.linear_1 = nn.Linear(d_model, d_model)
        self.linear_2 = nn.Linear(d_model, d_model)
    
    def forward(self, vision_feature, text_feature, vision_fusion_feature):
        # dfferent from forward, using elem-add instead of elem-multiple
        sum_vision_feature = vision_feature + vision_fusion_feature
        Mb = torch.einsum('bsd,btd->bst', self.linear_1(sum_vision_feature), self.linear_2(text_feature))
        backward_score = F.softmax(Mb.sum(dim=-1), dim=-1)

        vision_backward_featrue = torch.einsum('bs,bsd->bsd', backward_score, sum_vision_feature)
        return vision_backward_featrue

class MTCForwrd(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(MTCForwrd, self).__init__()
        self.gate = MTCGateFowrad(d_model)
        self.multi_transformer = MultiModalTransformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,dropout_rate=dropout_rate)
    
    def forward(self, vision_feature, text_feature, last_fusion_v, last_fusion_t):
        # step=1: get the gate fusion feature
        fusion_v = self.gate(vision_feature, last_fusion_t, last_fusion_v)
        multi_feature = self.multi_transformer(text_feature, fusion_v, fusion_v)
        return fusion_v, multi_feature

class MTCbackward(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(MTCbackward, self).__init__()
        self.gate = MTCGateBackward(d_model)
        self.multi_transformer = MultiModalTransformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads,
        dff=dff,dropout_rate=dropout_rate)
    
    def forward(self, vision_feature, text_feature, last_fusion_v, last_fusion_t):
        # step=1: get the gate fusion feature
        fusion_v = self.gate(vision_feature, last_fusion_t, last_fusion_v)
        multi_feature = self.multi_transformer(text_feature, fusion_v, fusion_v)
        return fusion_v, multi_feature


class BiMCF(nn.Module):
    def __init__(self, d_model=768, num_layers=4, num_heads=8, dff=2048, 
                task_nums=3, 
                task1_class_nums=3, 
                task2_class_nums=4, 
                task3_class_nums=6):
        super(BiMCF, self).__init__()
        # flatten_classes: sum of the total classes
        self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.task_nums = task_nums
        self.forward_layers = nn.ModuleList([MTCForwrd(num_layers=num_layers, d_model=d_model, 
                    num_heads=num_heads, dff=dff) for _ in range(task_nums)])
        self.backward_layers = nn.ModuleList(
            [MTCbackward(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff) 
                for _ in range(task_nums)])
        task1_Linear = nn.ModuleList([
            torch.nn.Linear(2*d_model, d_model), 
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, task1_class_nums)
        ])
        task2_Linear = nn.ModuleList([
            torch.nn.Linear(2*d_model, d_model), 
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, task2_class_nums)
        ])
        task3_Linear = nn.ModuleList([
            torch.nn.Linear(2*d_model, d_model), 
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d_model, task3_class_nums)
        ])
        self.task_linear = nn.ModuleList(
            [
                task1_Linear, 
                task2_Linear,
                task3_Linear
            ]
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        
    
    def forward(self, data):
        # data with key as: vision_feuarue, text_feature, and optional labels, flatted here as task_nums label-hot encoding

        task_logits = []

        vision_outputs = self.vision_encoder(pixel_values=data['pixel_values'])
        vision_feature = vision_outputs.last_hidden_state

        text_outputs = self.bert_encoder(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        text_feature = text_outputs.last_hidden_state

        vision_forward_featrue_ = vision_feature.clone()
        vision_backward_feature_ = vision_feature.clone()
        text_forward_feature_ = text_feature.clone()
        text_backward_feature_ = text_feature.clone()
        task1_loss, task2_loss, task3_loss = 0, 0, 0
        for task_num in range(self.task_nums):

            vision_forward_featrue_, text_forward_feature_ = self.forward_layers[task_num].forward(
                vision_feature, text_feature, vision_forward_featrue_, text_forward_feature_
            )
            vision_backward_feature_, text_backward_feature_ = self.backward_layers[self.task_nums-1-task_num].forward(
                vision_feature, text_feature, vision_backward_feature_, text_backward_feature_
            )
            
            # concat forward and backward for lienar classification, eg, level-1 / level-2 / level-3 etcs.
            logit_ = torch.cat([text_forward_feature_, text_backward_feature_], dim=-1).mean(dim=1).squeeze()
            # print(logit_.shape)
            for i_, layer in enumerate(self.task_linear[task_num]):
                logit_ = layer.forward(logit_)
            task_logits.append(logit_)

        if 'task1_label' in data and 'task2_label' in data and 'task3_label' in data:
            task1_loss, task2_loss, task3_loss = self.criterion(task_logits[0], data['task1_label']), \
            self.criterion(task_logits[1], data['task2_label']), \
            self.criterion(task_logits[2], data['task3_label'])
        res = {
            'logit1': task_logits[0],
            'logit2': task_logits[1],
            'logit3': task_logits[2],
            'task1_loss': task1_loss,
            'task2_loss': task2_loss,
            'task3_loss': task3_loss
        }
        return res
