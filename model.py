import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import AttentionLayer as AL, GlobalAttentionLayer as GoAL, StructAttentionLayer as SAL
from .dataset import get_lm_path

class Classifer(nn.Module):
    def __init__(self,hidden_size,device,attr_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.attr_num = attr_num
        self.source = nn.Linear(hidden_size,hidden_size)
    def forward(self,zs,fc,y):
        random_noise = torch.rand(zs.shape).to(self.device)
        noise = torch.sigmoid(self.source(random_noise))

        mask = (y == 1).unsqueeze(1) 
        zs = noise * mask + zs
        logits = fc(zs)  
        logits = torch.softmax(logits, dim=-1)
        logits = logits[:, 0]
        pos_mask = (y == 1)
        pos_logits = logits[pos_mask]
        pos_loss = pos_logits.mean(dim=-1) if pos_logits.numel() > 0 else torch.tensor(0.0, device=self.device)
        return pos_loss, zs

class TranHGAT(nn.Module):
    def __init__(self, attr_num, device='cpu', finetuning=True, lm='bert', lm_path=None):
        super().__init__()

        # load the model or model checkpoint
        path = get_lm_path(lm, lm_path)
        self.lm = lm
        if lm == 'bert':
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetModel
            self.bert = XLNetModel.from_pretrained(path)
        self.device = device
        self.finetuning = finetuning

        hidden_size = 768
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])
        self.conts = nn.ModuleList([
            AL(hidden_size + hidden_size, 0.2, device)
            for _ in range(attr_num)])
        self.out = SAL(hidden_size * (attr_num + 1), 0.2)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)
        self.enti = nn.Linear(hidden_size*2, hidden_size)

        self.enc_source = nn.Linear(hidden_size * attr_num, hidden_size)
        self.enc_target = nn.Linear(hidden_size * attr_num, hidden_size)
        self.attr_num = attr_num
        self.to_xs = nn.Linear(hidden_size +  hidden_size, hidden_size)
        self.classifer = Classifer(hidden_size,self.device,self.attr_num)

    def forward(self, xs, y, masks,left_zs, right_zs):
        xs = xs.to(self.device)
        y = y.to(self.device)
        left_zs, right_zs = left_zs.to(self.device), right_zs.to(self.device)
        masks = masks.to(self.device)
        xs = xs.permute(1, 0, 2)
        left_zs = left_zs.permute(1, 0, 2) 
        right_zs = right_zs.permute(1, 0, 2) 
        masks = masks.permute(0, 2, 1)

        pooled_outputs = []
        for x in left_zs:
            if self.lm == 'distilbert':
                words_emb = self.bert.embeddings(x)
            else:
                words_emb = self.bert.get_input_embeddings()(x)
            output = self.bert(inputs_embeds=words_emb)
            pooled_output = output[0][:, 0, :]
            pooled_output = self.dropout(pooled_output)
            pooled_outputs.append(pooled_output)
           
        attr_outputs_left = torch.stack(pooled_outputs).permute(1, 0, 2)

        pooled_outputs = []
        for x in right_zs:
            if self.lm == 'distilbert':
                words_emb = self.bert.embeddings(x)
            else:
                words_emb = self.bert.get_input_embeddings()(x)
            output = self.bert(inputs_embeds=words_emb)
            pooled_output = output[0][:, 0, :]
            pooled_output = self.dropout(pooled_output)
            pooled_outputs.append(pooled_output)
        attr_outputs_right = torch.stack(pooled_outputs).permute(1, 0, 2)

        attr_outputs = []
        pooled_outputs = []
        attns = []
        if self.training and self.finetuning:
            self.bert.train()
            for x, init, cont in zip(xs, self.inits, self.conts):
                attr_embeddings = init(self.bert.get_input_embeddings()(x))
                attr_outputs.append(attr_embeddings)
                attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings) 
                attns.append(attn)
            attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks 
            attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)

            for x in xs:
                if self.lm == 'distilbert':
                    words_emb = self.bert.embeddings(x)
                else:
                    words_emb = self.bert.get_input_embeddings()(x)

                for i in range(words_emb.size()[0]):
                    words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                output = self.bert(inputs_embeds=words_emb)
                pooled_output = output[0][:, 0, :]
                pooled_output = self.dropout(pooled_output)

                pooled_outputs.append(pooled_output)

            attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)            
            entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
            entity_output = self.out(attr_outputs, entity_outputs)

        else:
            self.bert.eval()
            with torch.no_grad():
                for x, init, cont in zip(xs, self.inits, self.conts):
                    attr_embeddings = init(self.bert.get_input_embeddings()(x))
                    attr_outputs.append(attr_embeddings)
                    attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)
                    attns.append(attn)

                attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks
                attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)
                for x in xs:
                    if self.lm == 'distilbert':
                        words_emb = self.bert.embeddings(x)
                    else:
                        words_emb = self.bert.get_input_embeddings()(x)

                    for i in range(words_emb.size()[0]):
                        words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                    output = self.bert(inputs_embeds=words_emb)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    pooled_outputs.append(pooled_output)

                attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
                entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
                entity_output = self.out(attr_outputs, entity_outputs)

        attr_right = attr_outputs_right.mean(dim=1)
        attr_left = attr_outputs_left.mean(dim=1)
        magnitude = torch.sigmoid(attr_right * attr_left)
        random_noise =  torch.normal(0, 1, size=(entity_output.shape[0], 5, entity_output.shape[1])).cuda() 
        mask = (y == 1).unsqueeze(1).unsqueeze(2)
        candidate_embeds = entity_output.unsqueeze(dim=1) + random_noise * magnitude.unsqueeze(dim=1) * mask
        
        final_embeds = entity_output.unsqueeze(dim=1) + random_noise * magnitude.unsqueeze(dim=1) * mask
        final_embeds = final_embeds.mean(dim=1)
        
        scores = (candidate_embeds * entity_output.unsqueeze(dim=1)).mean(dim=-1) 

        indices = torch.min(scores, dim=1)[1].detach()
        indices_expanded = indices.unsqueeze(1).unsqueeze(2)
        final_embeds = torch.gather(candidate_embeds, 1, indices_expanded.expand(-1, -1, candidate_embeds.size(2))).squeeze(dim=1)

        batch_size = entity_output.size(0)
        random_indices = torch.randint(0, batch_size, (batch_size,)).cuda()
        random_embeds = entity_output[random_indices]
        
        similarity_with_original = (final_embeds * entity_output).sum(dim=-1)
        similarity_with_random = (final_embeds * random_embeds).sum(dim=-1)

        eps = 1e-8
        diff = similarity_with_original - similarity_with_random
        sigmoid_diff = torch.sigmoid(diff - eps)
        contrastive_loss = -torch.log(sigmoid_diff + eps).mean()
        logits = self.fc(entity_output)


        class_loss,aug_entity_output = self.classifer(entity_output,self.fc,y)
        logits_aug = self.fc(final_embeds)

        y_hat = logits.argmax(-1)
        y_hat_aug = logits_aug.argmax(-1)
        return logits, y, y_hat, class_loss , logits_aug, y_hat_aug, contrastive_loss
