import torch.nn as nn
import torch
from transformers import AutoModel, AutoConfig

from . import cfg


class MyModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(MyModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config["model_name"])

        if pretrained:
            self.roberta = AutoModel.from_pretrained(config["model_name"])
        else:
            self.roberta = AutoModel.from_config(self.config)

        self.pre_classifier = nn.Linear(cfg.HIDDEN_SIZE + 1, cfg.HIDDEN_SIZE + 1)
        self.classifier = nn.Linear(cfg.HIDDEN_SIZE + 1, cfg.NUM_LABELS)
        self.dropout = nn.Dropout(cfg.HIGH_DROPOUT)

        for module in [self.pre_classifier, self.classifier, self.dropout]:
            self._init_weights(module)

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, lengths):
        return_dict = self.config.use_return_dict

        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = torch.cat(
            [hidden_state[:, 0], lengths.unsqueeze(dim=1)], dim=1
        )  # (bs, dim+1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim+1)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim+1)
        pooled_output = self.dropout(pooled_output)  # (bs, dim+1)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        return logits

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--high_dropout", type=int, default=cfg.HIGH_DROPOUT)
        return parser
