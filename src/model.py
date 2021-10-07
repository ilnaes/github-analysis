import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

import cfg


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = AutoConfig.from_pretrained(config["model_name"], num_labels=5)
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=5,
        )

        # self.high_dropout = nn.Dropout(config["high_dropout"])
        # self.norm = nn.LayerNorm(self.config.hidden_size)
        # self.classifier = nn.Linear(self.config.hidden_size, 1)

        # self._init_weights(self.norm)
        # self._init_weights(self.classifier)

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

    def forward(self, **kwargs):
        out = self.roberta(**kwargs)
        # out = self.roberta(**kwargs)["pooler_output"]
        # out = self.norm(out)
        # out = self.high_dropout(out)
        # out = self.classifier(out)
        return out

        # mean-max pooling
        # out = torch.stack(
        #     tuple(out[-i - 1] for i in range(cfg.N_LAST_HIDDEN)), dim=0
        # )
        # out_mean = torch.mean(out, dim=0)
        # out_max, _ = torch.max(out, dim=0)
        # out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        # out = torch.mean(
        #     torch.stack(
        #         [self.classifier(self.high_dropout(out)) for _ in range(5)],
        #         dim=0,
        #     ),
        #     dim=0,
        # )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--high_dropout", type=int, default=cfg.HIGH_DROPOUT)
        return parser