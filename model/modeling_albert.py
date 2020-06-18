from transformers import AlbertModel, AlbertPreTrainedModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
from .Layers import MultiLinearLayer


class AlbertForConversationalQuestionAnswering(AlbertPreTrainedModel):
    def __init__(
            self,
            config,
            n_layers=2,
            activation='relu',
            beta=100,
    ):
        super(AlbertForConversationalQuestionAnswering, self).__init__(config)
        self.albert = AlbertModel(config)
        hidden_size = config.hidden_size
        self.rational_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.logits_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 2, activation)
        self.unk_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.attention_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.yn_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 2, activation)
        self.beta = beta

        self.init_weights()

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            start_positions=None,
            end_positions=None,
            rational_mask=None,
            cls_idx=None,
            head_mask=None,
    ):

        outputs = self.albert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        # last layer hidden-states of sequence, first token:classification token
        final_hidden, pooled_output = outputs
        # rational_logits
        rational_logits = self.rational_l(final_hidden)
        rational_logits = torch.sigmoid(rational_logits)
        final_hidden = final_hidden * rational_logits

        # attention layer to cal logits
        attention = self.attention_l(final_hidden).squeeze(-1)
        attention.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        attention = F.softmax(attention, dim=-1)
        attention_pooled_output = (attention.unsqueeze(-1) * final_hidden).sum(dim=-2)

        # on to find answer in the article
        segment_mask = token_type_ids.type(final_hidden.dtype)
        rational_logits = rational_logits.squeeze(-1) * segment_mask

        # get span logits
        logits = self.logits_l(final_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_logits, end_logits = start_logits * rational_logits, end_logits * rational_logits
        start_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float('inf'))

        # cal unkown/yes/no logits
        unk_logits = self.unk_l(pooled_output)
        yn_logits = self.yn_l(attention_pooled_output)
        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        # start_positions and end_positions is None when evaluate
        # return loss during training
        # return logits during evaluate
        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx

            new_start_logits = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            new_end_logits = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # rational part
            alpha = 0.25
            gamma = 2.

            # use rational span to help calculate loss
            rational_mask = rational_mask.type(final_hidden.dtype)
            rational_loss = -alpha * ((1 - rational_logits)**gamma) * rational_mask * torch.log(rational_logits + 1e-7) \
                            - (1 - alpha) * (rational_logits**gamma) * (1 - rational_mask) * \
                            torch.log(1 - rational_logits + 1e-7)

            rational_loss = (rational_loss * segment_mask).sum() / segment_mask.sum()

            total_loss = (start_loss + end_loss) / 2 + rational_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits