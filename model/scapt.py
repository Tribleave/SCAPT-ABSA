import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel
from model.module.bert_post_ln import BertMLMHead, BertPostLayerNormalizationModel, ABSAOutput
from model.module.bert_pre_ln import BertPreLayerNormalizationModel


class SCAPT(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", 'cls.predictions.bias',
                                          'cls.predictions.transform.dense.weight',
                                          'cls.predictions.transform.dense.bias',
                                          'cls.predictions.decoder.weight',
                                          'cls.seq_relationship.weight',
                                          'cls.seq_relationship.bias',
                                          'cls.predictions.transform.LayerNorm.weight',
                                          'cls.predictions.transform.LayerNorm.bias']
    _keys_to_ignore_on_load_missing = [r"position_ids", r"decoder.bias", r"classifier",
                                       'cls.bias', 'cls.transform.dense.weight',
                                       'cls.transform.dense.bias', 'cls.transform.LayerNorm.weight',
                                       'cls.transform.LayerNorm.bias', 'cls.decoder.weight',
                                       'cls_representation.weight', 'cls_representation.bias',
                                       'aspect_representation.weight', 'aspect_representation.bias']

    def __init__(self, config, hidden_size=256):
        super().__init__(config)

        if config.model == 'BERT':
            self.bert = BertPostLayerNormalizationModel(config, add_pooling_layer=False)
        elif config.model == 'TransEnc':
            self.bert = BertPreLayerNormalizationModel(config, add_pooling_layer=False)
        else:
            raise TypeError(f"Not supported model {config['model']}")
        self.cls = BertMLMHead(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_representation = nn.Linear(config.hidden_size, hidden_size)
        self.aspect_representation = nn.Linear(config.hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(2 * hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        aspect_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        multi_card=False,
        has_opposite_labels=False,
        pretrain_ce=False
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        expand_aspect_mask = (1 - aspect_mask).unsqueeze(-1).bool()
        cls_hidden = self.cls_representation(sequence_output[:, 0])
        aspect_hidden = torch.div(torch.sum(sequence_output.masked_fill(expand_aspect_mask, 0), dim=-2),
                                  torch.sum(aspect_mask.float(), dim=-1).unsqueeze(-1))
        aspect_hidden = self.aspect_representation(aspect_hidden)
        merged = self.dropout(torch.cat((cls_hidden, aspect_hidden), dim=-1))
        sentiment = self.classifier(merged)
        if multi_card:
            if has_opposite_labels:
                return cls_hidden, outputs.last_hidden_state[:, 0], masked_lm_loss
            else:
                return outputs.last_hidden_state[:, 0], masked_lm_loss
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return sentiment, cls_hidden, masked_lm_loss, output
        return ABSAOutput(
            sentiment=sentiment,
            loss=masked_lm_loss,
            cls_hidden=cls_hidden,
            logits=prediction_scores,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
