from collections import OrderedDict
import transformers as tfs
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead
from transformers.modeling_utils import ModuleUtilsMixin
import operator
import numpy as np
from softmasked_bert.utils.evaluations import compute_corrector_prf, compute_sentence_level_prf
import logging
import pytorch_lightning as pl
from softmasked_bert.solver.build import make_optimizer, build_lr_scheduler

class DetectionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.gru = nn.GRU(
            self.config.hidden_size,
            self.config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=self.config.hidden_dropout_prob,
            bidirectional=True,
        )

    def forward(self, hidden_states):
        output, _ = self.gru(hidden_states)
        prob = self.sigmoid(self.linear(output))
        return prob

class BertCorrectionModel(torch.nn.Module, ModuleUtilsMixin):
    def __init__(self, config, tokenizer, device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.embeddings = BertEmbeddings(self.config)
        self.corrector = BertEncoder(self.config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls = BertOnlyMLMHead(self.config)
        self._device = device

    def forward(self,texts,prob,embed=None,cor_labels=None,residual_connection=False):
        text_labels = self.tokenizer(cor_labels,padding=True,return_tensors='pt')['input_ids'].to(self._device)
        text_labels[text_labels == 0] = -100
        encoded_texts = self.tokenizer(texts,padding=True,return_tensors='pt').to(self._device)
        if embed is None:
            embed = self.embeddings(input_ids=encoded_texts['input_ids'],token_type_ids=encoded_texts['token_type_ids'])
        mask_embed = self.embeddings(torch.ones_like(prob.squeeze(-1)).long()*self.mask_token_id).detach()
        cor_embed = prob * mask_embed + (1 - prob) * embed

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(encoded_texts['attention_mask'], encoded_texts['input_ids'].size(),encoded_texts['input_ids'].device)
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        encoder_outputs = self.corrector(
            cor_embed,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output + embed if residual_connection else sequence_output
        prediction_scores = self.cls(sequence_output)
        output = (prediction_scores, sequence_output)

        loss_fct = nn.CrossEntropyLoss()
        cor_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size),text_labels.view(-1))
        output = (cor_loss, ) + output
        return output

    def load_from_transformers_state_dict(self, gen_fp):
        state_dict = OrderedDict()
        gen_state_dict = tfs.AutoModelForMaskedLM.from_pretrained(gen_fp).state_dict()
        for k, v in gen_state_dict.items():
            name = k
            if name.startswith('bert'):
                name = name[5:]
            if name.startswith('encoder'):
                name = f'corrector.{name[8:]}'
            if 'gamma' in name:
                name = name.replace('gamma', 'weight')
            if 'beta' in name:
                name = name.replace('beta', 'bias')
            state_dict[name] = v
        self.load_state_dict(state_dict, strict=False)

class BaseM(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self._logger = logging.getLogger(cfg.MODEL.NAME)

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [scheduler]

    def on_validation_epoch_start(self) -> None:
        self._logger.info('Valid.')

    def on_test_epoch_start(self) -> None:
        self._logger.info('Testing...')

class TrainModelStep(BaseM):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.w = cfg.MODEL.HYPER_PARAMS[0]

    def training_step(self, batch):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        return loss

    def validation_step(self, batch):
        ori_text, cor_text, det_labels = batch
        outputs = self.forward(ori_text, cor_text, det_labels)
        loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
        det_y_hat = (outputs[2] > 0.5).long()
        cor_y_hat = torch.argmax((outputs[3]), dim=-1)
        encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
        encoded_x.to(self._device)
        cor_y = encoded_x['input_ids']
        cor_y_hat *= encoded_x['attention_mask']

        results = []
        det_acc_labels = []
        cor_acc_labels = []
        for src, tgt, predict, det_predict, det_label in zip(ori_text, cor_y, cor_y_hat, det_y_hat, det_labels):
            _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
            _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
            _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
            cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
            det_acc_labels.append(det_predict[1:len(_src) + 1].equal(det_label[1:len(_src) + 1]))
            results.append((_src,_tgt,_predict,))

        return loss.cpu().item(), det_acc_labels, cor_acc_labels, results

    def validation_epoch_end(self, outputs) -> None:
        det_acc_labels = []
        cor_acc_labels = []
        results = []
        for out in outputs:
            det_acc_labels += out[1]
            cor_acc_labels += out[2]
            results += out[3]
        loss = np.mean([out[0] for out in outputs])
        self.log('val_loss', loss)
        self._logger.info(f'loss: {loss}')
        self._logger.info(f'Detection:\n'f'acc: {np.mean(det_acc_labels):.4f}')
        self._logger.info(f'Correction:\n'f'acc: {np.mean(cor_acc_labels):.4f}')
        compute_corrector_prf(results, self._logger)
        compute_sentence_level_prf(results, self._logger)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        self._logger.info('Test.')
        self.validation_epoch_end(outputs)

    def predict(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        inputs.to(self.cfg.MODEL.DEVICE)
        with torch.no_grad():
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs[1], dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        return rst

class SoftMaskedBertModel(TrainModelStep):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.config = tfs.AutoConfig.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.detector = DetectionNetwork(self.config)
        self.tokenizer = tokenizer
        self.corrector = BertCorrectionModel(self.config, tokenizer,cfg.MODEL.DEVICE)
        self.corrector.load_from_transformers_state_dict(self.cfg.MODEL.BERT_CKPT)
        self._device = cfg.MODEL.DEVICE

    def forward(self, texts, cor_labels=None, det_labels=None):
        encoded_texts = self.tokenizer(texts,padding=True,return_tensors='pt').to(self._device)
        embed = self.corrector.embeddings(input_ids=encoded_texts['input_ids'],token_type_ids=encoded_texts['token_type_ids'])
        prob = self.detector(embed)
        cor_out = self.corrector(texts,prob,embed,cor_labels,residual_connection=True)

        if det_labels is not None:
            det_loss_fct = nn.BCELoss()
            active_loss = encoded_texts['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            outputs = (det_loss, cor_out[0], prob.squeeze(-1)) + cor_out[1:]
        else:
            outputs = (prob.squeeze(-1), ) + cor_out

        return outputs
