# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules import MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM
from src.modules.dropout import IndependentDropout, SharedDropout
from src.modules.treecrf import CRFConstituency
from src.modules.module_fence_rnn import EncoderFenceDiscourseRnn, DecoderRNN
from src.utils.fn import parsingorder2spandfs
from src.utils import Config
from src.utils.alg import cky
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PointingDiscourseModel(nn.Module):
    """
    The implementation of Pointing Discourse Parser.

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_feats (int):
            Size of the feat vocabulary.
        n_labels (int):
            Number of labels.
        feat (str):
            Specifies which type of additional feature to use: 'char' | 'bert' | 'tag'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained langugae models like `XLNet` are also feasible.
            'tag': POS tag embeddings.
            Default: 'char'.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_feat_embed (int):
            Size of feature representations. Default: 100.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use. Required if feat='bert'.
            The final outputs would be the weight sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float):
            Dropout ratio of LSTM. Default: .33.
        n_mlp_span (int):
            Span MLP size. Default: 500.
        n_mlp_label  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        feat_pad_index (int):
            The index of the padding token in the feat vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    """

    def __init__(self,
                 n_words,
                 n_feats,
                 n_labels,
                 feat='char',
                 n_embed=100,
                 n_feat_embed=100,
                 n_char_embed=50,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 feat_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())
        # the embedding layer
        self.encoder = EncoderFenceDiscourseRnn(n_words=n_words,
                 n_feats=n_feats,
                 n_labels=n_labels,
                 feat=feat,
                 n_embed=n_embed,
                 n_feat_embed=n_feat_embed,
                 n_char_embed=n_char_embed,
                 bert=bert,
                 n_bert_layers=n_bert_layers,
                 mix_dropout=mix_dropout,
                 embed_dropout=embed_dropout,
                 n_lstm_hidden=n_lstm_hidden,
                 n_lstm_layers=n_lstm_layers,
                 lstm_dropout=lstm_dropout,
                 n_mlp_span=n_mlp_span,
                 n_mlp_label=n_mlp_label,
                 mlp_dropout=mlp_dropout,
                 feat_pad_index=feat_pad_index,
                 pad_index=pad_index,
                 unk_index=unk_index,
                 **kwargs)
        self.decoder=DecoderRNN(input_size=n_mlp_span * 2,
		                        hidden_size=n_lstm_hidden * 2,
		                        rnn_layers=n_lstm_layers,
		                        dropout=lstm_dropout)
        self.mlp_span_l_decoder = MLP(n_in=n_lstm_hidden * 2,
                                      n_out=n_mlp_span,
                                      dropout=mlp_dropout)
        self.mlp_span_r_decoder = MLP(n_in=n_lstm_hidden * 2,
                                      n_out=n_mlp_span,
                                      dropout=mlp_dropout)
        self.mlp_span_decoder = MLP(n_in=n_lstm_hidden * 2,
                                    n_out=n_mlp_span,
                                    dropout=mlp_dropout)
        self.span_attn = Biaffine(n_in=n_mlp_span,
                                  bias_x=True,
                                  bias_y=False)

        self.mlp_label_l = MLP(n_in=n_lstm_hidden * 4,
                               n_out=n_mlp_label,
                               dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=n_lstm_hidden * 4,
                               n_out=n_mlp_label,
                               dropout=mlp_dropout)

        # the Biaffine layers
        # self.span_attn = Biaffine(n_in=n_mlp_span,
        #                           bias_x=True,
        #                           bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label,
                                   n_out=n_labels,
                                   bias_x=True,
                                   bias_y=True)

        self.label_criterion = nn.CrossEntropyLoss()
        # self.pad_index = pad_index
        # self.unk_index = unk_index

    def forward(self):
        raise RuntimeError('Parsing Network does not have forward process.')

    def loss(self, words, feats, spans, labels, parsing_orders):
        fencepost, span_split, decoder_init_state = self.encoder(words, feats)
        batch_size, seq_len = words.shape
        _, _, dec_len = parsing_orders.shape
        lens = words.ne(self.args.pad_index).sum(1) - 1

        # pointing mask
        mask_l = lens.new_tensor(range(seq_len - 1)) > 0
        mask_r = lens.new_tensor(range(seq_len - 1)) <= (lens - 1).view(-1, 1, 1)
        mask_point = ~(mask_l & mask_r)
        mask_point = mask_point.expand(batch_size, dec_len, seq_len - 1)


        span_l = fencepost[torch.arange(batch_size).unsqueeze(1), parsing_orders[:, 0, :]]
        span_r = fencepost[torch.arange(batch_size).unsqueeze(1), parsing_orders[:, 2, :]]
        span_l = self.mlp_span_l_decoder(span_l)
        span_r = self.mlp_span_r_decoder(span_r)
        decoder_input = torch.cat([span_l, span_r], dim=-1)
        decoder_output, _ = self.decoder(input_hidden_states=decoder_input,
                                         last_hidden=decoder_init_state)
        decoder_output = self.mlp_span_decoder(decoder_output)

        # [batch_size, dec_len, seq_len-1]
        s_point = self.span_attn(decoder_output, span_split)
        s_point = s_point.masked_fill(mask_point, float('-inf'))
        s_point = F.log_softmax(s_point, dim=-1)

        s_gold = parsing_orders[:, 1, :].contiguous().view(-1)
        s_point = s_point.view(-1, seq_len - 1)
        mask_s = (s_gold != self.args.pad_index).float()
        num_s = int(torch.sum(mask_s).item())
        s_point = s_point[range(s_point.shape[0]), s_gold] * mask_s
        s_point[s_point != s_point] = 0

        # label mask
        _, _, span_len=spans.shape
        mask_span=torch.eye(span_len,span_len, dtype=torch.bool).to(spans.device)
        label_lens = labels.ne(self.args.pad_index).sum(1)


        l_lelf_point = fencepost[torch.arange(batch_size).unsqueeze(1), spans[:, 0, :]]
        l_split_point = fencepost[torch.arange(batch_size).unsqueeze(1), spans[:, 1, :]]
        l_right_point = fencepost[torch.arange(batch_size).unsqueeze(1), spans[:, 2, :]]

        l_left_span=torch.cat([l_lelf_point,l_split_point], dim=-1)
        l_right_span=torch.cat([l_split_point, l_right_point], dim=-1)

        l_left_span=self.mlp_label_l(l_left_span)
        l_right_span=self.mlp_label_r(l_right_span)

        s_label=self.label_attn(l_left_span,l_right_span).permute(0, 2, 3, 1)[:,mask_span,:]
        mask_label = lens.new_tensor(range(span_len)) < label_lens.view(-1, 1)
        # mask_label = mask_label & mask_label.new_ones(seq_len - 1, seq_len - 1).triu_(1)
        # mask_label = spans &
        if int(label_lens.max())>0:
            label_loss = self.label_criterion(s_label[mask_label], labels[mask_label])
        else:
            label_loss =0

        point_loss = -torch.sum(s_point) / num_s

        loss = point_loss + label_loss
        return loss

    def decode(self, words, feats, beam_size=1):
        fencepost, span_split, decoder_init_state = self.encoder(words, feats)
        batch_size, seq_len = words.shape
        dec_len = seq_len - 2
        lens = words.ne(self.args.pad_index).sum(1) - 1
        node_lens = lens - 2
        # print(lens)

        # pointing mask
        mask_l = lens.new_tensor(range(0, seq_len - 1)) > 0
        mask_r = lens.new_tensor(range(0, seq_len - 1)) <= (lens - 1).view(-1, 1, 1)
        mask_point = ~(mask_l & mask_r)

        # label mask
        # mask_label = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
        # mask_label = mask_label & mask_label.new_ones(seq_len - 1, seq_len - 1).triu_(1)

        # initialize the decoding stage
        num_hyp = 1
        last_hidden_state = decoder_init_state
        stacked_inputspan = lens.new_zeros(batch_size, num_hyp, 2, dec_len)
        stacked_parsing_order = lens.new_zeros(batch_size, num_hyp, 3, dec_len)
        hypothesis_scores = fencepost.new_zeros(batch_size, num_hyp)
        stacked_inputspan[:, 0, 1, 0] = lens - 1
        num_steps = dec_len - 1
        for t in range(num_steps):
            curr_input_l = stacked_inputspan[:, :, 0, t]
            curr_input_r = stacked_inputspan[:, :, 1, t]
            point_range_l = lens.new_tensor(range(0, seq_len - 1)) > \
                            curr_input_l.unsqueeze(-1).expand(batch_size, num_hyp, seq_len - 1)
            point_range_r = lens.new_tensor(range(0, seq_len - 1)) <= \
                            curr_input_r.unsqueeze(-1).expand(batch_size, num_hyp, seq_len - 1)
            point_range = ~(point_range_l & point_range_r)
            mask_decodelens = (t >= node_lens).view(batch_size, 1, 1).expand(batch_size, num_hyp, seq_len - 1)
            mask_no_parsing = ((curr_input_l.unsqueeze(-1).expand(point_range.shape).eq(0)) &
                    curr_input_r.unsqueeze(-1).expand(point_range.shape).eq(0))

            span_l = fencepost[torch.arange(batch_size).unsqueeze(1), curr_input_l]
            span_r = fencepost[torch.arange(batch_size).unsqueeze(1), curr_input_r]
            span_l = self.mlp_span_l_decoder(span_l)
            span_r = self.mlp_span_r_decoder(span_r)
            decoder_input = torch.cat([span_l, span_r], dim=-1)
            # print('step: ',t)
            # print(stacked_inputspan[0])
            # print(stacked_parsing_order[0])
            decoder_output, new_last_hidden_state = self.decoder(
                input_hidden_states=decoder_input.view(batch_size * num_hyp, 1, -1),
                last_hidden=last_hidden_state)
            decoder_output = self.mlp_span_decoder(decoder_output)
            decoder_output = decoder_output.view(batch_size, num_hyp, -1)

            s_point = self.span_attn(decoder_output, span_split)
            s_point = s_point.masked_fill(mask_point.expand(batch_size, num_hyp, seq_len - 1), float('-inf'))
            s_point = F.log_softmax(s_point, dim=-1)
            # print(s_point[0])
            s_point = s_point.masked_fill(point_range, float('-inf'))
            # print(s_point[0])
            s_point = s_point.masked_fill(mask_decodelens, 0)
            # print(s_point[0])
            s_point = s_point.masked_fill(mask_no_parsing, 0)
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + s_point
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch_size, -1),
                                                      dim=1, descending=True)
            # prev_num_hyp = num_hyp
            # num_hyp = (~point_range).view(batch_size, -1).sum(dim=1).max().clamp(max=1).item()
            # print(hypothesis_scores[0])
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            # print(hypothesis_scores[0])
            # print(hyp_index[0])
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index / (seq_len - 1)
            split_index = hyp_index % (seq_len - 1)
            # print(split_index)
            # print(curr_input_l)
            # print(base_index)
            hyp_l = curr_input_l.gather(dim=1, index=base_index)
            # print(hyp_l)
            hyp_r = curr_input_r.gather(dim=1, index=base_index)
            # print(split_index)
            # print(split_index.shape)
            base_index_expand = base_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_hyp, 2, dec_len)
            stacked_inputspan = stacked_inputspan.gather(dim=1, index=base_index_expand)
            base_index_parsing_order = base_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_hyp, 3, dec_len)
            stacked_parsing_order = stacked_parsing_order.gather(dim=1, index=base_index_parsing_order)
            stacked_parsing_order[:, :, 0, t] = hyp_l
            stacked_parsing_order[:, :, 1, t] = torch.where(
                (split_index > hyp_l) & (split_index <= hyp_r), split_index, hyp_l)
            stacked_parsing_order[:, :, 2, t] = hyp_r

            candidate_leftspan = stacked_inputspan[:, :, :, t + 1]
            candidate_leftspan[:, :, 0] = torch.where((split_index > hyp_l + 1) & (split_index < hyp_r),
                                                      hyp_l, candidate_leftspan[:, :, 0])
            candidate_leftspan[:, :, 1] = torch.where((split_index > hyp_l + 1) & (split_index < hyp_r),
                                                      split_index, candidate_leftspan[:, :, 1])
            stacked_inputspan[:, :, :, t + 1] = candidate_leftspan

            position_rightspan = (t + split_index - hyp_l).clamp(max=dec_len - 1)
            # print(position_rightspan)
            position_rightspan_expand = position_rightspan.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_hyp, 2, 1)
            candidate_rightspan = stacked_inputspan.gather(dim=3, index=position_rightspan_expand).squeeze(-1)
            candidate_rightspan[:, :, 0] = torch.where(1 + split_index < hyp_r, split_index,
                                                       candidate_rightspan[:, :, 0])
            candidate_rightspan[:, :, 1] = torch.where(1 + split_index < hyp_r, hyp_r,
                                                       candidate_rightspan[:, :, 1])
            stacked_inputspan.scatter_(dim=3, index=position_rightspan_expand, src=candidate_rightspan.unsqueeze(-1))

            batch_index = lens.new_tensor(range(batch_size)).view(batch_size, 1)
            # hx_index = (base_index + batch_index * prev_num_hyp).view(batch_size * num_hyp)
            hx_index = (base_index + batch_index * num_hyp).view(batch_size * num_hyp)
            if isinstance(new_last_hidden_state, tuple):
                new_hx, new_cx = new_last_hidden_state
                new_hx = new_hx[:, hx_index]
                new_cx = new_cx[:, hx_index]
                hx, cx = last_hidden_state
                last_hx = torch.where(hyp_r.eq(0), hx, new_hx)
                last_cx = torch.where(hyp_r.eq(0), cx, new_cx)
                last_hidden_state = (last_hx, last_cx)
                # decoder_init_state = (hx, cx)
            else:
                new_last_hidden_state = new_last_hidden_state[:, hx_index]
                last_hidden_state=torch.where(hyp_r.eq(0), last_hidden_state, new_last_hidden_state)

        final_stacked_parsing_order = stacked_parsing_order[:, 0, :, :-1]
        final_stacked_inputspan = stacked_inputspan[:,0,:,:-1]
        padding_nonparsing_mask = final_stacked_parsing_order[:, 2, :].ne(
            final_stacked_parsing_order[:, 1, :]).to(torch.long)
        _, padding_nonparsing_mask_index = torch.sort(padding_nonparsing_mask, dim=1, descending=True)
        final_parsing_order = final_stacked_parsing_order.gather(dim=-1, index=padding_nonparsing_mask_index.unsqueeze(
            1).expand(batch_size, 3, num_steps))
        final_parsing_length = padding_nonparsing_mask.sum(dim=-1)
        max_parsing_length = int(final_parsing_length.max())
        final_parsing_order = final_parsing_order[:,:,:max_parsing_length]

        l_lelf_point = fencepost[torch.arange(batch_size).unsqueeze(1), final_parsing_order[:, 0, :]]
        l_split_point = fencepost[torch.arange(batch_size).unsqueeze(1), final_parsing_order[:, 1, :]]
        l_right_point = fencepost[torch.arange(batch_size).unsqueeze(1), final_parsing_order[:, 2, :]]

        l_left_span = torch.cat([l_lelf_point, l_split_point], dim=-1)
        l_right_span = torch.cat([l_split_point, l_right_point], dim=-1)

        l_left_span = self.mlp_label_l(l_left_span)
        l_right_span = self.mlp_label_r(l_right_span)
        mask_span = torch.eye(max_parsing_length, max_parsing_length,dtype=torch.bool).to(final_parsing_length.device)
        if max_parsing_length>0:
            s_label = self.label_attn(l_left_span, l_right_span).permute(0, 2, 3, 1)[:, mask_span, :]
            pred_labels = s_label.argmax(-1).tolist()
        else:
            pred_labels =[]


        parsing_order_list = final_parsing_order.transpose(1, 2).tolist()

        for i, parsing_order_len in enumerate(final_parsing_length.tolist()):
            parsing_order_list[i] = parsing_order_list[i][:parsing_order_len]

        # print(parsing_order_list)
        # pred_spans = [parsingorder2spandfs(order) for order in parsing_order_list]
        # pred_labels = s_label.argmax(-1).tolist()
        preds = [[(i, k, j, label) for (i, k, j),label in zip(spans, labels)]
                 for spans, labels in zip(parsing_order_list, pred_labels)]
        return preds

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.encoder.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.encoder.word_embed.weight)
        return self
