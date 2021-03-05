import torch
from torch import nn
from math import sqrt
import numpy as np
from hparams import hparams


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    mask = torch.zeros(lengths.shape[0], max_len, device=lengths.device).byte().zero_()
    for idx, l in enumerate(lengths):
        mask[idx][:l] = 1
    return mask



# init初始化, 这就叫 Norm? Tacotron 要求的? 不知有啥用 --Linear
class Linear_init(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias, w_init_gain):
        super(Linear_init, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.linear(x)
        return x



# init初始化, 这就叫 Norm? Tacotron 要求的? 不知有啥用 --CNN
# -> (B, channel=32, Text_length)
class Conv1d_init(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, w_init_gain):
        super(Conv1d_init, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.conv(x)
        return x




# Local sensitive
# [1] Rayhane 用 cumulate, NVIDIA 用两通道: (cumulate, previous)
# [2] 指定了 padding 的个数, 保持 1d 上大小不变, 使用 zero-padding
class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        # CNN-1d
        NVIDIA_local_channel = 2
        self.conv = Conv1d_init(in_channels=NVIDIA_local_channel,
                                         out_channels=n_filters,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding=int((kernel_size - 1) / 2),
                                         bias=False, 
                                         w_init_gain='linear')


        # Dense
        self.dense = Linear_init(n_filters, attention_dim, bias=False, w_init_gain='tanh')



    # x 代表了  attention_weights_cat: (cumulate, previous)
    def forward(self, x):
        # x (B, 2, Text_length)
        x = self.conv(x)                # -> (B, n_filters=32, Text_length)
        x = x.transpose(1, 2)           # -> (B, Text_length, n_filters=32)
        x = self.dense(x)               # -> (B, Text_length, attention_dim=128)
        return x





class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, location_n_filters, location_kernel_size):
        super(Attention, self).__init__()
        self.query_linear =  Linear_init(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh'  )
        # 类外使用 memory_linear, 只进行一次 process
        self.memory_linear = Linear_init(embedding_dim,     attention_dim, bias=False, w_init_gain='tanh'  )
        # 改了一点, 把 bias=Flase -> bias=True
        self.score_dot =     Linear_init(attention_dim,     1,             bias=True, w_init_gain='linear')


        # local sensitive
        self.local = LocationLayer(location_n_filters, location_kernel_size, attention_dim)

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)


    # 算 add dot score 相关分数, 未归一化
    # PARAMS
    # ------
    # query: decoder output and then attention rnn (B, attention_rnn_dim)
    # processed_memory: processed encoder outputs (B, Text_length, attention_dim=128)
    # attention_weights_cat: tuple of cumulative and prev (B, 2, Text_length)

    # RETURNS
    # -------
    # alignment (batch, Text_length)
    def _get_alignment_scores(self, query, processed_memory, attention_weights_cat):
        # 简称
        q = query                               # (B, attention_rnn_dim=1024)
        k = processed_memory                    # (B, Text_length, attention_dim=128)
        local_input = attention_weights_cat     # (B, 2, Text_length)


        # 开始计算
        q = q.unsqueeze(1)                      # (B, 1, attention_rnn_dim=1024)
        q = self.query_linear(q)                # (B, 1, attention_dim=128)
        local_input = self.local(local_input)   # (B, 2, Text_length) -> (B, Text_length, attention_dim=128)
        

        # dot score
        scores = self.score_dot(torch.tanh(q + local_input + k))  # (B, Text_length, 128) -> (B, Text_length, 1)
        scores = scores.squeeze(-1)                             # (B, Text_length)
        return scores



    # 注意, 当前的 attention 代码的输出 (contex) 就是 encoder_outputs 的一帧加权, 和 Rayhane 的不同 差别很大
    # attention_rnn_last_output: attention rnn last output, 未变成 mel 和 stopToken 的 feature (B, attention_rnn_dim=1024)
    # memory: encoder outputs, 用来作为 key, 和 Rayhane 不同 (B, Text_length, encoder_output_dim[2]=512)
    # processed_memory: processed encoder outputs, 用来作为 add score 相关性的基底 (B, Text_length, attention_dim=128)
    # attention_weights_cat: previous and cummulative attention weights (B, 2, Text_length)
    # mask_seq: memory 序列中后面几个需要根据 mask_seq 来 mask (B, Text_length)
    def forward(self, attention_rnn_last_output, memory, processed_memory, attention_weights_cat, reverse_memory_mask):
        # scores
        scores = self._get_alignment_scores(attention_rnn_last_output, processed_memory, attention_weights_cat)


        # masks, in-place 操作 mask 应该没事吧
        if reverse_memory_mask is not None:
            mask_min_inf = -float("inf")
            scores.data.masked_fill_(reverse_memory_mask, mask_min_inf)


        # energys / attentions
        # (B, Text_length) -> (B, Text_length)
        now_attention_weights = self.softmax(scores)                                 


        # context
        context = torch.bmm(now_attention_weights.unsqueeze(dim=1), memory)    # bmm 去掉 B: 矩阵 (1, Text_length) * (Text_length, attention_dim=128)
        context = context.squeeze(1)                   # (1, attention_dim=128) -> (attention_dim=128)

        return context, now_attention_weights




class Prenet(nn.Module):
    def __init__(self, mel_dim, prenet_dim, dropout_p): # mel_dim=80, prenet_dim=[256, 256]
        super(Prenet, self).__init__()
        self.pre1 = Linear_init(in_dim=mel_dim, out_dim=prenet_dim[0], bias=False, w_init_gain='linear')
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout_p)


        self.pre2 = Linear_init(in_dim=prenet_dim[0], out_dim=prenet_dim[1], bias=False, w_init_gain='linear')
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout_p)


    def forward(self, x):
        x = self.pre1(x)
        x = self.relu1(x)
        x = self.drop1(x)


        x = self.pre2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        return x



class Postnet(nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        # 5层 CNN
        self.postnet_num_convolutions = hparams.postnet_num_convolutions # 5
        self.postnet_embedding_dim_in = hparams.postnet_embedding_dim_in # [80, 512, 512, 512]
        self.postnet_embedding_dim_out = hparams.postnet_embedding_dim_out # [512, 512, 512, 512]
        self.final_postnet_embedding_dim_in = hparams.final_postnet_embedding_dim_in # 512
        self.final_postnet_embedding_dim_out = hparams.final_postnet_embedding_dim_out # 80
        assert self.postnet_num_convolutions - 1 == len(self.postnet_embedding_dim_in) # 最后一层单独写 final
        # kernel size
        self.postnet_kernel_size = hparams.postnet_kernel_size
        # drop_p
        self.postnet_dropout_p = hparams.postnet_dropout_p
        # List 的方式搭建
        self.convolution_list = nn.ModuleList()


        for i in range(0, hparams.self.postnet_num_convolutions - 1):
            self.convolution_list.append(
                nn.Sequential(
                    Conv1d_init(in_channels=self.postnet_embedding_dim_in[i],
                                out_channels=self.postnet_embedding_dim_out[i],
                                kernel_size=hparams.postnet_kernel_size,
                                stride=1,
                                padding=int((self.postnet_kernel_size - 1) / 2),
                                bias=True,
                                w_init_gain='tanh'),
                    nn.BatchNorm1d(self.postnet_embedding_dim_out[i])),
                    nn.Tanh(),
                    nn.Dropout(p=self.postnet_dropout_p),
            )

        self.convolution_list.append(
                nn.Sequential(
                    Conv1d_init(in_channels=self.final_postnet_embedding_dim_in,
                                out_channels=self.final_postnet_embedding_dim_out,
                                kernel_size=hparams.postnet_kernel_size,
                                stride=1,
                                padding=int((self.postnet_kernel_size - 1) / 2),
                                bias=True,
                                w_init_gain='linear'),
                    # nn.BatchNorm1d(self.postnet_embedding_dim_out[i])),
                    # nn.Tanh(),
                    nn.Dropout(p=self.postnet_dropout_p),
                )
        )


    def forward(self, x):
        for i in range(self.postnet_num_convolutions):
            x = self.convolution_list[i](x)
        return x




class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.encoder_embedding_dim = hparams.encoder_embedding_dim # 512
        self.encoder_n_convolutions = hparams.encoder_n_convolutions # 3
        self.encoder_input_dim = hparams.encoder_input_dim # [512, 512, 512]
        self.encoder_output_dim = hparams.encoder_output_dim # [512, 512, 512]
        assert self.encoder_embedding_dim == self.encoder_input_dim[0]

        self.encoder_kernel_size = hparams.encoder_kernel_size # 5
        self.encoder_cnn_dropout_p = hparams.encoder_cnn_dropout_p # 0.5


        self.convolution_list = nn.ModuleList()
        for i in range(self.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, w_init_gain):
                Conv1d_init(in_channels=self.encoder_input_dim[i],
                            out_channels=self.encoder_output_dim[i],
                            kernel_size=self.encoder_kernel_size, 
                            stride=1,
                            padding=int((self.encoder_kernel_size - 1) / 2),
                            bias=True,
                            w_init_gain='relu'),
                nn.BatchNorm1d(self.encoder_output_dim[i]),
                nn.ReLU(),
                nn.Dropout(p=self.encoder_cnn_dropout_p),
                )
            self.convolution_list.append(conv_layer)


        self.lstm = nn.LSTM(input_size=hparams.encoder_output_dim[-1], 
                            hidden_size=int(hparams.encoder_output_dim[-1] / 2), 
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=True)


    def forward(self, x):
        # (B, 512, Text_length)
        assert x.shape[1] == 512

        # CNN
        for i in range(self.encoder_n_convolutions):
            x = self.convolution_list[i](x)

        # LSTM
        # (B, 512, Text_length) -> (B, Text_length, 512)
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)

        return outputs



class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        # reduce factor
        self.n_mel_channels = hparams.n_mel_channels # 和 audio 模块共用这个参数
        self.n_frames_per_step = hparams.n_frames_per_step
        assert self.n_frames_per_step == 1

        # 两个 rnn 和两层 prenet
        self.attention_rnn_dim = hparams.attention_rnn_dim # 和 attention 模块共用这个参数
        self.attention_rnn_dropout_p = hparams.attention_rnn_dropout_p # 0.1, attention_rnn 用的
        self.final_encoder_output_dim = hparams.encoder_output_dim[-1] # 和 encoder 模块共用这个参数, prenet_out 和 context 会拼接
        assert hparams.encoder_output_dim[-1] == hparams.encoder_embedding_dim

        self.decoder_rnn_dim = hparams.decoder_rnn_dim # 比 Rayhane 多一个 decoder rnn, 1 + 1 模式
        self.decoder_rnn_dropout_p = hparams.decoder_rnn_dropout_p # 0.1, decoder_rnn 用的

        self.prenet_dim = hparams.prenet_dim # [256, 256]
        self.prenet_dropout_p = hparams.prenet_dropout_p # 0.5, prenet 用的

        # stop
        self.max_decoder_steps = hparams.max_decoder_steps # 1000
        self.gate_threshold = hparams.gate_threshold # 0.5
        

        # prenet, 和 r9y9 的不同, 和论文以及 Rayhane 的一致: 不管 reduce factor 只用最后一帧
        self.prenet = Prenet(mel_dim=self.n_mel_channels, prenet_dim=self.prenet_dim, dropout_p=self.prenet_dropout_p)


        # attention_rnn
        self.attention_rnn = nn.LSTMCell(input_size =self.prenet_dim[-1] + self.final_encoder_output_dim, hidden_size =self.attention_rnn_dim)
        self.attention_rnn_drop = nn.Dropout(p=self.attention_rnn_dropout_p)

        # attention
        # def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, location_n_filters, location_kernel_size)
        self.attention_dim = hparams.attention_dim
        self.location_n_filters = hparams.location_n_filters
        self.location_kernel_size = hparams.location_kernel_size

        self.attention_module = Attention(
                                attention_rnn_dim=self.attention_rnn_dim, 
                                embedding_dim=self.final_encoder_output_dim,
                                attention_dim=self.attention_dim, 
                                location_n_filters=self.location_n_filters,
                                location_kernel_size=self.location_kernel_size)


        # decoder_rnn
        # 和之前的不一样
        self.decoder_rnn = nn.LSTMCell(input_size=self.attention_rnn_dim + self.final_encoder_output_dim, hidden_size=hparams.decoder_rnn_dim)
        self.decoder_rnn_drop = nn.Dropout(p=self.decoder_rnn_dropout_p)

        # FC
        # def __init__(self, in_dim, out_dim, bias, w_init_gain):
        self.linear_projection = Linear_init(in_dim=self.decoder_rnn_dim + self.final_encoder_output_dim,
                                             out_dim=self.n_mel_channels * self.n_frames_per_step,
                                             bias=True,
                                             w_init_gain='linear')

        self.gate_layer = Linear_init(in_dim=self.decoder_rnn_dim + self.final_encoder_output_dim,
                                      out_dim=1,
                                      bias=True, 
                                      w_init_gain='sigmoid')



    def get_go_frame(self, memory):
        B = memory.size(0)

        # (B, 80) 只用一帧作为查询
        decoder_input = torch.zeros((B, self.n_mel_channels), dtype=memory.dtype, device=memory.device)
        return decoder_input

        

    def initialize_decoder_states(self, memory):
        B = memory.size(0)
        Text_length = memory.size(1)
        

        # 开始初始化
        self.attention_rnn_hidden = torch.zeros((B, self.attention_rnn_dim), dtype=memory.dtype, device=memory.device)
        self.attention_rnn_cell = torch.zeros((B, self.attention_rnn_dim), dtype=memory.dtype, device=memory.device)

        self.decoder_rnn_hidden = torch.zeros((B, self.decoder_rnn_dim), dtype=memory.dtype, device=memory.device)
        self.decoder_rnn_cell =torch.zeros((B, self.decoder_rnn_dim), dtype=memory.dtype, device=memory.device)

        self.attention_weights = torch.zeros((B, Text_length), dtype=memory.dtype, device=memory.device)
        self.attention_weights_cum = torch.zeros((B, Text_length), dtype=memory.dtype, device=memory.device)
        self.context = torch.zeros((B, self.final_encoder_output_dim), dtype=memory.dtype, device=memory.device)



    def prepare_whole_loop_variable(self, memory, reverse_memory_mask):
        self.memory = memory
        self.processed_memory = self.attention_module.memory_linear(memory)
        self.reverse_memory_mask = reverse_memory_mask



    def decode_one_step(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.context), -1)
        self.new_attention_rnn_hidden, self.new_attention_rnn_cell = self.attention_rnn(cell_input, (self.attention_rnn_hidden, self.attention_rnn_cell))
        self.new_attention_rnn_hidden = self.attention_rnn_drop(self.new_attention_rnn_hidden)



        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)


        # def forward(self, attention_rnn_last_output, memory, processed_memory, attention_weights_cat, mask_seq):
        new_context, new_attention_weights = self.attention_module(
            attention_rnn_last_output=self.new_attention_rnn_hidden, 
            memory=self.memory, 
            processed_memory=self.processed_memory,
            attention_weights_cat=attention_weights_cat, 
            reverse_memory_mask=self.reverse_memory_mask)


        decoder_input = torch.cat((self.new_attention_rnn_hidden, new_context), -1)

        self.new_decoder_rnn_hidden, self.new_decoder_rnn_cell = self.decoder_rnn(decoder_input, (self.decoder_rnn_hidden, self.decoder_rnn_cell))
        self.new_decoder_rnn_hidden = self.decoder_rnn_drop(self.new_decoder_rnn_hidden)

        # mel
        new_decoder_hidden_context = torch.cat((self.new_decoder_rnn_hidden, new_context), dim=1)
        new_decoder_output = self.linear_projection(new_decoder_hidden_context)
        assert new_decoder_output.shape[1] == 80


        # stop
        new_decoder_hidden_context = torch.cat((self.new_decoder_rnn_hidden, new_context), dim=1)
        new_gate_prediction = self.gate_layer(new_decoder_hidden_context)
        assert new_gate_prediction.shape[1] == 1
        new_gate_prediction = new_gate_prediction.squeeze(dim=1)


        # [1]
        self.attention_rnn_hidden = self.new_attention_rnn_hidden
        self.attention_rnn_cell = self.new_attention_rnn_cell

        # [2]
        self.decoder_rnn_hidden = self.new_decoder_rnn_hidden
        self.decoder_rnn_cell = self.new_decoder_rnn_cell

        # [3]
        self.context = new_context
        self.attention_weights = new_attention_weights
        self.attention_weights_cum += new_context

        return new_decoder_output, new_gate_prediction, new_attention_weights



    # PARAMS
    # ------
    # memory: Encoder outputs, (B, Text_length, 512)
    # decoder_inputs: mel-specs-teacher, (B, output_lengths, 80) / None
    # memory_lengths: memory lengths for attention masking, (B,) / None

    # RETURNS
    # -------
    # 后面还有 postnet, 为了 CNN 方便, 就先 channel first 吧, 先不修正
    # mel_outputs: (B, n_mel_channels-80, output_length)
    # gate_outputs: (B, output_length)
    # alignments: (B, output_length, max_Text_length)
    def forward(self, memory, decoder_inputs, memory_lengths):
        # check 参数
        # memory, (B, Text_length, 512)
        assert memory.shape[2] == 512

        # decoder_inputs, (B, output_lengths, 80) / None
        if decoder_inputs is not None:
            assert decoder_inputs.shape[2] == 80

        # memory_lengths, (B,) / None
        if memory_lengths is not None:
            assert len(memory_lengths.shape) == 1

        # 增加一个临时参数, tag, 为了方便
        if decoder_inputs is not None:
            teacherForce_tag = True
        else:
            teacherForce_tag = False


        # 开始了:
        # --- (1) --- go frame
        if teacherForce_tag:
            decoder_input = self.get_go_frame(memory=memory).unsqueeze(0)
        else:
            decoder_input = self.get_go_frame(memory=memory)

        # --- (2) --- 准备 两个 rnn 的初始状态, 以及
        self.initialize_decoder_states(memory=memory)

        # --- (3) --- memory, processed_memory, reverse_memory_mask 的准备
        if teacherForce_tag and (memory_lengths is not None):
            self.prepare_whole_loop_variable(memory=memory, reverse_memory_mask=~get_mask_from_lengths(memory_lengths))
        else:
            self.prepare_whole_loop_variable(memory=memory, reverse_memory_mask=None)
        

        # --- (4) --- 自回归, alignments 就是 attention_weights 的图
        mel_outputs, gate_outputs, alignments = [], [], []

        if teacherForce_tag:
            # 构建自回归的 teacherForce 数组:
            # [1] (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
            decoder_inputs = decoder_inputs.transpose(0, 1)
            # [2] cat
            decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)

            # prenet
            decoder_inputs = self.prenet(decoder_inputs)


            # loop
            while len(mel_outputs) < decoder_inputs.size(0) - 1:
                decoder_input = decoder_inputs[len(mel_outputs)]
                mel_output, gate_output, attention_weights = self.decode_one_step(decoder_input)

                # add List
                mel_outputs += [mel_output]
                gate_outputs += [gate_output]
                alignments += [attention_weights]
        else:
            while True:
                # (B, Time, dim) 和 (B, dim) 均可以使用 prenet
                decoder_input = self.prenet(decoder_input)

                # loop 的核心
                mel_output, gate_output, alignment = self.decode_one_step(decoder_input)

                mel_outputs += [mel_output]
                gate_outputs += [gate_output]
                alignments += [alignment]

                if torch.sigmoid(gate_output.data) > self.gate_threshold:
                    break
                elif len(mel_outputs) * self.n_frames_per_step >= self.max_decoder_steps:
                    assert False
                    break

                decoder_input = mel_output



        # 对 while 循环进行总结
        # List -> Tensor
        # [1] mels
        # (output_length, B, n_mel_channels) 
        # -> (B, output_length, n_mel_channels)
        # -> (B, n_mel_channels, output_length)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).transpose(1, 2)
        assert mel_outputs.shape[1] == 80
        
        # [2] gate_outputs
        # (output_length, B) -> (B, output_length)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        assert gate_outputs.shape[1] == mel_outputs.shape[2]

        # [3] alignments
        # (output_length, B, max_Text_length) -> (B, output_length, max_Text_length)
        alignments = torch.stack(alignments).transpose(0, 1)
        assert alignments.shape[1] == mel_outputs.shape[2]

        return mel_outputs, gate_outputs, alignments




class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        # lookup-table 和 weights 初始化
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.encoder_embedding_dim) 
        std = sqrt(2.0 / (hparams.n_symbols + hparams.encoder_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder_module = Encoder(hparams)
        self.decoder_module = Decoder(hparams)
        self.postnet_module = Postnet(hparams)


    def forward(self, text_padded, text_lengths, mel_padded):
        # text_padded, (B, Text_length) 
        # text_lengths, (B, ) / None
        # mel_padded, (B, output_length, 80) / None
        
        # 开始 flow
        # lookup-table
        embedded_inputs = self.embedding(text_padded)
        
        
        # encoder - module
        # (B, Text_length, 512) -> (B, 512, Text_length)
        embedded_inputs = embedded_inputs.transpose(1, 2)
        # (B, 512, Text_length) -> (B, Text_length, 512) (LSTM 的时候修正回来了)
        encoder_outputs = self.encoder_module(x=embedded_inputs)


        # decoder - module
        # [1] 输入
        # encoder_outputs 已经在 LSTM 时修正 -> (B, Text_length, 512)
        # mel_padded, (B, output_length, 80) / None
        # text_lengths 用于设计 reverse_mask_seq, (B, output_length) / None
        # [2] 输出
        # mels_pre, (B, 80, output_length)
        # gates_pre, (B, output_length)
        # alignments_pre, (B, output_length, max_Text_length)
        # [3] 训练或者预测:
        mels_pre, gates_pre, alignments_pre = self.decoder_module(memory=encoder_outputs, decoder_inputs=mel_padded, memory_lengths=text_lengths)

        
        # postnet - module
        # (B, 80, output_length) -> (B, 80, output_length)
        mels_pre_postnet = self.postnet_module(x=mels_pre) + mels_pre


        # 最后 -> (B, output_lenght, 80), 更习惯
        mels_pre_seq_first = mels_pre.transpose(1, 2)
        mels_pre_postnet_seq_first = mels_pre_postnet.transpose(1, 2)
        assert mels_pre_seq_first.shape[-1] == mels_pre_postnet_seq_first.shape[-1] and mels_pre_postnet_seq_first.shape[-1] == 80
        
        return mels_pre_seq_first, mels_pre_postnet_seq_first, gates_pre, alignments_pre

