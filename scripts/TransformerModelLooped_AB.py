import torch
import torch.nn as nn
from nano_gpt import GPT2Config, LayerNorm, Block
import math

class GPT2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # this is the same as huggingface repo:
        # initialize range: https://huggingface.co/transformers/v3.0.2/_modules/transformers/configuration_gpt2.html#GPT2Config
        # initialize function: https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_gpt2.html
        # search there _init_weights function
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs_embeds, which_layer, position_ids=None, rm_pos_embd=False, add_inputs_embeds=False, output_intermediate=False):
        device = inputs_embeds.device
        b, t, d = inputs_embeds.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        if position_ids is None:
           position_ids = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        if output_intermediate:
            embeds = [inputs_embeds]

        # forward the GPT model itself
        # tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)
        if rm_pos_embd:
            pos_emb = torch.zeros_like(pos_emb, device=device)
        x = self.transformer.drop(inputs_embeds + pos_emb)
        if which_layer == 'A':
            block = self.transformer.h[0]
        elif which_layer == 'B':
            block = self.transformer.h[1]
        if add_inputs_embeds:
            x = block(x + inputs_embeds)
        else:
            x = block(x)
        if output_intermediate:
            embeds.append(x)
        x = self.transformer.ln_f(x)
        if output_intermediate:
            return x, embeds

        return x
        
class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, pred_type='regression'):

        super(TransformerModel, self).__init__()
        self.freq = 2
        self.ind = 0
        configuration = GPT2Config()
        configuration.block_size = self.freq * n_positions + 1
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.dropout = 0.
        self.configuration = configuration

        self.n_positions = n_positions  # n = points in this setting
        self.n_dims = n_dims  # input dimension, d_in
        self.n_embd = n_embd  # d
        self.n_layer = n_layer
        self._pred_type = pred_type

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        if self._pred_type == 'regression':
            self._read_out = nn.Linear(n_embd, 1)
        elif self._pred_type == 'classification':
            self._read_out = nn.Linear(n_embd, MAX_NUM_CLASS)  # NOTE: hard-code

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, 2n, d_in + 1]
        """
        B, n, d = xs_b.shape
        device = xs_b.device

        ys_b_wide = torch.cat(
            (
                ys_b.view(B, n, 1),
                torch.zeros(B, n, d-1, device=device),
            ),
            axis=2,
        )

        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(B, self.freq * n, d)

        return zs

    def forward(self, xs, ys, add_inputs_embeds=False):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """

        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]

        f_output = self._backbone(
            inputs_embeds=embeds, position_ids=None, rm_pos_embd=False, add_inputs_embeds=add_inputs_embeds)  # [B, 2n, d]
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]
        if self._pred_type == 'regression':
            y = prediction[:, self.ind::self.freq, 0]
        elif self._pred_type == 'classification':
            y = prediction[:, self.ind::self.freq]
        else:
            raise NotImplementedError

        return y



class TransformerModelLooped_AB(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=2, n_head=4, loop_func='z=f(x+z)', pred_type='regression'):

        super(TransformerModelLooped_AB, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pred_type)
        self.loop_func = loop_func

    def f(self, output, embeds, which_layer):
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds, which_layer=which_layer)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds, which_layer=which_layer)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys, n_loop_start, n_loops):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        if self.loop_func in ['z=f(x+z)']:
            output = torch.zeros_like(embeds)  # also of shape [B, 2n, d]
        elif self.loop_func in ['z=f(x*z)']:
            output = torch.ones_like(embeds)  # also of shape [B, 2n, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")

        pred_list = []
        which_layer = 'A'
        for idx in range(n_loops):
            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    output = self.f(output, embeds, which_layer)
            else:
                output = self.f(output, embeds, which_layer)
                prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
                if self._pred_type == 'regression':
                    y = prediction[:, self.ind::self.freq, 0]
                elif self._pred_type == 'classification':
                    y = prediction[:, self.ind::self.freq]
                else:
                    raise NotImplementedError
                pred_list.append(y)
            if not self.print_flag:
                print(idx)
                self.print_flag = True
            if which_layer == 'A':
                which_layer = 'B'
            else:
                which_layer = 'A'
        if which_layer == 'B':
            output = self.f(output, embeds, which_layer)
            prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
            if self._pred_type == 'regression':
              y = prediction[:, self.ind::self.freq, 0]
            elif self._pred_type == 'classification':
              y = prediction[:, self.ind::self.freq]
            else:
              raise NotImplementedError
            pred_list.append(y)
        return pred_list
