class TransformerModelLooped_N_LastTokens(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, loop_func='z=f(x+z)', pred_type='regression'):

        super(TransformerModelLooped_N_LastTokens, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, pred_type)
        self.loop_func = loop_func

    def f(self, output, embeds):
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds= self.n_last_tokens(output, n) + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds= self.n_last_tokens(output, n) * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output
    ############################################ new function
    def n_last_tokens(self, output, n):
        if self.loop_func == 'z=f(x+z)':
            new_output = output[:, :n, :] 
            zeros = torch.zeros_like(output[:, n:, :])
            return torch.cat((new_output, zeros), 1)
        elif self.loop_func == 'z=f(x*z)':
            new_output = output[:, :n, :] 
            zeros = torch.ones_like(output[:, n:, :])
            return torch.cat((new_output, zeros), 1)
        else:
            raise NotImplementedError

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
        for idx in range(n_loops):
            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    output = self.f(output, embeds)
            else:
                output = self.f(output, embeds)
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

        return pred_list
