import torch, math, copy
import einops

try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None


def variable_embedding(x: torch.LongTensor, embeds: torch.Tensor):
    num_embeds = embeds.size()[-2]
    embed_dim = embeds.size()[-1]
    max_x = torch.max(x).item()
    min_x = torch.min(x).item()
    assert(max_x < num_embeds and min_x >= 0)
    x = einops.rearrange(x, '... -> ... ()')
    x = einops.repeat(x, '... -> ... embeds', embeds=embed_dim)
    return torch.gather(embeds, -2, x).squeeze()


def ve_unit_test():
    def iter_tensor_loop(x: torch.Tensor):
        if x.dim() == 1:
            return [((i,), x_i) for i, x_i in enumerate(x)]
        else:
            return [((j,)+ i, x_i) for j, x_j in enumerate(x) for i, x_i in iter_tensor_loop(x_j)]

    def slow_embedding(x: torch.LongTensor, embeds: torch.Tensor):
        batches = x.size()
        emb_size = (embeds.size()[-1],)
        final_tensor = torch.empty(batches+emb_size)
        for final_tensor_indx, embedding_indx in iter_tensor_loop(x):
            final_tensor[final_tensor_indx] = embeds[final_tensor_indx][embedding_indx]
        return final_tensor

    # dim = 1
    y = torch.LongTensor(torch.randint(10, (1000,)))
    embeds = torch.randn(1000, 10, 50)
    assert(torch.equal(variable_embedding(y, embeds), slow_embedding(y, embeds)))

    # dim = 2
    y = torch.LongTensor(torch.randint(10, (1000, 200)))
    embeds = torch.randn(1000, 200, 10, 50)
    assert(torch.equal(variable_embedding(y, embeds), slow_embedding(y, embeds)))

    # dim = 3
    y = torch.LongTensor(torch.randint(10, (1000, 200, 30)))
    embeds = torch.randn(1000, 200, 30, 10, 50)
    assert(torch.equal(variable_embedding(y, embeds), slow_embedding(y, embeds)))


class RegularizedParam(torch.nn.Module):

    def __init__(self, original_parameter: torch.Tensor, lam: float, weight_decay: float = 0, is_bias: bool = False,
                 temperature: float = 2 / 3, droprate_init=0.2, limit_a=-.1, limit_b=1.1, epsilon=1e-6
                 ):
        super(RegularizedParam, self).__init__()
        self.param = copy.deepcopy(original_parameter)
        self.mask = torch.nn.Parameter(torch.Tensor(self.param.size()))

        self.is_bias = is_bias
        self.lam = lam
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        self.constrain_parameters()
        torch.nn.init.normal_(self.mask, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        self.to("cpu")
    ''' 
    Below code direct copy with adaptations from codebase for: 

    Louizos, C., Welling, M., & Kingma, D. P. (2017). 
    Learning sparse neural networks through L_0 regularization. 
    arXiv preprint arXiv:1712.01312.
    '''

    def constrain_parameters(self):
        self.param.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def reset_parameters(self):
        if self.is_bias:
            torch.nn.init.constant_(self.param, 0.0)
        elif self.param.data.ndimension() >= 2:
            torch.nn.init.xavier_uniform_(self.param)
        else:
            torch.nn.init.uniform_(self.param, 0, 1)

        torch.nn.init.normal_(self.mask, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # references parameters
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.mask).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        # references parameters
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.mask) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def regularization(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        """is_neural is old method, calculates wrt columns first multiplied by expected values of gates
           second method calculates wrt all parameters
        """

        # why is this negative? will investigate behavior at testing
        # reversed negative value, value should increase with description length
        logpw_l2 = - (.5 * self.weight_decay * self.param.pow(2)) - self.lam
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_l2)

        return -logpw

    def count_l0(self):
        total = torch.sum(1 - self.cdf_qz(0))
        return total

    def count_l2(self):
        return (self.sample_weights(1, False) ** 2).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = torch.rand(size) * (1 - 2 * self.epsilon) + self.epsilon
        return eps

    def sample_z(self, size: torch.Size(), sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        size = size + self.mask.size()
        if sample:
            device = self.mask.device
            eps = self.get_eps(size).to(device)
            z = self.quantile_concrete(eps)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.mask)
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self, size: torch.Size, sample=True):
        mask = self.sample_z(size, sample)
        return mask * self.param

    def set_lam(self, lam: float):
        self.lam = lam

    def set_wd(self, wd: float):
        self.weight_decay = wd


class RegLSTM(torch.nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int, lam: float, num_layers: int = 1,
                 bidirectional: bool = False, weight_decay: float = 0):
        super(RegLSTM, self).__init__()
        gate_sz = hidden_sz*4
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_sz
        self.input_size = input_sz
        self.lam = lam
        self.wd = weight_decay
        num_directions = 2 if bidirectional else 1

        self._weights_names = []
        #self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                w_ih = RegularizedParam(torch.nn.Parameter(torch.empty((gate_sz, input_sz))), lam, weight_decay=self.wd)
                w_hh = RegularizedParam(torch.nn.Parameter(torch.empty((gate_sz, hidden_sz))), lam, weight_decay=self.wd)
                b_ih = RegularizedParam(torch.nn.Parameter(torch.empty(gate_sz)), lam, is_bias=True, weight_decay=self.wd)
                b_hh = RegularizedParam(torch.nn.Parameter(torch.empty(gate_sz)), lam, is_bias=True, weight_decay=self.wd)

                layer_params = (w_ih, w_hh, b_ih, b_hh)
                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._weights_names.extend(param_names)
                #self._all_weights.append(param_names)

        self._weights = [getattr(self, wn) if hasattr(self, wn) else None
                              for wn in self._weights_names]
        self.reset_parameters()

    def reset_parameters(self):
        for reg_param in self._weights:
            reg_param.reset_parameters()

    def set_lam(self, lam: float):
        self.lam = lam
        for w in self._weights:
            w.set_lam(lam)

    def set_wd(self, wd):
        for w in self._weights:
            w.set_wd(wd)

    def regularization(self):
        return torch.stack([w.regularization() for w in self._weights]).sum()

    def count_l0(self):
        return torch.stack([w.count_l0() for w in self._weights]).sum()

    def count_l2(self):
        return torch.stack([w.count_l2() for w in self._weights]).sum()

    def forward(self, seq, hx=None):
        assert (seq.dim() in (2, 3)), f"LSTM: Expected input to be 2-D or 3-D but received {seq.dim()}-D tensor"
        #breakpoint()
        is_batched = seq.dim() == 3
        batch_dim = 0
        if is_batched:
            batch_sz = seq.size()[batch_dim]
        else:
            batch_sz = 1
        if not is_batched:
            seq = seq.unsqueeze(batch_dim)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  1, self.hidden_size,
                                  dtype=seq.dtype, device=seq.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  1, self.hidden_size,
                                  dtype=seq.dtype, device=seq.device)
            hx = (h_zeros, c_zeros)
            # if is_batched:
            #     if (hx[0].dim() != 3 or hx[1].dim() != 3):
            #         msg = ("For batched 3-D input, hx and cx should "
            #                f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
            #         raise RuntimeError(msg)
            # else:
            #     if hx[0].dim() != 2 or hx[1].dim() != 2:
            #         msg = ("For unbatched 2-D input, hx and cx should "
            #                f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
            #         raise RuntimeError(msg)
            #     hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.

        samples = zip(*[w.sample_weights(torch.Size([batch_sz]), self.training) for w in self._weights])

        outputs = []
        hiddens = []
        for x, weights in zip(seq, samples):
            if x.is_cuda:
                weights = torch._cudnn_rnn_flatten_weight(
                    weights, 4,
                    self.input_size, _cudnn.RNNMode.lstm,
                    self.hidden_size, 0, self.num_layers,
                    True, bool(self.bidirectional))
            result = torch._VF.lstm(x.unsqueeze(dim=batch_dim), hx, weights, True, self.num_layers,
                                    0, self.training, self.bidirectional, True)
            outputs.append(result[0])
            hiddens.append(result[1:])
        output = torch.cat(outputs, dim=batch_dim)
        hidden = (torch.stack([h[0].squeeze(1) for h in hiddens], dim=batch_dim),
                  torch.stack([h[1].squeeze(1) for h in hiddens], dim=batch_dim))

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = (hidden[0], hidden[1])

        return output, hidden


class RegEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, lam: float, weight_decay: float = 0):
        super(RegEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.lam = lam
        self.wd = weight_decay

        self.weight = RegularizedParam(torch.empty(num_embeddings, embedding_dim), lam, weight_decay=self.wd)

    def reset_parameters(self):
        self.weight.reset_parameters()

    def set_lam(self, lam: float):
        self.lam = lam
        self.weight.set_lam(lam)

    def set_wd(self, wd):
        self.wd = wd
        self.weight.set_wd(wd)

    def regularization(self):
        return self.weight.regularization()

    def count_l0(self):
        return self.weight.count_l0()

    def count_l2(self):
        return self.weigt.count_l2()

    def forward(self, x: torch.LongTensor, batch_ends: int = None):
        '''
        :param x: input of LongTensor which will be mapped to embeddings
        :param batch_ends: final batch dimension, typical behavior will be zero, assumption is batch-first input
        :return: Tensor of Size(x) x embedding_dimension
        '''
        if batch_ends:
            assert(isinstance(batch_ends, int)), "sample_beginning must be of type int"
            assert (0 <= batch_ends <= len(x.size())), \
                "sample_beginning must be >= 0 and <= dimension of input tensor"
            if self.training:
                sampled_dims = x.size()[:batch_ends+1]
                num_non_batch_dimensions = len(x.size()[batch_ends+1:])
                unsampled_dims = torch.Size([1 for _ in range(num_non_batch_dimensions)])
                new_size = sampled_dims + unsampled_dims
                embedding_samples = self.weight.sample_weights(x.size()[:batch_ends+1], self.training)
                embedding_samples.expand(new_size + torch.Size(embedding_samples)[-2:])
                end_size = x.size() + embedding_samples.size()[len(x.size()):]
                embedding_samples = embedding_samples.expand(end_size)
                return variable_embedding(x, embedding_samples)
            else:
                embedding = self.weight.sample_weights(torch.Size((1,)), self.training).squeeze(dim=0)
                return torch.nn.functional.embedding(x, embedding)
        else:
            if self.training:
                embedding_samples = self.weight.sample_weights(x.size(), self.training)
                return variable_embedding(x, embedding_samples)
            else:
                embedding = self.weight.sample_weights(torch.Size((1,)), self.training).squeeze(dim=0)
                return torch.nn.functional.embedding(x, embedding)


class RegLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, lam: float, weight_decay: float = 0):
        super(RegLinear, self).__init__()
        self.weight = RegularizedParam(torch.empty(input_size, output_size), lam, weight_decay=weight_decay)
        self.bias = RegularizedParam(torch.Tensor(output_size), lam, is_bias=True, weight_decay=weight_decay)

    def reset_parameters(self):
        self.weight.reset_parameters()
        self.bias.reset_parameters()

    def set_lam(self, lam: float):
        self.lam = lam
        self.weight.set_lam(lam)
        self.bias.set_lam(lam)

    def set_wd(self, wd):
        self.wd = wd
        self.weight.set_wd(wd)
        self.bias.set_wd(wd)

    def regularization(self):
        return self.weight.regularization() + self.bias.regularization()

    def count_l0(self):
        return self.weight.count_l0() + self.bias.count_l0()

    def count_l2(self):
        return self.weigt.count_l2() + self.bias.count_l2()

    def forward(self, x: torch.LongTensor):
        if self.training:
            weight_samples = self.weight.sample_weights(x.size()[:-1], self.training)
            bias_samples = self.bias.sample_weights(x.size()[:-1], self.training)
            return torch.einsum('...ji,...j->...i', weight_samples, x) + bias_samples
        else:
            weight = self.weight.sample_weights(torch.Size((1,)), self.training).squeeze(dim=0)
            bias = self.bias.sample_weights(torch.Size((1,)), self.training).squeeze(dim=0)
            return torch.nn.functional.linear(x, weight, bias)
