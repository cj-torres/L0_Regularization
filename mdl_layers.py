import torch, math, copy

try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None

class RegularizedParam(torch.nn.Module):

    def __init__(self, original_parameter: torch.nn.Parameter, lam: float, weight_decay: float = 0, is_bias: bool = False,
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

        # below code guts the module of its previous parameters,
        # allowing them to be replaced by non-leaf tensors

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

    # def regularization(self):
    #     r_total = torch.Tensor([])
    #     for param in self.param_names:
    #         device = self.mask_parameters[param + "_m"].device
    #         r_total = torch.cat([r_total.to(device), self._reg_w(param).unsqueeze(dim=0)])
    #     return r_total.sum()

    def count_l0(self):
        total = torch.sum(1 - self.cdf_qz(0))
        return total

    def count_l2(self):
        return (self.sample_weights(False) ** 2).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = torch.rand(size) * (1 - 2 * self.epsilon) + self.epsilon
        return eps

    def sample_z(self, batches: int, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        size = torch.Size([batches]) + self.mask.size()
        if sample:
            device = self.mask.device
            eps = self.get_eps(size).to(device)
            z = self.quantile_concrete(eps)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.mask)
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self, batches: int, sample=True):
        mask = self.sample_z(batches, sample)
        return mask * self.param

    def set_lam(self, lam: int):
        self.lam = lam


class RegLSTM(torch.nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int, lam: float, num_layers: int = 1,
                 bidirectional: bool = False):
        super(RegLSTM, self).__init__()
        gate_sz = hidden_sz*4
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_sz
        self.input_size = input_sz
        self.lam = lam
        num_directions = 2 if bidirectional else 1

        self._weights_names = []
        #self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                w_ih = RegularizedParam(torch.nn.Parameter(torch.empty((gate_sz, input_sz))), lam)
                w_hh = RegularizedParam(torch.nn.Parameter(torch.empty((gate_sz, hidden_sz))), lam)
                b_ih = RegularizedParam(torch.nn.Parameter(torch.empty(gate_sz)), lam, is_bias=True)
                b_hh = RegularizedParam(torch.nn.Parameter(torch.empty(gate_sz)), lam, is_bias=True)

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
            if is_batched:
                if (hx[0].dim() != 3 or hx[1].dim() != 3):
                    msg = ("For batched 3-D input, hx and cx should "
                           f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    msg = ("For unbatched 2-D input, hx and cx should "
                           f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors")
                    raise RuntimeError(msg)
                hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))

            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.

        samples = zip(*[w.sample_weights(batch_sz, self.training) for w in self._weights])

        outputs = []
        hiddens = []
        for x, weights in zip(seq, samples):
            print(weights)
            print(x)
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
        hidden = (torch.cat([h[0] for h in hiddens], dim=batch_dim),
                  torch.cat([h[1] for h in hiddens], dim=batch_dim))

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = (hidden[0].squeeze(batch_dim), hiddens[1].squeeze(batch_dim))

        return output, hidden
