import torch, math, copy

class L0_Regularizer(torch.nn.Module):

    def __init__(self, original_module: torch.nn.Module, lam: float, weight_decay: float = 0,
                 temperature: float = 2 / 3, droprate_init=0.2, limit_a=-.1, limit_b=1.1, epsilon=1e-6
                 ):
        super(L0_Regularizer, self).__init__()
        self.module = copy.deepcopy(original_module)

        self.pre_parameters = torch.nn.ParameterDict(
            {name.replace(".", "#") + "_p": param for name, param in self.module.named_parameters()}
        )

        self.param_names = [name.replace(".", "#") for name, param in self.module.named_parameters()]
        self.mask_parameters = torch.nn.ParameterDict()
        self.lam = lam
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        for name, param in self.module.named_parameters():
            mask = torch.nn.Parameter(torch.Tensor(param.size()))
            self.mask_parameters.update({name.replace(".", "#") + "_m": mask})

        # below code guts the module of its previous parameters,
        # allowing them to be replaced by non-leaf tensors

        self.to("cpu")

        for name in self.param_names:
            L0_Regularizer.recursive_del(self.module, name)
            L0_Regularizer.recursive_set(self.module, name, self.sample_weights(name, 1))

    ''' 
    Below code direct copy with adaptations from codebase for: 

    Louizos, C., Welling, M., & Kingma, D. P. (2017). 
    Learning sparse neural networks through L_0 regularization. 
    arXiv preprint arXiv:1712.01312.
    '''

    def reset_parameters(self):
        for name, weight in self.pre_parameters.items():
            if "bias" in name:
                torch.nn.init.constant_(weight, 0.0)
            elif self.param.data.ndimension() >= 2:
                torch.nn.init.xavier_uniform_(self.param)
            else:
                torch.nn.init.uniform_(self.param, 0, 1)

        for name, weight in self.mask_parameters.items():
            torch.nn.init.normal_(weight, math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

    def constrain_parameters(self):
        for name, weight in self.mask_parameters.items():
            weight.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x, param):
        """Implements the CDF of the 'stretched' concrete distribution"""
        # references parameters
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(
            logits * self.temperature - self.mask_parameters[param + "_m"]).clamp(min=self.epsilon,
                                                                                  max=1 - self.epsilon)

    def quantile_concrete(self, x, param):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        # references parameters
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.mask_parameters[param + "_m"]) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def _reg_w(self, param):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        """is_neural is old method, calculates wrt columns first multiplied by expected values of gates
           second method calculates wrt all parameters
        """

        # why is this negative? will investigate behavior at testing
        # reversed negative value, value should increase with description length
        logpw_l2 = - (.5 * self.weight_decay * self.pre_parameters[param + "_p"].pow(2)) - self.lam
        logpw = torch.sum((1 - self.cdf_qz(0, param)) * logpw_l2)

        return -logpw

    def regularization(self):
        r_total = torch.Tensor([])
        for param in self.param_names:
            device = self.mask_parameters[param + "_m"].device
            r_total = torch.cat([r_total.to(device), self._reg_w(param).unsqueeze(dim=0)])
        return r_total.sum()

    def count_l0(self):
        total = []
        for param in self.param_names:
            total.append(torch.sum(1 - self.cdf_qz(0, param)).unsqueeze(dim=0))
        return torch.cat(total).sum()

    def count_l2(self):
        total = []
        for param in self.param_names:
            total.append(self._l2_helper(param).unsqueeze(dim=0))
        return torch.cat(total).sum()

    def _l2_helper(self, param):
        return (self.sample_weights(param, False) ** 2).sum()

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        # Variable deprecated and removed
        eps = torch.rand(size) * (1 - 2 * self.epsilon) + self.epsilon
        return eps

    def sample_z(self, param, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        size = self.mask_parameters[param + "_m"].size()
        if sample:
            device = self.mask_parameters[param + "_m"].device
            eps = self.get_eps(size).to(device)
            z = self.quantile_concrete(eps, param)
            return torch.nn.functional.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.mask_parameters[param + "_m"])
            return torch.nn.functional.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def sample_weights(self, param, sample=True):
        mask = self.sample_z(param, sample)
        return mask * self.pre_parameters[param + "_p"]

    def forward(self, x):
        """rewrite parameters (tensors) of core module and feedforward"""
        for param in self.param_names:
            L0_Regularizer.recursive_set(self.module, param, self.sample_weights(param, sample=self.training))

        return self.module(x)

    @staticmethod
    def recursive_get(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_get(getattr(obj, first), last)
        else:
            return getattr(obj, att_name)

    @staticmethod
    def recursive_set(obj, att_name, val):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_set(getattr(obj, first), last, val)
        else:
            setattr(obj, att_name, val)

    @staticmethod
    def recursive_del(obj, att_name):
        if "#" in att_name:
            first, last = att_name.split("#", 1)
            L0_Regularizer.recursive_del(getattr(obj, first), last)
        else:
            delattr(obj, att_name)
