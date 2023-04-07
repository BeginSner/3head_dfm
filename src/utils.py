import re
import torch.optim as optim


def parse_float_arg(input, prefix):
    p = re.compile(prefix + "_[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    if m is None:
        return None
    input = m.group()
    p = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    return float(m.group())


def get_optimizer(name, params):
    if name == "SGD":
        if "momentum" not in params:
            params["momentum"] = 0.0
        if "nesterov" not in params:
            params["nesterov"] = False
        return optim.SGD(lr=params["lr"],
                         momentum=params["momentum"],
                         nesterov=params["nesterov"])
    elif name == "Adam":
        return optim.Adam(lr=params["lr"], betas=(0.9, 0.999))
