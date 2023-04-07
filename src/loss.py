import torch

def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "fake_negative_weighted_loss":
        return fake_negative_weighted_loss
    elif name == "delayed_feedback_loss":
        return exp_delay_loss
    elif name == "tn_dp_pretraining_loss":
        return delay_tn_dp_loss
    elif name == "fsiw_loss":
        return fsiw_loss
    elif name == "esdfm_loss":
        return delay_tn_importance_weight_loss
    elif name == "3head_dfm_loss":
        return three_head_dfm_loss
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))

def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor([0.0], device=x.device)) + torch.log(1+torch.exp(-torch.abs(x)))

def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = x.view(-1)
    z = z.to(torch.float32)
    loss = torch.mean(
        torch.nn.functional.binary_cross_entropy_with_logits(x, z)
    )
    return {"loss": loss}

def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = x.view(-1)
    z = z.to(torch.float32)
    p_no_grad = torch.sigmoid(x.detach())
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = torch.mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}

def exp_delay_loss(targets, outputs, params=None):
    z = torch.reshape(torch.tensor(targets["label"][:, 0], dtype=torch.float32, device=params['device']), (-1, 1))
    x = outputs["logits"]
    lamb = torch.nn.functional.softplus(outputs["log_lamb"])
    log_lamb = torch.log(lamb)
    d = torch.reshape(torch.tensor(targets["label"][:, 1], dtype=torch.float32, device=params['device']), (-1, 1))
    e = d
    p = torch.sigmoid(x)
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb*d)
    neg_loss = -torch.log(1 - p + p*torch.exp(-lamb*e))
    return {"loss": torch.mean(pos_loss*z + neg_loss*(1-z))}

def delay_tn_dp_loss(targets, outputs, params=None):
    tn = outputs["tn_logits"].float()
    dp = outputs["dp_logits"].float()
    z = targets["label"].float()
    tn_label = torch.reshape(z[:, 0], (-1, 1))
    dp_label = torch.reshape(z[:, 1], (-1, 1))
    pos_label = torch.reshape(z[:, 2], (-1, 1))
    tn_mask = (1-pos_label)+dp_label
    tn_loss = torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(tn, tn_label, reduction='none')*tn_mask)\
        / torch.sum(tn_mask)
    dp_loss = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(dp, dp_label))
    loss = tn_loss + dp_loss
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }

def fsiw_loss(targets, outputs, params=None):
    x = outputs["logits"]
    logits0 = outputs["logits0"].detach()
    logits1 = outputs["logits0"].detach()
    prob0 = torch.sigmoid(logits0)
    prob1 = torch.sigmoid(logits1)
    z = torch.reshape(torch.tensor(targets["label"], dtype=torch.float32, device=params['device']), (-1, 1))

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1/(prob1+1e-8)
    neg_weight = prob0

    clf_loss = torch.mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {
        "loss": loss,
    }

def delay_tn_importance_weight_loss(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = torch.reshape(torch.tensor(z, dtype=torch.float32, device=params['device']), (-1, 1))
    prob = torch.sigmoid(x).detach()
    dist_prob = torch.sigmoid(tn_logits)
    dp_prob = torch.sigmoid(dp_logits)

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = neg_weight.detach()
    pos_weight = pos_weight.detach()

    clf_loss = torch.mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}

def three_head_dfm_loss(targets, outputs, params=None):
    z = torch.reshape(torch.tensor(targets["label"][:, 0], dtype=torch.float32, device=params['device']), (-1, 1))
    x = outputs["logits"]
    y = outputs[]
    lamb = torch.nn.functional.softplus(outputs["log_lamb"])
    log_lamb = torch.log(lamb)
    d = torch.reshape(torch.tensor(targets["label"][:, 1], dtype=torch.float32, device=params['device']), (-1, 1))
    e = d
    po = torch.sigmoid(x)
    pc = torch.sigmoid(y)
    pos_loss = -(-stable_log1pex(x) -stable_log1pex(y) + log_lamb - lamb * d)
    neg_loss = -torch.log(1 - po + po*(1-pc) + po*pc*torch.exp(-lamb * e))
    return {"loss": torch.mean(pos_loss * z + neg_loss * (1 - z))}