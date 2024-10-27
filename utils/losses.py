import torch


def balance_loss_switch_transformer(routing_weights_all_layers, k=2, alpha=1e-2):
    '''
    :param routing_topk_weights: tuple[(batch_size x num_tokens, k)] with len number of hidden_layers
    :param routing_weights: (batch_size x num_tokens, num_experts)
    :param alpha:
    :return:
    '''
    routing_weights = torch.stack([rw for rw in routing_weights_all_layers], dim=0)
    _, selected_experts = torch.topk(routing_weights, k, dim=-1)
    num_layers, num_tokens, num_experts = routing_weights.size()
    tokens_dispatch = torch.zeros((num_layers, num_experts), device=routing_weights.device)
    # fi
    for i in range(num_experts):
        tokens_dispatch[:, i] = torch.sum(selected_experts == i, dim=[-2, -1])
    tokens_dispatch /= (num_tokens * k)
    # Pi
    experts_probs = torch.sum(routing_weights, dim=1) / num_tokens
    # mean(alpha * N * sum(fi * Pi), after num_layers)
    loss = torch.mean(alpha * num_experts * torch.sum(tokens_dispatch * experts_probs, dim=-1))
    return loss


def mlm_loss(output, labels, criterion):
    output = output.view(-1, output.size(-1))
    labels = labels.view(-1)
    return criterion(output, labels)


def total_loss(output, labels, criterion, routing_logits, args):
    mlm_loss_value = mlm_loss(output, labels, criterion)
    if args.balance_loss:
        balancing_loss = balance_loss_switch_transformer(routing_logits, k=2, alpha=args.alpha)
    else:
        balancing_loss = torch.tensor(0.0, device=output.device)
    return mlm_loss_value + balancing_loss, mlm_loss_value, balancing_loss

# if __name__ == '__main__':
#     routing_weights_all_layers = tuple([torch.randn((2*128,8), requires_grad=True).softmax(dim=-1) for _ in range(6)])
#     k = 2
#     alpha = 1e-2
#     balance_loss_switch_transformer(routing_weights_all_layers, k, alpha)
