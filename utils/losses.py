import torch


def balance_loss_switch_transformer(routing_weights, k=2, alpha=1e-2):
    '''
    :param routing_topk_weights: tuple[(batch_size x num_tokens, k)] with len number of hidden_layers
    :param routing_weights: (batch_size x num_tokens, num_experts)
    :param alpha:
    :return:
    '''
    _, selected_experts = torch.topk(routing_weights, k, dim=-1)
    num_tokens, num_experts = routing_weights.size()
    tokens_dispatch = torch.zeros(num_experts, device=routing_weights.device)
    # fi
    for i in range(num_experts):
        tokens_dispatch[i] = torch.sum(selected_experts == i)
    tokens_dispatch /= (num_tokens)
    # Pi
    experts_probs = torch.zeros(num_experts, device=routing_weights.device)
    for i in range(num_experts):
        experts_probs[i] = torch.sum(routing_weights[:, i])
    experts_probs /= (num_tokens)
    # alpha * N * sum(fi * Pi)
    loss = alpha * num_experts * torch.sum(tokens_dispatch * experts_probs, dim=-1)
    return loss


def mlm_loss(output, labels, criterion, mask_labels):
    output = output.view(-1, output.size(-1))
    labels = labels.view(-1)
    loss = criterion(output, labels)
    masked_lm_loss = (loss * mask_labels.view(-1)).sum() / mask_labels.sum()

    return masked_lm_loss


def total_loss(output, labels, criterion, routing_logits, mask_labels, args):
    mlm_loss_value = mlm_loss(output, labels, criterion, mask_labels)
    if args.balance_loss:
        balancing_loss = torch.mean(torch.stack(
            [balance_loss_switch_transformer(rl, k=args.num_experts_per_token, alpha=args.alpha)
             for rl in routing_logits], dim=0))
    else:
        balancing_loss = torch.tensor(0.0, device=output.device)
    return mlm_loss_value + balancing_loss, mlm_loss_value, balancing_loss


# if __name__ == '__main__':
#     routing_weights_all_layers = torch.randn((2*128, 8))
#     # routing_weights_all_layers[:, 1] -=100
#     routing_weights_all_layers = routing_weights_all_layers.softmax(dim=-1)
#     k = 1
#     alpha = 1e-2
#     l = balance_loss_switch_transformer(routing_weights_all_layers, k, alpha)
#     print(l)
