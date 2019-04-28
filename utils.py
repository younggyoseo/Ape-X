import random
import io
from PIL import Image

import numpy as np
import torch


def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))


def set_global_seeds(seed, use_torch=False):
    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)


def array2png(arr):
    rgb_image = Image.fromarray(arr)
    output_io = io.BytesIO()
    rgb_image.save(output_io, format="PNG")
    png_image = output_io.getvalue()
    output_io.close()
    rgb_image.close()
    return png_image


def png2array(png):
    png_io = io.BytesIO(png)
    png_image = Image.open(png_io)
    rgb_array = np.array(png_image)
    png_image.close()
    png_io.close()
    return rgb_array


def compute_loss(model, tgt_model, batch, n_steps, gamma=0.99):
    states, actions, rewards, next_states, dones, weights = batch

    q_values = model(states)
    next_q_values = model(next_states)
    tgt_next_q_values = tgt_model(next_states)

    q_a_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_actions = next_q_values.max(1)[1].unsqueeze(1)
    next_q_a_values = tgt_next_q_values.gather(1, next_actions).squeeze(1)
    expected_q_a_values = rewards + (gamma ** n_steps) * next_q_a_values * (1 - dones)

    td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
    prios = (td_error + 1e-6).data.cpu().numpy()

    loss = torch.where(td_error < 1, 0.5 * td_error ** 2, td_error - 0.5)
    loss = (loss * weights).mean()
    return loss, prios


def update_parameters(loss, model, optimizer, max_norm):
    """
    Update parameters with loss
    """
    optimizer.zero_grad()
    loss.backward()
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** (1. / 2)
    total_norm = total_norm ** (1. / 2)
    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    return total_norm
