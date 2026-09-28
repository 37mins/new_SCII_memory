"""Read HF generation outputs; do not alter sampling or PPO probability tensors."""
import torch


def selected_log_probs(output, responses, response_mask, temperature):
    # HF logits precede temperature/top-p; scores include all sampling processors.
    # Reduce one step at a time instead of stacking the full vocabulary history.
    unwarped = torch.zeros_like(responses, dtype=torch.float32)
    sampling = torch.zeros_like(unwarped)
    for index, (raw, processed) in enumerate(zip(output.logits, output.scores, strict=True)):
        labels = responses[:, index:index + 1]
        unwarped[:, index] = torch.log_softmax(raw.float() / temperature, -1).gather(1, labels).squeeze(1)
        sampling[:, index] = torch.log_softmax(processed.float(), -1).gather(1, labels).squeeze(1)
    active = response_mask.bool()
    return torch.where(active, unwarped, 0.), torch.where(active, sampling, 0.)
