import torch


def reconstruct_image(patches: torch.Tensor, original: torch.Tensor, num_rows: int, patch_size: int, stride: int,
                      batch=1, channels=1):
    _, _, width, height = original.shape

    x_steps = [x + (stride // 2) for x in range(0, width, stride)]
    x_steps[0], x_steps[-1] = 0, width
    y_steps = [y + (stride // 2) for y in range(0, height, stride)]
    y_steps[0], y_steps[-1] = 0, height

    patches = patches.view(batch, channels, -1, num_rows, patch_size, patch_size)
    canvas = torch.zeros_like(original)
    for j in range(len(x_steps) - 1):
        for i in range(len(y_steps) - 1):
            patch = patches[0, :, j, i, :, :]
            x1_abs, x2_abs = x_steps[j], x_steps[j + 1]
            y1_abs, y2_abs = y_steps[i], y_steps[i + 1]
            x1_rel, x2_rel = x1_abs - (j * stride), x2_abs - (j * stride)
            y1_rel, y2_rel = y1_abs - (i * stride), y2_abs - (i * stride)
            canvas[0, :, x1_abs:x2_abs, y1_abs:y2_abs] = patch[:, x1_rel:x2_rel, y1_rel:y2_rel]

    return canvas
