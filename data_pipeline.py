import torch
INPUT_FORMAT = '[B, 10, 1, H, W]'
TARGET_FORMAT = '[B, 12, 1, H, W]'
VALUE_RANGE = '[0, 1]'

class PrivateDataPipelineUnavailable(RuntimeError):
    pass

def tensor_format():
    return {'input': INPUT_FORMAT, 'target': TARGET_FORMAT, 'range': VALUE_RANGE}

def validate_tensor_format(inputs: torch.Tensor, targets: torch.Tensor):
    if inputs.ndim != 5 or inputs.shape[1] != 10 or inputs.shape[2] != 1:
        raise ValueError(f'Expected input format {INPUT_FORMAT}.')
    if targets.ndim != 5 or targets.shape[1] != 12 or targets.shape[2] != 1:
        raise ValueError(f'Expected target format {TARGET_FORMAT}.')
    if inputs.shape[0] != targets.shape[0] or inputs.shape[-2:] != targets.shape[-2:]:
        raise ValueError('Input and target batch or spatial dimensions do not match.')
    if inputs.numel() and (inputs.min().item() < 0 or inputs.max().item() > 1):
        raise ValueError(f'Expected input range {VALUE_RANGE}.')
    if targets.numel() and (targets.min().item() < 0 or targets.max().item() > 1):
        raise ValueError(f'Expected target range {VALUE_RANGE}.')
    return True

def unavailable():
    raise PrivateDataPipelineUnavailable('The executable data preparation and loading implementation is not included in this protected reference release.')
