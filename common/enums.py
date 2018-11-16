from enum import Enum


class ModelDataFormat(Enum):
    NCHW = "NCHW"   # Channel first
    NHWC = "NHWC"   # Channel Last (optimal format for NVIDIA GPUs using cuDNN)
