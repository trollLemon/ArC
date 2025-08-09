num_classes = 256
sequence_length = 64

def get_basic_model_config():
    return {
    "image_size": 256,
    "patch_size": 32,
    "dim": 768,
    "depth": 4,
    "heads": 4,
    "mlp_dim": 768
}
