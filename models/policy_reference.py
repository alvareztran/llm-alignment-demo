from models.load_model import load_model, load_reference_model


def load_policy_model():
    return load_model()


def load_frozen_reference_model():
    reference_model = load_reference_model()
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model
