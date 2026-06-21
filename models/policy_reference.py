from models.load_model import load_model, load_reference_model


def load_policy_model(model_name_or_path=None):
    return load_model(model_name_or_path)


def load_frozen_reference_model(model_name_or_path=None):
    reference_model = load_reference_model(model_name_or_path)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model
