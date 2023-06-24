import torch


class ModelOp:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def compare_models(model1, model2):
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            print((param1 == param2).all())

    @staticmethod
    def print_model_state_dict(model):
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    @staticmethod
    def print_optimier_state_dict(optimizer):
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])