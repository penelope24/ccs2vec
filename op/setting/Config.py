import torch


class Config:
    """
    统一管理训练所需的参数以及超参数，方便调参与维护
    基本参数：
        random walk 参数
    训练超参数：
        epoch
        batch_size
        学习率
    训练设备:
        device
    """

    def __init__(self):
        self.MAX_WALK_TIMES = None
        self.MAX_WALK_LENGTH = None
        self.EPOCHS = None
        self.BATCH_SIZE = None
        self.device = None
        self.set_random_walk_params()
        self.set_training_params()
        self.set_device_params()

    def set_random_walk_params(self, max_walk_times=48, max_walk_length=11):
        self.MAX_WALK_TIMES = max_walk_times
        self.MAX_WALK_LENGTH = max_walk_length

    def set_training_params(self, epochs=10, batch_size=64):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def set_device_params(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device