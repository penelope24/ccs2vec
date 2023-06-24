import os
from data.dataset import CCSDataset
from torch.utils.data import random_split
from data.batch import *
from model.transformer.Model import Model
from run.RunEpoch import RunEpoch
from evaluating.run.deprecated import eval3
from data.vocab import Vocab, special_tokens, min_freq_value
from op.model.utils import save_checkpoint, save_dict_to_json, metrics_dict_msg
from training.loss.wrapper.KLDivLossWithLabelSmoothing import KLDivLossWithLabelSmoothing
from training.loss.compute.SimpleLossCompute import SimpleLossCompute
from training.opt.OptimWrapper import OptimWrapper

# path
path = "/zfy/samples"

# preliminary
MAX_WALK_TIMES = 48
EPOCHS = 60
BATCH_SIZE = 64
DATA_BASE = "/home/qwe/disk1/data_SoC/files/"
SAVE_DATA_BASE = "/home/qwe/zfy_lab/fytorch/output/dataset/"
SAVE_MODEL_BASE = "/home/qwe/zfy_lab/fytorch/output/trained/"
USE_TRIM = False
USE_MIN_FREQ = True

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("current device is :", device)

dataset = CCSDataset(path)
vocab = Vocab.create_from_dataset(dataset, special_tokens, min_freq_value)
stoi = vocab.word2idx

# 定义分割比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 计算划分后的样本数
total_samples = len(dataset)
train_samples = int(train_ratio * total_samples)
val_samples = int(val_ratio * total_samples)
test_samples = total_samples - train_samples - val_samples

# 使用random_split函数进行数据集的分割
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])

# 创建DataLoader对象，用于加载数据
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# wrapper
train_wrapper = BasicBatchWrapper(train_dataloader)
test_wrapper = BasicBatchWrapper(test_dataloader)
val_wrapper = BasicBatchWrapper(val_dataloader)

"""model"""
model = Model.make_model(len(vocab.word2idx), len(vocab.word2idx), N=6)
model.cuda()
criterion = KLDivLossWithLabelSmoothing(len(vocab), padding_idx=stoi["<pad>"], smoothing=0.1)
criterion.cuda()
opt = OptimWrapper.get_std_opt(model)
train_loss_compute = SimpleLossCompute(model.generator, criterion, opt)
val_loss_compute = SimpleLossCompute(model.generator, criterion, None)

"""run"""
best_f1 = 0.0
no_new_best_count = 0
for epoch in range(EPOCHS):
    print("EPOCH: " + str(epoch))
    model.train()
    loss = RunEpoch.run_epoch(train_wrapper, model, train_loss_compute)
    print("loss: %f" % loss)
    model.eval()
    val_metrics = eval3(model, val_wrapper, stoi["<s>"],
                        stoi["<eos>"], stoi["<pad>"], stoi["<unk>"])
    test_metrics = eval3(model, test_wrapper, stoi["<s>"],
                         stoi["<eos>"], stoi["<pad>"], stoi["<unk>"])
    print("val metric: " + metrics_dict_msg(val_metrics))
    print("test metric: " + metrics_dict_msg(test_metrics))
    val_f1 = val_metrics["f1"]

    is_best = val_f1 > best_f1
    checkpoint = {
        epoch: epoch,
        "state_dict": model.state_dict(),
        "optim_dict": train_loss_compute.opt.optimizer.state_dict()
    }
    save_checkpoint(checkpoint, is_best, SAVE_MODEL_BASE)

    if is_best:
        print("find new best f1 value")
        best_f1 = val_f1

        best_json_path = os.path.join(
            SAVE_MODEL_BASE, "metrics_val_best_weights.json")

        save_dict_to_json(val_metrics, best_json_path)
        no_new_best_count = 0
    else:
        no_new_best_count += 1

    if no_new_best_count > 5:
        print("5 epochs without new best, end training")
        break
