from utility.word import CFG
import torch
from torch import optim
from data import *
from model import *
from train_data import *
from training import *

def taggcn_comp(args):
    data = TGCN_load(args)
    model = TagGCN(data).to(CFG['device'])
    train_data = [BPR_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test




model_dict = {
    'taggcn': taggcn_comp,
}
