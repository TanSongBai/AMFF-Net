import torch
import numpy as np
from model import MTD_IQA_modify
import random
from tools import set_dataset4, _preprocess2, _preprocess3, convert_models_to_fp32, compute_metric
import os

##############################general setup####################################
AGIQA3K_set = r'/public/tansongbai/dataset/AGIQA-3K'
AIGCIQA2023_set = r'/public/tansongbai/dataset/AIGCIQA2023'
PKUI2IQA_set = r'/public/tansongbai/dataset/I2IQA'

seed = 2222

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################### hyperparameter #####################
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
datasets = ["PKUI2IQA"] #choose AGIQA3K | AIGCIQA2023 | PKUI2IQA
radius = [336, 224, 112]
initial_lr1 = 5e-4
initial_lr2 = 5e-6
weight_decay = 0.001
num_epoch = 100
bs = 32
early_stop = 0
clip_net = 'RN50'
in_size = 1024
istrain = True
goal = 'avg'

##############################general setup####################################

preprocess2 = [_preprocess2(radius[0]), _preprocess2(radius[1]), _preprocess2(radius[2])]
preprocess3 = [_preprocess3(radius[0]), _preprocess3(radius[1]), _preprocess3(radius[2])]
loss_fn = torch.nn.MSELoss().to(device)


def freeze_model(f_model, opt):
    f_model.logit_scale.requires_grad = False
    if opt == 0: #do nothing
        return
    elif opt == 1: # freeze text encoder
        for p in f_model.token_embedding.parameters():
            p.requires_grad = False
        for p in f_model.transformer.parameters():
            p.requires_grad = False
        f_model.positional_embedding.requires_grad = False
        f_model.text_projection.requires_grad = False
        for p in f_model.ln_final.parameters():
            p.requires_grad = False
    elif opt == 2: # freeze visual encoder
        for p in f_model.visual.parameters():
            p.requires_grad = False
    elif opt == 3:
        for p in f_model.parameters():
            p.requires_grad = False
    elif opt == 4:
        for p in f_model.parameters():
            p.requires_grad = True


def do_batch(x_l, x_m, x_s, con_text):

    input_token_c = con_text.view(-1, 77)
    logits_per_qua, logits_per_con, logits_per_aes = model.forward(x_l, x_m, x_s, input_token_c)

    return logits_per_qua, logits_per_con, logits_per_aes


def eval(loader):
    model.eval()
    y_q = []
    y_pred_q = []
    y_a = []
    y_pred_a = []
    y_c = []
    y_pred_c = []
    for step, sample_batched in enumerate(loader):
        x_l, x_m, x_s, mos_q, mos_a, mos_c, con_tokens = sample_batched['img_l'], sample_batched['img_m'], \
            sample_batched['img_s'], sample_batched['mos_q'], \
            sample_batched['mos_a'], sample_batched['mos_c'], \
            sample_batched['con_tokens']

        img_name = sample_batched['img_name']
        x_l = x_l.to(torch.float32).to(device)
        x_m = x_m.to(torch.float32).to(device)
        x_s = x_s.to(torch.float32).to(device)

        mos_q = mos_q.to(torch.float32).to(device)
        mos_a = mos_a.to(torch.float32).to(device)
        mos_c = mos_c.to(torch.float32).to(device)
        con_tokens = con_tokens.to(device)

        with torch.no_grad():
            logits_per_qua, logits_per_con, logits_per_aes = do_batch(x_l, x_m, x_s, con_tokens)

            weight_qua = logits_per_qua[:, 0]
            weight_aes = logits_per_aes[:, 0]
            weight_con = logits_per_con[:, 0]

            y_pred_q.extend(weight_qua.cpu().numpy())
            y_pred_a.extend(weight_aes.cpu().numpy())
            y_pred_c.extend(weight_con.cpu().numpy())
            y_q.extend(mos_q.cpu().numpy())
            y_a.extend(mos_a.cpu().numpy())
            y_c.extend(mos_c.cpu().numpy())

    _, PLCC1, SRCC1, KRCC1 = compute_metric(np.array(y_q), np.array(y_pred_q), istrain)
    if mtl != 0:
        _, PLCC2, SRCC2, KRCC2 = compute_metric(np.array(y_a), np.array(y_pred_a), istrain)
    else:
        _, PLCC2, SRCC2, KRCC2 = 0.0, 0.0, 0.0, 0.0
    _, PLCC3, SRCC3, KRCC3 = compute_metric(np.array(y_c), np.array(y_pred_c), istrain)

    out = [SRCC1, PLCC1, KRCC1,
           SRCC2, PLCC2, KRCC2,
           SRCC3, PLCC3, KRCC3]

    return out


num_workers = 8
for dataset in datasets:
    mtl_map = {'AGIQA3K': 0, 'AIGCIQA2023': 1, 'PKUI2IQA': 2}
    mtl = mtl_map[dataset]

    nss_set = {'AGIQA3K': [os.path.join('brisuqe_feature', 'AGIQA3Kname.mat'),
                           os.path.join('brisuqe_feature', 'AGIQA3Kfeat.mat')],
               'AIGCIQA2023': [os.path.join('brisuqe_feature', 'AIGCIQA2023name.mat'),
                               os.path.join('brisuqe_feature', 'AIGCIQA2023feat.mat')],
               'PKUI2IQA': [os.path.join('brisuqe_feature', 'PKUI2IQAname.mat'),
                            os.path.join('brisuqe_feature', 'PKUI2IQAfeat.mat')]}

    print('test on ', dataset)

    for session in range(0, 10):
        model = MTD_IQA_modify.MTD_IQA(device=device, clip_net=clip_net, in_size=in_size)
        model = model.to(device)
        pth = torch.load(os.path.join(f'checkpoints/{dataset}', 'MTD_IQA', str(session + 1), goal+'_best_ckpt.pt'))
        model.load_state_dict(pth['model_state_dict'], strict=True)
        print('success loading pth')

        early_stop = 0
        start_epoch = 0
        global_step = 0
        best_result = {'avg': 0.0, 'quality': 0.0, 'authenticity': 0.0, 'correspondence': 0.0}
        best_epoch = {'avg': 0, 'quality': 0, 'authenticity': 0, 'correspondence': 0}

        AGIQA3K_train_txt = os.path.join('./IQA_Database/AGIQA-3K', str(session+1), 'train.txt')
        AGIQA3K_test_txt = os.path.join('./IQA_Database/AGIQA-3K', str(session + 1), 'test.txt')

        AIGCIQA2023_train_txt = os.path.join('./IQA_Database/AIGCIQA2023', str(session + 1), 'train.txt')
        AIGCIQA2023_test_txt = os.path.join('./IQA_Database/AIGCIQA2023', str(session + 1), 'test.txt')

        PKUI2IQA_train_txt = os.path.join('./IQA_Database/PKU-I2IQA', str(session + 1), 'train.txt')
        PKUI2IQA_test_txt = os.path.join('./IQA_Database/PKU-I2IQA', str(session + 1), 'test.txt')

        AGIQA3K_train_loader = set_dataset4(AGIQA3K_train_txt, bs, AGIQA3K_set, radius, num_workers, preprocess3, 0, False)
        AGIQA3K_test_loader = set_dataset4(AGIQA3K_test_txt, bs, AGIQA3K_set, radius, num_workers, preprocess2, 0, True)

        AIGCIQA2023_train_loader = set_dataset4(AIGCIQA2023_train_txt, bs, AIGCIQA2023_set, radius, num_workers, preprocess3, 1, False)
        AIGCIQA2023_test_loader = set_dataset4(AIGCIQA2023_test_txt, bs, AIGCIQA2023_set, radius, num_workers, preprocess2, 1, True)

        PKUI2IQA_train_loader = set_dataset4(PKUI2IQA_train_txt, bs, PKUI2IQA_set, radius, num_workers, preprocess3, 2, False)
        PKUI2IQA_test_loader = set_dataset4(PKUI2IQA_test_txt, bs, PKUI2IQA_set, radius, num_workers, preprocess2, 2, True)

        train_loders_dir = {'AGIQA3K': AGIQA3K_train_loader, 'AIGCIQA2023': AIGCIQA2023_train_loader, 'PKUI2IQA': PKUI2IQA_train_loader}
        test_loaders_dir = {'AGIQA3K': AGIQA3K_test_loader, 'AIGCIQA2023': AIGCIQA2023_test_loader, 'PKUI2IQA': PKUI2IQA_test_loader}
        train_loaders, test_loaders = train_loders_dir[dataset], test_loaders_dir[dataset]

        out = eval(test_loaders)

        print(goal + ' best\n', str(session + 1))

        if goal == 'avg':
            print(out[0:3])
            print(out[3:6])
            print(out[6:9])
