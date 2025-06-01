import torch
import torch.optim as optim
import os
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from test import test
import option
from tqdm import tqdm
torch.set_default_tensor_type('torch.FloatTensor')


def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss


class SigmoidMAELoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        from torch.nn import Sigmoid
        self.__sigmoid__ = Sigmoid()
        self.__l1_loss__ = MSELoss()

    def forward(self, pred, target):
        return self.__l1_loss__(pred, target)


class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score_abnormal = score_abnormal
        score_normal = score_normal

        score = torch.cat((score_normal, score_abnormal), 0)
        score = score.squeeze()

        label = label.cuda()

        loss_cls = self.criterion(score, label)  # BCE loss in the score space

        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))

        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)

        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)

        loss_total = loss_cls + self.alpha * loss_rtfm

        return loss_total


def train(nloader, aloader, model, batch_size, seg_num, optimizer, device):
    with torch.set_grad_enabled(True):
        model.train()

        ninput1, ninput2, ninput3, nlabel = next(nloader)
        ainput1, ainput2, ainput3, alabel = next(aloader)

        input1 = torch.cat((ninput1, ainput1), 0).to(device)
        input2 = torch.cat((ninput2, ainput2), 0).to(device)
        input3 = torch.cat((ninput3, ainput3), 0).to(device)
        score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input1, input2, input3)

        scores = scores.view(batch_size * seg_num * 2, -1) # BX32X2, 1

        scores = scores.squeeze()
        abn_scores = scores[batch_size * seg_num:]

        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)

        loss_RTFM = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        cost = loss_RTFM + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


def main():
    args = option.train_parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_models):
        os.makedirs(args.save_models)

    feature_size = args.feature_size
    model = Model(feature_size, args.batch_size, args.seg_num)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.005)
    test_info = {"epoch": [], "AUC": [], "AP": []}
    best_result = -1
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    _, overall_auc, ap = test(dataloader=test_loader,
                              model=model,
                              device=device,
                              gen_scores=False,
                              save_dir=None)

    for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(nloader=loadern_iter,
              aloader=loadera_iter,
              model=model,
              batch_size=args.batch_size,
              seg_num=args.seg_num,
              optimizer=optimizer,
              device=device)

        if step % 5 == 0 and step > 200:
            _, overall_auc, ap = test(dataloader=test_loader,
                                      model=model,
                                      device=device,
                                      gen_scores=False,
                                      save_dir=None)

            test_info["epoch"].append(step)
            test_info["AUC"].append(overall_auc)
            test_info["AP"].append(ap)

            # if test_info["AUC"][-1] > best_result:
            #     best_result = test_info["AUC"][-1]
            #     torch.save(model.state_dict(), os.path.join(args.save_models, args.model_name + '-{}.pkl'.format(step)))
            #     file_path = os.path.join(output_dir, '{}-step-AUC.txt'.format(step))
            #     with open(file_path, "w") as fo:
            #         for key in test_info:
            #             fo.write("{}: {}\n".format(key, test_info[key][-1]))

            metric = args.metric
            if test_info[metric][-1] > best_result:
                best_result = test_info[metric][-1]
                torch.save(model.state_dict(), os.path.join(args.save_models, args.model_name + '-{}.pkl'.format(step)))
                file_path = os.path.join(output_dir, '{}-step-result.txt'.format(step))
                with open(file_path, "w") as fo:
                    for key in test_info:
                        fo.write("{}: {}\n".format(key, test_info[key][-1]))


if __name__ == '__main__':
    main()







