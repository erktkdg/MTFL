import torch
from sklearn.metrics import auc, roc_curve, average_precision_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import option

from torch.utils.data import DataLoader
from dataset import Dataset
from model import Model
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def get_gt(start_end_couples, num_frames, device):
    """
    Generate a ground truth tensor representing events in a time sequence based on given start and end pairs.

    Args:
        start_end_couples (list): A list containing pairs of start and end frames.
            If None or all '-1', no events are present.
        num_frames (int): Total number of frames in the time sequence.
        device: Device where the tensor should be placed.

    Returns:
        gt: A tensor of shape (num_frames,) representing whether each frame belongs to an anomalous event.
            '1' means anomalous, and '0' means normal.
    """
    gt = torch.zeros(num_frames).to(device)
    if start_end_couples is not None and num_frames is not None:
        for i in range(0, len(start_end_couples) - 1, 2):
            if start_end_couples[i].item() != -1 and start_end_couples[i + 1].item() != -1:
                couple = start_end_couples[i:i + 2]
                gt[couple[0].item():couple[1].item()] = 1.0

    return gt


def save_scores(pred, start_end_couples, save_path):
    """
    Save plots containing anomaly scores and annotated regions.

    Args:
        pred (list): List of anomaly scores.
        start_end_couples (Tensor): Pairs of start and end frames indicating anomalous regions.
        save_path (str): Path to save the generated plot.
        file_name (str): Name to be displayed in the legend of the plot.
    """

    plt.figure()
    file_name = os.path.basename(save_path).split(".")[0]
    plt.plot(pred, label=file_name, color='blue')

    # Plot anomalous regions
    for i in range(0, len(start_end_couples) - 1, 2):
        if start_end_couples[i].item() != -1 and start_end_couples[i + 1].item() != -1:
            plt.axvspan(start_end_couples[i].item(), start_end_couples[i + 1].item(), color='red', alpha=0.3)

    plt.ylim(0, 1)
    plt.xlabel('Frames', fontdict={'size': 16})
    plt.ylabel('Anomaly Score', fontdict={'size': 16})
    plt.yticks(size=14)
    plt.xticks(size=14)

    plt.legend(prop={'size': 16})
    #plt.show()
    plt.savefig(save_path)
    plt.close()


def test(dataloader, model, device, gen_scores=False, save_dir=None):
    """
    Test the model's performance on the given dataloader.

    Args:
        dataloader (DataLoader): DataLoader for test data.
        model: The model to be tested.
        device: Device to perform testing on.
        gen_scores (bool): Whether to generate and save anomaly scores plot.
        save_dir (str): Directory to save generated plots.

    Returns:
        single_video_AUC (dict): A dictionary containing AUC values for each video.
        overall_auc (float): Overall AUC value.
        ap (float): average precision
    """
    single_video_AUC = {"video": [], "AUC": []}

    with torch.no_grad():
        model.to(device).eval()
        pred = torch.zeros(0, device=device)
        gt = torch.zeros(0, device=device)

        for input1, input2, input3, label, start_end_couples, num_frames, file in tqdm(dataloader):
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input1, input2, input3)
            sig = torch.squeeze(scores, dim=(0, 2)) # T scores
            segment = num_frames.item() // sig.size()[0]
            sig = sig.repeat_interleave(segment) # Frames
            if len(sig) < num_frames.item():
                last_ele = sig[-1]
                sig = torch.cat((sig, last_ele.repeat(num_frames.item()-len(sig)))) # 1 x Frames

            pred = torch.cat((pred, sig))
            cur_gt = get_gt(start_end_couples, num_frames, device)
            gt = torch.cat((gt, cur_gt))

            sig = sig.cpu().detach().numpy()
            cur_gt = cur_gt.cpu().detach().numpy()
            fpr, tpr, threshold = roc_curve(cur_gt, sig)
            video_auc = auc(fpr, tpr)
            single_video_AUC["video"].append(file)
            single_video_AUC["AUC"].append(video_auc)

            if gen_scores:
                save_path = os.path.join(save_dir, file[0] + '.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_scores(sig, start_end_couples, save_path)

        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        ap = average_precision_score(gt, pred)
        fpr, tpr, threshold = roc_curve(gt, pred)
        overall_auc = auc(fpr, tpr)
        print('\n' + 'Overall auc : ' + str(overall_auc) + ', Average Precision : ' + str(ap) + '\n')

        return single_video_AUC, overall_auc, ap


def main():
    args = option.test_parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    AUC_path = os.path.join(args.output_dir, 'AUC')
    scores_path = os.path.join(args.output_dir, 'scores')

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = Model(feature_dim=args.feature_size, batch_size=1, seg_num=args.seg_num)
    model.load_state_dict(torch.load(args.detection_model))

    single_video_AUC, overall_auc, ap = test(dataloader=test_loader,
                                             model=model,
                                             device=device,
                                             gen_scores=True,
                                             save_dir=scores_path)

    # save AUC results
    video_sub_dir = os.path.basename(os.path.dirname(single_video_AUC["video"][0][0]))
    file_path = os.path.join(AUC_path, video_sub_dir, 'results.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for video, single_auc in zip(single_video_AUC["video"], single_video_AUC["AUC"]):
            f.write(f"Video: {video}, AUC: {single_auc}\n")
        f.write("Overall AUC: {}, Average Precision: {}\n".format(overall_auc, ap))


if __name__ == '__main__':
    main()


