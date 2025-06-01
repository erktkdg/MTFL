import torch
from tqdm import tqdm
import numpy as np
import os
import option
from torch.utils.data import DataLoader
from dataset import class_to_int, Dataset
from model import Model


def top_k_accuracy(scores, labels, topk=(1, 5)):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def test(dataloader, model, device, test_dataset='UCF'):
    """
    Evaluate the model's performance on the test dataset and return the top-1 accuracy.

    Args:
        dataloader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): The trained neural network model.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        test_dataset (str, optional): The name of the test dataset, either 'UCF' or 'VAD'. Default is 'UCF'.
                The overall accuracy is calculated only for 'VAD' and 'UCF' because it does not make sense when testing
                on only a few videos.

    Returns:
        float: The top-1 accuracy of the model on the test dataset.
        dict: A dictionary containing video filenames and their corresponding predicted classes.

    """
    video_class = {"video": [], "class": []}
    with torch.no_grad():
        model.to(device).eval()
        outputs = torch.zeros(0, device=device)
        labels = torch.zeros(0, device=device)

        for input1, input2, input3, label, file in tqdm(dataloader):
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            label = label.to(device)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, scores = model(input1, input2, input3)
            # cat for acc evaluation
            outputs = torch.cat((outputs, score_abnormal))
            labels = torch.cat((labels, label))
            # obtain the prediction result
            score_abnormal = score_abnormal.cpu().detach().numpy()
            pred = np.argmax(score_abnormal, axis=1)
            found_class = [key for key, value in class_to_int.items() if value == pred[0]]
            file_name = os.path.basename(file[0])
            video_class["video"].append(file_name)
            video_class["class"].append(found_class)

        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        res = [-1]

        if test_dataset == 'UCF':  # all road accidents in UCF are labelled as 13
            for row in outputs:
                max_value = max(row[13], row[14], row[15])
                row[13] = max_value
                row[14] = 0.0
                row[15] = 0.0

        # Accuracy makes sense only when the test classes are involved in VAD
        if test_dataset == 'UCF' or test_dataset == 'VAD':
            res = top_k_accuracy(outputs, labels)
            print('\n' + str(test_dataset) + ' top1 : ' + str(res[0]) + ' top5 : ' + str(res[1]) + '\n')

        return res[0], video_class


def main():
    args = option.test_parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_path = os.path.join(args.output_dir, 'rec_results')

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = Model(feature_dim=args.feature_size, batch_size=1, seg_num=args.seg_num)
    model.load_state_dict(torch.load(args.recognition_model))

    _, video_class = test(dataloader=test_loader,
                          model=model,
                          device=device,
                          test_dataset=args.test_dataset)
    # save recognition results
    video_sub_dir = os.path.basename(os.path.dirname(video_class["video"][0][0]))
    file_path = os.path.join(out_path, video_sub_dir, 'output_pred.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for video, cls in zip(video_class["video"], video_class["class"]):
            f.write(f"Video: {video}, class: {cls}\n")


if __name__ == '__main__':
    main()