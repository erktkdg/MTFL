import argparse

############ Test args ########################
test_parser = argparse.ArgumentParser(description='MTFL_detection_test')
# input path
test_parser.add_argument('--lf_dir', type=str, default='features/L64', help='long frame length feature path')
test_parser.add_argument('--mf_dir', type=str, default='features/L32', help='media frame length feature path')
test_parser.add_argument('--sf_dir', type=str, default='features/L8', help='short frame length feature path')
test_parser.add_argument('--test_anno', default='annotation/Anomaly_videos.txt', help='test annotation file')
test_parser.add_argument('--detection_model', default='/media/DataDrive/yiling/Test/models/MTFL/MTFL-vst-VAD.pkl',
                         help='model path')
# output path
test_parser.add_argument('--output_dir', default='results',
                         help='The path to store the generated scores and AUC results')
# feature size depending on which feature extractor used
test_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
test_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
# running cfg
test_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
test_parser.add_argument('--workers', default=8, help='number of workers in dataloader')


############ Train args ########################
train_parser = argparse.ArgumentParser(description='MTFL_detection_train')
# input path
train_parser.add_argument('--lf_dir', type=str, default='/media/DataDrive/yiling/features/VST_VAD_MT/L64R1',
                          help='long feature path')
train_parser.add_argument('--mf_dir', type=str, default='/media/DataDrive/yiling/features/VST_VAD_MT/L32R1',
                          help='media feature path')
train_parser.add_argument('--sf_dir', type=str, default='/media/DataDrive/yiling/features/VST_VAD_MT/L8R1',
                          help='short feature path')
train_parser.add_argument('--train_anno', default='/media/DataDrive/yiling/annotation/VAD_train_annotation.txt',
                          help='the annotation file for training')
train_parser.add_argument('--test_anno', default='/media/DataDrive/yiling/annotation/UCF_test_annotation_with_frames.txt',
                          help='the annotation file for test')
# output path and saving info
train_parser.add_argument('--model-name', default='MTFL', help='name to save model')
train_parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/demo/detection',
                          help='the path for saving models')
train_parser.add_argument('--output_dir', default='/media/DataDrive/yiling/results/demo/detection',
                          help='The path to store AUC results')
# training cfg and paras
train_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu id')
train_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
train_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
train_parser.add_argument('--lr', type=float, default='0.0001', help='learning rates for steps(list form)')
train_parser.add_argument('--batch-size', type=int, default=64, help='batch size')
train_parser.add_argument('--workers', type=int, default=8, help='number of workers in dataloader')
train_parser.add_argument('--max-epoch', type=int, default=2000, help='maximum iteration to train (default: 100)')
train_parser.add_argument('--metric', type=str, choices=["AP", "AUC"], default="AUC", help='the used metric')






