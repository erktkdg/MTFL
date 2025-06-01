import argparse

############ Test args ########################
test_parser = argparse.ArgumentParser(description='MTFL_recognition_test')
# input path
test_parser.add_argument('--lf_dir', type=str, default='features/L64', help='long frame length feature path')
test_parser.add_argument('--mf_dir', type=str, default='features/L32', help='media frame length feature path')
test_parser.add_argument('--sf_dir', type=str, default='features/L8', help='short frame length feature path')
test_parser.add_argument('--test_anno', type=str, default='annotation/Anomaly_videos.txt', help='test annotation file')
test_parser.add_argument('--test_dataset', type=str, default='other', choices=['UCF', 'VAD', 'other'],
                         help='The test data. The test results are the recognized labels of all input videos. '
                              'For UCF and VAD datasets, the overall accuracy would be printed out')
test_parser.add_argument('--recognition_model', type=str,
                         default='/media/DataDrive/yiling/Test/models/MTFL_recog/split_1_best_VAD.pkl',
                         help='recognition checkpoint path, choose 1 from 7 checkpoints trained on different splits')
# output path
test_parser.add_argument('--output_dir', type=str, default='results',
                         help='The path to store the recognition result')
# feature size depending on which feature extractor used
test_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
test_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
# running cfg
test_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu')
test_parser.add_argument('--workers', default=8, help='number of workers in dataloader')


############ Train args ########################
train_parser = argparse.ArgumentParser(description='MTFL_recognition_train')
# input path
train_parser.add_argument('--lf_dir', type=str, default='/media/DataDrive/yiling/features/recognition/split1_L64R1',
                          help='long feature path')
train_parser.add_argument('--mf_dir', type=str, default='/media/DataDrive/yiling/features/recognition/split1_L32R1',
                          help='media feature path')
train_parser.add_argument('--sf_dir', type=str, default='/media/DataDrive/yiling/features/recognition/split1_L8R1',
                          help='short feature path')
train_parser.add_argument('--train_anno', default='/media/DataDrive/yiling/annotation/recognition/splits/VAD/VAD_train_001.txt',
                          help='the annotation file for training')
train_parser.add_argument('--test_anno', default='/media/DataDrive/yiling/annotation/recognition/splits/VAD/VAD_test_001.txt',
                          help='the annotation file for test')
train_parser.add_argument('--test_dataset', type=str, default='UCF', choices=['UCF', 'VAD'],
                         help='The validation data')
# output path and saving info
train_parser.add_argument('--model-name', default='MTFL_recognition', help='name to save model')
train_parser.add_argument('--save_models', default='/media/DataDrive/yiling/models/demo/recognition',
                          help='the path for saving models')
train_parser.add_argument('--output_dir', default='/media/DataDrive/yiling/results/demo/recognition',
                          help='The path to store AUC results')
# training cfg and paras
train_parser.add_argument('--gpu', default="0", type=str, choices=["0", "1"], help='gpu id')
train_parser.add_argument('--feature_size', type=int, default=1024, help='feature dim (default: VST feature)')
train_parser.add_argument('--seg_num', type=int, default=32, help='the number of snippets')
train_parser.add_argument('--lr', type=float, default='0.0001', help='learning rates for steps(list form)')
train_parser.add_argument('--batch-size', type=int, default=32, help='batch size')
train_parser.add_argument('--workers', default=8, help='number of workers in dataloader')
train_parser.add_argument('--max-epoch', type=int, default=2000, help='maximum iteration to train (default: 100)')

