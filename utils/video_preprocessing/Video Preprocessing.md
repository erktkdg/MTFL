# Video Preprocessing

If you need to perform video annotation or standardize video formats, 
run 'utils/video_preprocessing/video_annotator.py' or 'utils/video_preprocessing/video_format_unifier.py'.

## Annotation
To run video_annotator.py, you need to specify the following parameters in the code according to your needs:
```
--root_dir: Denotes the root directory where video files are stored. 
--video_subdir: Specifies the subdirectory containing the videos to annotate.
--annotation_file: Represents the directory where the generated annotation file should be saved.
--anomaly_type: Indicates the type of the videos. 
```
Feel free to use anomaly types other than those in the VAD dataset for the testing data annotations, 
as they will be processed as 'anomaly' in both detection and recognition, without affecting the inference.


When the video playback window starts, the corresponding hotkeys are as follows: 
* space to pause
* 'a' to reverse 10 frames
* 'd' to skip 10 frames
* ',' to reverse 1 frame
* '.' to skip 1 frame
* 'm' to mark the current frame as the start or end of an anomalous event
* 'z' to cancel the last marking
* 's' to skip this video
* 'q' to exit.

Each line of the generated annotation includes:
* the relative path of the video with respect to the root directory
* the video label
* the number of frames in the video
* pairs of start and end frames for anomalous events.

Using relative paths here is to explicitly define the relative path of a video with
respect to the 'test_videos' folder, and to maintain this relative structure for the 
corresponding features and results storage.

## Unifying Format
To run video_format_unifier.py, you need to specify the following parameters in the code according to your needs:
```
--video_dir: Path to the input directory containing the video files.
--out_dir: Path to the output directory where processed videos will be saved.
```
The videos in [video_dir] will be modified to a resolution of 320x240 and a frame rate of 30fps, 
and they will be saved with the same names in the [out_dir]. If you want to try other format, feel free to change
<target_res> and <target_fps> in the code.