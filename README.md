# SeeHow: Workflow Extraction from Programming Screencasts through Action-Aware Video Analytics

This is source code for SeeHow.

We implement a [TOOL](http://seecollections.com/seehow/)

## Dataset
You can download the dataset [Here](https://drive.google.com/file/d/1gT3afHGFIxfPlWK4LCq8Cp1RlhPboRgw/view?usp=sharing) or process your own data using the code, please follow the instructions.

If you use our dataset, you can run `python3 extract_workflow.py` directly to extract workflow.

## Data processing
Note: You need to change your own path for each process.
1. Decode videos to frames by `python3 clip_video_ffmpeg.py`. This step uses ffmpeg. You can install it by `apt-get install ffmpeg`. We also process captions in this step. But it's an option for you.
2. Action region detection by `python3 compare_image.py`. 
3. Action category classification by `python3 extract_action.py`. This step uses ActionNet. Please refere to [ActionNet](https://github.com/DehaiZhao/ActionNet) for details. The model can be downloaded [Here](https://drive.google.com/file/d/1lbrtNbnfR9T6epawMRCMC-xdj1wsm4IK/view?usp=sharing)
4. Text detection by [EAST](https://github.com/argman/EAST) and text recognition by [CRNN](https://github.com/bgshih/crnn). You can refer to the two methods to process your own data. You can refer to `connect_box.py` to connect word level boxes to text lines.
5. Extrac workflow by `python3 extract_workflow.py`. We store the results in mysql database. You can build a database to store the results or use any other forms.

## Additional process
We also consider captions in this work, which is used to generate text summary to describe the coding steps. Please refer to our [TOOL](http://seecollections.com/seehow/) for the results.

In order to obtain the text summary, you need to:
1. Parse vtt file, which has been done in `clip_video_ffmpeg.py`
2. Punctuation restoration by `segment_punctuation.py`, as the captions do not have any punctuation.
3. Caption group by `next_sentence.py`. This step groups related sentences together.
4. Caption summarization by `summarize.py`. This step summarize long sentences to short sentences. Please refer to [This](https://github.com/nlpyang/PreSumm) method.

## Figures in paper
We provide the high resolution figures for the paper.
![](/images/actionnet.jpg)
  Fig. 1: An Example of Programming Workflow

