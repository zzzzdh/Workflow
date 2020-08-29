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

| ![](/images/example.jpg) | 
|:--:| 
| *Fig. 1: An Example of Programming Workflow* |

| ![](/images/step.jpg) | 
|:--:| 
| *Fig. 2: Main Steps of Our Approach* |

| ![](/images/actionnet.jpg) | 
|:--:| 
| *Fig. 3: Illustration of how ActionNet works* |

| ![](/images/popup.jpg) | 
|:--:| 
| *Fig. 4: An example of processing pop-up assisted code editing* |

| ![](/images/scroll.jpg) | 
|:--:| 
| *Fig. 5: An example of scroll-content in between coding actions* |

| ![](/images/textbox.jpg) | 
|:--:| 
| *Fig. 6: Example of text line detection* |

| ![](/images/fragment.jpg) | 
|:--:| 
| *Fig. 7: Illustration of coding step identification* |

| ![](/images/box.jpg) | 
|:--:| 
| *Fig. 8: An example of locating active text line(s)* |

| ![](/images/frag_len_all.jpg) | 
|:--:| 
| *Fig. 9: Distribution of coding-step length and total duration* |

| ![](/images/offset.jpg) | 
|:--:| 
| *Fig. 10: Coding-step distribution at different IoU and timeoffset thresholds* |

| ![](/images/iou.jpg) | 
|:--:| 
| *Fig. 11: Per-video F1 distribution at different IoU and timeoffset thresholds* |

| ![](/images/playlist.jpg) | 
|:--:| 
| *Fig. 12: Distribution of videos in different F1 ranges for each playlist* |

| ![](/images/usefulness.jpg) | 
|:--:| 
| *Fig. 13: Correctness ratings by human evaluation* |

