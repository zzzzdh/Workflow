# SeeHow: Workflow Extraction from Programming Screencasts through Action-Aware Video Analytics

This is source code for SeeHow.

We implement a tool for it
[TOOL](http://seecollections.com/seehow/)

## Dataset
You can download the dataset [Here](http://seecollections.com/seehow/) or process your own data using the code, please follow the instructions.

## Data processing
Note: You need to change your own path for each process.
1. Decode videos to frames by `python3 clip_video_ffmpeg.py`. This step uses ffmpeg. You can install it by `apt-get install ffmpeg`. We also process captions in this step. But it's an option for you.
2. Action region detection by `python3 compare_image.py`. 
3. Action category classification by `python3 extract_action.py`. This step uses ActionNet. Please refere to [ActionNet](https://github.com/DehaiZhao/ActionNet) for details. The model can be downloaded [Here](https://drive.google.com/file/d/1lbrtNbnfR9T6epawMRCMC-xdj1wsm4IK/view?usp=sharing)
4. Text detection by [EAST](https://github.com/argman/EAST) and text recognition by [CRNN](https://github.com/bgshih/crnn). You can refer to the two methods to process your own data.
5. Extrac workflow by `python3 extract_workflow.py`. We store the results in mysql database. You can build a database to store the results or use any other forms.

