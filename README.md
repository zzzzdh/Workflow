# Workflow

## Data
We consider 2 programming language, python and java.

Each programming language has 3 playlists. There are totally 6 playlists and 360 vidoes for evaluation.

## Overall result
This image shows the overall result including IOU and bias.
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/all_result.png">

## Playlist result
### IOU
There is an outlier that have pretty low result. This is playlist 10. The author in this playlist have too much useless actions. For example, he keeps selecting many code lines again and again, which add much noise to the results. It's also pretty hard for me to label this playlist because of these useless actions.
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/playlist_results_IOU.png">

### bias
The outlier is playlist 2 which has higer results. In this playlist the actions are clear and simple. So it is easy to recognize.
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/playlist_results_bias.png">

### punctuation
The outlier is playlist 10. I found that the captions in the other 5 playlists are auto generated by Youtube. The speech recognitione algorithm will keep all the words the authors say. Some times there will be multiple words like 'the the the' because of the unsmooth speaking. However, the caption in playlist 10 is edited by author. So the author will remove the noise words and make the caption readable.
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/playlist_results_punctuation.png">

## Programming language result
The results of two programming language are very close. Python is a little bit higher. The main reason is the syntax. In python, we use space or table to keep syntax, which means there are few long code line. But in Java, we use ; which means there are many long code lines. Our method have better performance on short code line. In tutorial video, when the author need to show a long code line, he usually scroll the window horizentally or zoom in/out. Our method can't solve these complex situation.

### java
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/java_result.png">

### python
<img src="https://github.com/zzzzdh/Workflow/blob/master/image/python_result.png">

## TODO
I have finised evaluating the performance of fragment extraction and caption punctuation.

I'm conducting the evaluation of the usefulness of the extracted workflow. I think it will be finished this week. Then I plan to use one week to finish drafting this paper.

