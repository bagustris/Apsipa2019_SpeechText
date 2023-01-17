# APSIPA2019_SpeechText
Repository for code, paper, and slide ~submitted~ presented ~for~ at APSIPA 2019:

[**Speech emotion recognition Using Speech Feature and Word Embedding**](https://github.com/bagustris/Apsipa2019_SpeechText/blob/master/apsipa2019_bta.pdf)

Pre-processing:   
Run the following file with some adjusments (location of IEMOCAP data, output file name, etc.).  
https://github.com/bagustris/Apsipa2019_SpeechText/blob/master/code/python_files/mocap_data_collect.py 


Main codes:  
- [speeh_emo.ipynb](code/python_files/speech_emo.ipynb): for speech emotion recognition
- [text_emo.ipynb](./code/python_files/text_emo.ipynb): for text emotion recognition
- [speech_text.ipynb](code/python_files/speech_text.ipynb): for speech and text emotion recognition (main proposal)

Other (python) files can be explored and run indepently.

In case of the jupyter notebook is not rendered by Github, see the following nbviewer instead:
- https://nbviewer.jupyter.org/github/bagustris/Apsipa2019_SpeechText/blob/master/code/python_files/speech_emo.ipynb
- https://nbviewer.jupyter.org/github/bagustris/Apsipa2019_SpeechText/blob/master/code/python_files/text_emo.ipynb
- https://nbviewer.jupyter.org/github/bagustris/Apsipa2019_SpeechText/blob/master/code/python_files/speech_text.ipynb

By employing acoustic feature from voice parts of speech and word embedding from text we got boost accuracy of 75.49%. Here the list of obtained accuracy from different models (Text+Speech):
~~~~
------------------------------
Model           | Accuracy (%)
------------------------------
Dense+Dense     | 63.86
Conv1D+Dense    | 68.82
LSTM+BLSTm      | 69.13
LSTM+Dense      | 75.49 
-----------------------------
~~~~

<!---Due to license issue, the script to save acoustic feature is not included, but it can easily implemented by reading the paper (pdf of paper will be uploaded soon) -->

## Sample of feature
Due to Github's limitation, a sample of feature can be downloaded here (voiced feature without SIL removal): 
https://cloud.degoo.com/share/Ov563dopNnEW14jN_DeBig.
You can use the following script inside `code/python_files` directory to generate that feature file: https://github.com/bagustris/Apsipa2019_SpeechText/blob/master/code/python_files/save_feature.py

## Citation
~~~latex
B.T. Atmaja, M. Akagi, K. Shirai. "Speech Emotion Recognition from Speech Feature and Word Embedding", 
In Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA), IEEE, 2019.
~~~
