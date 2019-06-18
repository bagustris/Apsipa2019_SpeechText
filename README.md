# APSIPA2019_SpeechText
Repository for code and paper submitted for ~interspeech 2019~ APSIPA 2019:

**Speech emotion recognition Using Voice Feature and Text Embedding**

Main codes:  
- [speeh_emo.ipynb](code/python_files/speech_emo.ipynb): for speech emotion recognition
- [text_emo.ipynb](./code/python_files/text_emo.ipynb): for text emotion recognition
- [speech_text.ipynb](code/python_files/speech_text.ipynb): for speech and text emotion recognition (main proposal)

Other (python) files can be explored and run indepently.

In case of the jupyter notebook is not rendered by Github, see the following nbviewer instead:
- https://nbviewer.jupyter.org/github/bagustris/interspeech2019_SpeechText/blob/master/code/python_files/speech_emo.ipynb
- https://nbviewer.jupyter.org/github/bagustris/interspeech2019_SpeechText/blob/master/code/python_files/text_emo.ipynb
- https://nbviewer.jupyter.org/github/bagustris/interspeech2019_SpeechText/blob/master/code/python_files/speech_text.ipynb

By employing acoustic feature from voice parts of speech and word embedding from text we got boost accuracy of 75.49%. Here the list of obtained accuracy from different models (Text+Speech):
~~~~
------------------------------
Model           | Accuracy (%)
------------------------------
Conv1D+Dense    | 68.82
Dense+Dense     | 63.86
LSTM+Dense      | 75.49 
-----------------------------
~~~~

<!---Due to license issue, the script to save acoustic feature is not included, but it can easily implemented by reading the paper (pdf of paper will be uploaded soon) -->
