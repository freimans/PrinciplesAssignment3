import pandas as pd

data = pd.DataFrame(columns=('word','angr', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'))

with open('/Users/shaharfreiman/Desktop/Degree/Y4S1/Principles of Programming Languages/Assignments/Assignment3/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') as file:
    line = file.readline()
    line = file.readline()
    line = line[0:-1]
    elements = line.split('\t')
    while line:
        word = elements[0]
        word_row = [word, elements[2]]
        line = file.readline()
        line = line[0:-1]
        elements = line.split('\t')
        while elements[0] == word:
            word_row.append(elements[2])
            line = file.readline()
            line = line[0:-1]
            elements = line.split('\t')
        data.loc[len(data)] = word_row