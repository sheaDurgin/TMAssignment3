# TMAssignment3

## Task 1
This task scrapes from lyrics.com, to get genre-to-lyrics pairs. 
At the end it will create three files, train.tsv, val.tsv, test.tsv. 
This data is a 80:10:10 split of the total data scraped. 

Simply run `python task1.py`. 
This will take hours to run, which is why we have provided the resulting tsvs. 

## Inbetween
Run `python get_test_tsv.py` to convert 'Test Songs' directory to `test.tsv`

## Task 2
This task consist of using the scraped data from task 1, and use it to 
train a feed-forward neural network model with simple features from the 
songs like number of word, or unique words.

The goal of the model is to predict the genre of the songs by their lyrics.

Run `python task2.py` to train the model and also show the results.

## Task 3
