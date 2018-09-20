## Building a tf-idf matrix for string matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

class Matcher(object):
    def __init__(self,ngram_length=3):
        """
        Initialize a Matcher instance

        Parameters
        ----------
        ngram_length : length of the n-grams to be considered for string matching

        Returns
        -------
        None
        """
        self.ngram_length = ngram_length
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer=self.ngrams)
        self.tfidf_matrix = None
        self.sequences = None

    def ngrams(self,string):
        """
        Decomposes a string into its n-grams. Is used by the tf-idf vectorizer.

        Parameters
        ----------
        string : a string to be decomposed

        Returns
        -------
        A list of n-grams for the given string
        """
        ngrams = zip(*[string[i:] for i in range(self.ngram_length)])
        return [''.join(ngram) for ngram in ngrams]
    
    def fit(self,sequences):
        """
        Trains the string Matcher on a corpus of strings.

        Parameters
        ----------
        sequences : a list of strings against
                    which future comparisons will be made.

        Returns
        -------
        None
        """
        self.sequences = sequences
        self.tfidf_matrix = (self.vectorizer).fit_transform(sequences)

    def most_similar(self,string):
        """
        Returns the best match to a given string 

        Parameters
        ----------
        string : a string for which the Matcher will
                 find the best match

        Returns
        -------
        match,score: match is the best matching string from
                     the fitted sequences, and score is the
                     cosine similarity of the input string
                     to the best match.
        """
        vect = (self.vectorizer).transform([string])
        cos_values = cosine_similarity(vect,self.tfidf_matrix)
        idx = np.argsort(-cos_values)[0,0]
        return (self.sequences[idx],cos_values[0,idx])
    
    def get_scores(self,string):
        """
        Returns the similarity scores of a string with
        all the fitted sequences of the Matcher.

        Parameters
        ----------
        string : a string for which the Matcher will
                 calculate the similarity scores.

        Returns
        -------
        A list of pairs (s,score), where 's' is
        one of the fitted string, and 'score' is the cosine
        similarity between the input 'string' and 's'
        """
        vect = (self.vectorizer).transform([string])
        cos_values = cosine_similarity(vect,self.tfidf_matrix).squeeze().tolist()
        return list(zip(self.sequences,cos_values))
    
    def get_matcher_dict(self,corpus):
        """
        Returns a dictionary mapping all elements in
        a corpus of strings to their best match. Takes
        advantage of fast vectorization to calculate
        multiple matches.

        Parameters
        ----------
        corpus : a list of strings for which the Matcher will
                 calculate the best matches.

        Returns
        -------
        A dictionary, the keys of which are elements of the
        input corpus, the values of which are pairs (match,score),
        where match is the best matching string for the given
        key, and score is the cosine similarity between the key
        and the best match.
        """
        corpus_vects = (self.vectorizer).transform(corpus)

        cos_values = cosine_similarity(corpus_vects,self.tfidf_matrix)
        the_dict={}
        for text,idx,score in zip(corpus,np.argmax(cos_values,axis=1),np.max(cos_values,axis=1)):
            the_dict[text] = (self.sequences[idx],score)
            
        return the_dict