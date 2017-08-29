import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import logging
import cPickle as pickle
from multiprocessing.dummy import Pool as ThreadPool
import cProfile

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)    # type: pandas.Dataframe

STOP_WORDS = stopwords.words('english')


def review_to_words( raw_review ):
	"""
	Process a raw review into a string of words
	:param raw_review:
	:return:
	:type   raw_review: str
	:rtype  str
	"""
	# Initialize the BeautifulSoup object on a single movie review
	review_text = BeautifulSoup(raw_review, 'lxml').get_text()
	
	# Use re to filter non-alphabetic letter
	letters_only = re.sub('[^A-Za-z]', ' ', review_text)
	words = letters_only.lower().split()
	
	# Remove stop words from 'words'
	meaningful_words = [w for w in words if w not in STOP_WORDS]
	
	return (' '.join(meaningful_words))
	
	
def clean_review():
	logger.info('Cleaning and parsing the training set movie reviews...')
	num_review = train['review'].size
	clean_train_review = []
	for i in xrange(0, num_review):
		# logging
		if ( i % 100 == 0):
			logger.info('Progress: %f%%. Review %i of %i', float(i)/num_review * 100, i, num_review)
		
		clean_train_review.append(review_to_words(train['review'][i]))
	return clean_train_review


def clean_review_multi(worker = 4):
	"""
	Cleaning data. Enable multi threading features.
		cProfile( worker = 4 )  # Macbook Pro: 132 function calls (6988 primitive calls) in 148.486 seconds
	:param worker:
	:return:
	"""
	logger.info('Cleaning and parsing the training set movie reviews...')
	pool = ThreadPool(4)
	num_review = train['review'].size
	results = []
	for i in xrange(0, num_review - 1000, 1000):
		array = train['review'][i: i + 1000]
		logger.info('Progress: %f%%. Review %i of %i', float(i) / num_review * 100, i, num_review)
		results.extend(pool.map(review_to_words, array))
	return results

cProfile.run('result = clean_review_multi(worker = 4)')
with open('pickle/clean_train_review', 'w') as f:
	pickle.dump(result, f)



	

