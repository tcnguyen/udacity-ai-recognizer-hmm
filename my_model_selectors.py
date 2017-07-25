import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #print("**** train for word: ", self.this_word)
        #print("number of sequences: ", len(self.sequences))
        min_bic = float('inf')
        best_model = None
        d = len(self.X[0]) # number of features
        N = len(self.X) # number of data points

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                #print("try with n = ", n)
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)

                model.fit(self.X, self.lengths)
                L = model.score(self.X, self.lengths)
                p = n*n + 2*n*d -1
                bic = -2*L + p*np.log(N)

                if min_bic > bic:
                    #print("   best_n = ", n)
                    min_bic = bic
                    best_model = model


            except:
                #print("   failed")
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        #print("**** train for word: ", self.this_word)
        #print("number of sequences: ", len(self.sequences))
        if len(self.sequences) <= 2:
            return self.base_model(self.n_constant)

        max_score = float('-inf')


        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                #print("try with n = ", n)
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False)
                scores = []

                split_method = KFold(n_splits = min(3, len(self.sequences)))

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    model.fit(train_X, train_lengths)
                    scores.append( model.score(test_X, test_lengths) )

                cv_score = np.mean(scores)

                if max_score < cv_score:
                    max_score = cv_score
                    best_n = n
                    #print("   best_n = ", best_n)

            except:
                #print("   failed")
                pass

        best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                    random_state=self.random_state, verbose=False)

        best_model.fit(self.X, self.lengths)

        return best_model