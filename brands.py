from sklearn.feature_extraction import text
from multiprocessing import Pool, Manager, active_children, cpu_count, JoinableQueue, Queue, Process, current_process
from scipy.sparse import csr_matrix, diags
from pprint import pprint as pp
from functools import partial
from itertools import combinations
from ast import literal_eval
import numpy as np
import pandas as pd
import time

OCC_THRESHOLD = 500
NUM_CORES = 2
NUM_GRAMS = 10

it = pd.read_csv('...')
it['text'] = it.text.replace({'[^\x00-\x7F]': '', 'Untitled Item': '', 'untitled item': '', '[\/<>;~&@#\$\^\*\(\)\":_]': ' ', '[\'\.-]': ''}, regex=True)
it = it[it.text.str.len()>1]
it = it.reset_index(drop=True)
it['item_index'] = it.index
N_DOCS = it.shape[0]
STOP_WORDS = set(text.ENGLISH_STOP_WORDS)
STOP_WORDS = STOP_WORDS | set([_[0].upper()+_[1:] for _ in STOP_WORDS]) | set([_.upper() for _ in STOP_WORDS])
COLORS = pd.read_csv('/home/anindya/files/ebaycats/colors.csv', encoding='utf-8')
COLORS.color = COLORS.color.str.lower()
COLORS = set(COLORS.color.values)

filter_dict_by_value = lambda d,v: dict([(k,_) for k,_ in d.items() if _>v])
filter_dict_by_numwords = lambda d,v: dict([(k,_) for k,_ in d.items() if len(k.split())==v])
filter_keys_by_numwords = lambda d,v: filter(lambda _: len(_.split()) in set(v), d.keys())
filter_list_by_numwords = lambda l,v: filter(lambda _: len(_.split()) in set(v), l)

def fit_model_ngram(text_array, ngram, vocab=None, stop_words=None):
    cv = text.CountVectorizer(ngram_range=ngram, binary=True, analyzer='word', token_pattern='\\b\w+\\b', lowercase=False, stop_words=stop_words, vocabulary=vocab, min_df=OCC_THRESHOLD)
    fitted = cv.fit_transform(text_array)
    return fitted, cv.vocabulary_

def sieve_tokens(tokens):
    tokens = sorted(tokens, key=lambda _: len(_))
    sieved = list()
    for i in range(len(tokens)-1):
        if not tokens[i] in tokens[i+1]:
            sieved.append(tokens[i])
    sieved.append(tokens[-1])
    out = list()
    for i in range(len(sieved)):
        if not any(sieved[i] in _ for _ in sieved[i+1:]):
            out.append(sieved[i])
    return out

def get_adjacency_scores(fitted, vocab, cutoff=99):
    vi = dict(zip(vocab.values(), vocab.keys()))
    ngrams = set(vocab.keys())
    coc = fitted.T * fitted
    oc = dict(zip(sorted(vocab.keys()), np.array(coc.diagonal()).flatten()))
    coc.setdiag(0)
    coc.eliminate_zeros()
    coc = coc.tocoo()
    coc = pd.DataFrame(np.vstack([coc.row, coc.col, coc.data]).T, columns=['row', 'col', 'coc'])
    coc = coc[coc['coc']>np.percentile(coc.coc, 99)]
    coc['r'] = map(lambda r: vi[r], coc['row'])
    coc['c'] = map(lambda c: vi[c], coc['col'])
    coc['token'] = map(lambda r,c: r + ' ' + c, coc['r'], coc['c'])
    coc['valid'] = map(lambda t: t in ngrams, coc['token'])
    coc = coc[coc['valid']==True]
    coc['score'] = map(lambda r,c,co,t: (1. * oc[t])/(oc[r] * oc[c] * co), coc['r'], coc['c'], coc['coc'], coc['token'])
    coc['score2'] = map(lambda co,t: (1. * oc[t])/co, coc['coc'], coc['token'])
    coc = coc[coc['score'] >= np.percentile(coc.score.values, cutoff)]
    return oc, dict(zip(coc.token.values, coc.score.values))

def get_cooccurrence_for_numwords(values, numwords, cutoff=99):
    fitted, vocab = fit_model_ngram(values, (1,5))
    vi = dict(zip(vocab.values(), vocab.keys()))
    coc = fitted.T * fitted
    oc = dict(zip(range(len(vi)), np.array(coc.diagonal()).flatten()))
    coc.setdiag(0)
    coc.eliminate_zeros()
    coc = coc.tocoo()
    coc = pd.DataFrame(np.vstack([coc.row, coc.col, coc.data]).T, columns=['row', 'col', 'coc'])
    coc['score'] = map(lambda r,c,co: (1. * co)/(oc[r] * oc[c]), coc['row'], coc['col'], coc['coc'])
    coc = coc[coc['score'] >= np.percentile(coc.score.values, cutoff)]
    coc['row'] = coc['row'].map(vi)
    coc['col'] = coc['col'].map(vi)
    coc['token'] = coc['row'].map(str) + ' ' + coc['col'].map(str)
    return dict(zip(coc.token.values, coc.score.values))

def score_token(token, ng, cs):
    length = len(token.split())
    final = 0
    for g in range(1, length+1):
        cscores = [cs[_] for _ in ng[(token, g)]]
        score = reduce(lambda x,y: x*y, cscores)
        if score > final:
            final = score
    return (token, final)

def get_cond_score(gram):
    gs = gram.split()
    if len(gram.split())>1:
        return (gram, (1. * occur[gram])/occur[' '.join(gs[:-1])])
    else:
        return (gram, (1. * occur[gram])/N_DOCS)

def filter_tokens(length=1, cutoff=99):
    global scores, chosen
    vf = filter_dict_by_numwords(scores, length)
    chosen.update(filter_dict_by_value(vf, np.percentile(vf.values(), cutoff)))

def get_ngrams(s):
    strg = s.split()
    out = set()
    for g in range(1, len(strg)):
        out.update([' '.join(strg[:i+1]) for i in range(g-1)] + [' '.join(strg[i:i+g]) for i in range(len(strg)-g+1)])
    return out

global occur, scores, cscores, chosen, cchosen, cskeys, ngrams, condscore

class Consumer(Process):
    def __init__(self, task_queue, result_queue, dicts):
        Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.load_dicts(dicts)
        print current_process().name
    def load_dicts(self, dicts):
        for d in dicts:
            if d=='occur':
                global occur
                occur = pd.read_csv('occur.csv')
                occur = occur[~occur.token.isnull()]
                occur = occur.values
                occur = dict(zip(occur[:,0], map(float, occur[:,1])))
                #hack
                occur['NA'] = 209.0
                occur['NULL'] = 4.0
                occur['nan'] = 9.0
            elif d=='score':
                global condscore
                condscore = pd.read_csv('condscore.csv')
                condscore = condscore[~condscore.token.isnull()]
                condscore = condscore.values
                condscore = dict(zip(condscore[:,0], map(float, condscore[:,1])))
                #hack
                condscore['NA'] = 209.0/N_DOCS
                condscore['NULL'] = 4.0/N_DOCS
                condscore['nan'] = 9.0/N_DOCS
            elif d=='cscores':
                global cscores, cskeys
                cscores = pd.read_csv('cscores.csv')
                cskeys = set(cscores.token.values)
                cscores = dict(zip(cscores.token.values, cscores.score.values))
            elif d=='grouping':
                global finalg
                finalg = pd.read_csv('finalg.csv')
    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)

class Task:
    def __init__(self, token, name):
        self.token = token
        self.name = name
    def __call__(self):
        global occur
        global condscore
        global cscores, cskeys
        if self.name=='occur':
            return self.get_occur()
        elif self.name=='score':
            return self.get_score()
        elif self.name=='cscores':
            return self.score_by_cooccurrence()
        elif self.name=='grouping':
            return self.join_tokens()
    def get_occur(self):
        gs = self.token.split()
        if len(gs)>1:
            return (self.token, (1. * occur[self.token])/occur[' '.join(gs[:-1])])
        else:
            return (self.token, (1. * occur[self.token])/N_DOCS)
    def get_ngrams(self, s):
        strg = s.split()
        out = dict()
        for g in range(1, len(strg)+1):
            out[(s,g)] = [' '.join(strg[:i+1]) for i in range(g-1)] + [' '.join(strg[i:i+g]) for i in range(len(strg)-g+1)]
        return out
    def get_score(self):
        length = len(self.token.split())
        final = 0
        ngrams = self.get_ngrams(self.token)
        for g in range(1, length+1):
            cscores = [condscore[_] for _ in ngrams[(self.token, g)]]
            score = reduce(lambda x,y: x*y, cscores)
            if score>final:
                final = score
        return (self.token, final)
    def score_by_cooccurrence(self):
        combs = set([' '.join(_) for _ in combinations(self.token.split(), 2)]).intersection(cskeys)
        score = sum([cscores[_] for _ in combs])
        return (self.token, score)
    def sieve_tokens(self, tokens):
        tokens = sorted(tokens, key=lambda _: len(_))
        sieved = list()
        for i in range(len(tokens)):
            if not any(tokens[i] in _ for _ in tokens[i+1:]):
                sieved.append(tokens[i])
        return sieved
    def join_tokens(self):
        return (self.token, '|'.join(self.sieve_tokens(finalg[finalg['item_index']==self.token]['token'].values)))

def parexec(signal, out, num_consumers, iterator):
    t = time.time()
    tasks = JoinableQueue()
    results = Queue()
    print 'starting consumers'
    consumers = [Consumer(tasks, results, [signal]) for _ in range(num_consumers)]
    for w in consumers:
        w.start()
    print 'adding tasks'
    for i in iterator:
        tasks.put(Task(i, signal))
    for i in range(num_consumers):
        tasks.put(None)
    print 'collecting'
    for _ in range(len(iterator)):
        out.append(results.get())
        if _%100000 == 0:
            print _
    tasks.close()
    tasks.join_thread()
    print 'closing'
    for w in consumers:
        w.join()
    print time.time() - t

fitted, vocab = fit_model_ngram(it.text.str.lower().values, (1,NUM_GRAMS))
vi = dict(zip(vocab.values(), vocab.keys()))
del vocab

# compute cooccurrences
oc = np.array(fitted.sum(axis=0)).flatten()
oc = dict(zip(range(len(oc)), oc))
nk = pd.DataFrame({'nk': range(len(oc)), 'ok': sorted(oc.keys())})
nk['token'] = nk.ok.map(vi)
nk = dict(zip(nk.nk.values, nk.token.values))
fitted = fitted[:, sorted(oc.keys())]
oc = pd.DataFrame(oc.items(), columns=['token', 'oc'])
oc['token'] = oc.token.map(vi)
oc = dict(zip(oc.token.values, oc.oc.values))
del vi

# compute unigram scores
unigrams = filter_dict_by_numwords(oc, 1)
unigrams = pd.DataFrame({'token': unigrams.keys(), 'score': unigrams.values()})
unigrams = unigrams[unigrams.token.str.len()>2]

# score cooccurrences
coc = fitted.T * fitted
coc = coc.tocoo()
coc = pd.DataFrame(np.vstack([coc.row, coc.col, coc.data]).T, columns=['row', 'col', 'coc'])
#coc['score'] = map(lambda r,c,co: (1. * oc.get(nk[r]+' '+nk[c], 0))/(oc[nk[r]] * oc[nk[c]] * co), coc['row'], coc['col'], coc['coc'])
coc['score'] = map(lambda r,c,co: (1. * oc.get(nk[r]+' '+nk[c], 0))/(co), coc['row'], coc['col'], coc['coc'])
coc = coc[coc['score']>0]
coc['token'] = map(lambda r,c: nk[r]+' '+nk[c], coc['row'], coc['col'])
coc['num_tokens'] = coc.token.apply(lambda _: len(_.split()))

out = pd.DataFrame()
for n in coc.num_tokens.unique():
    cut = coc[coc['num_tokens']==n].copy()
    cut = cut[cut['score']>0.5] #np.percentile(cut.score, 50)]
    out = pd.concat([out, cut])
    print n

out = out[['token', 'score']]
out = pd.concat([out, unigrams])

cv = list(set(out.token.values))
import re
reg = re.compile('.*http.*')
rejects = set(filter(reg.match, cv))
reg = re.compile('^[0-9 ]+$')
rejects |= set(filter(reg.match, cv))
#reg = re.compile('^[0-9 ]{2,}')
#rejects |= set(filter(reg.match, cv))
cv = set(out.token.values) - STOP_WORDS - COLORS - rejects
cv = sorted(list(cv))
print len(cv)
cv = dict(zip(cv, range(len(cv))))
icv = dict(zip(cv.values(), cv.keys()))

fit, vocab = fit_model_ngram(it.text.str.lower().values, (1,NUM_GRAMS), cv)
final = fit.tocoo()
final = pd.DataFrame(np.vstack([final.row, final.col]).T, columns=['item_index', 'token'])
final['token'] = final['token'].map(icv)
final = final.merge(it)
f = final.groupby(['item_id', 'text'])['token'].apply(lambda _: '|'.join(sieve_tokens(_))).reset_index()
f.to_csv('../../extracted/final.csv500', index=None, sep='\t')

# tagging with brands
brands = pd.read_csv('../brands.csv', encoding='utf-8')
brands = set(brands.name.values) - STOP_WORDS
found = set()
for t in cv.keys():
    found.update(get_ngrams(t))

found = found|set(cv.keys())
found = found.intersection(brands)
final['valid'] = final.token.apply(lambda _: _ in found)
branded = final[final['valid']].copy()
branded = branded.groupby('item_id')['token'].apply(lambda _: '|'.join(sieve_tokens(_))).reset_index()
branded.columns=['item_id', 'brand']
#branded.to_csv('branded_trial1.csv', index=None, sep='\t')

m = f.merge(branded, how='left').fillna('')
m.to_csv('final_branded.csv2', index=None, sep='\t')

print 'computing and storing occurrences of each token'
occur = dict(zip(sorted(vi.values()), np.array(fitted.sum(axis=0)).flatten()))
occur = pd.DataFrame(occur.items(), columns=['token', 'occurrence'])
occur = occur[occur['occurrence']>OCC_THRESHOLD]
occur = occur[~occur.token.isin(['NA', 'NULL', 'nan'])]
occur.to_csv('occur.csv', index=None)
tokens = occur.token.values
del occur

print 'computing the conditional ngram score of each token'
condscore = []
parexec('occur', condscore, NUM_CORES, tokens)

print 'storing conditional scores'
condscore = pd.DataFrame(condscore, columns=['token', 'cs'])
condscore.to_csv('condscore.csv', index=None)
del condscore

print 'computing max ngram score for each token'
score = []
parexec('score', score, NUM_CORES, tokens)

print 'storing token scores'
score = pd.DataFrame(score, columns=['token', 'score'])
score.to_csv('score.csv', index=None)
score = dict(zip(score.token.values, score.score.values))

print 'choosing top tokens from each n-gram with score above 99th percentile'
chosen = dict()
for i in range(1,NUM_GRAMS+1):
    vf = filter_dict_by_numwords(score, i)
    chosen.update(filter_dict_by_value(vf, np.percentile(vf.values(), 95 if i==1 else 99)))
    print i

del score

