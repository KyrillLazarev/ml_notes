import os
import json
import pprint
import pandas as pd
##########################################
for fname in os.listdir(os.getcwd() + '\\data')[:1]:
    with open(os.getcwd() + '\\data\\' + fname, 'r', encoding='utf-8') as file:
        tmp_json = json.load(file)
        pprint.PrettyPrinter().pprint(tmp_json)
####################################################################################
def all_keys(dict_obj, indent=0, parent=''):
    ''' This function generates all keys of
        a nested dictionary.
    '''
    # Iterate over all keys of the dictionary
    for key , value in dict_obj.items():
        yield parent + ('-') + key
        # If value is of dictionary type then yield all keys
        # in that nested dictionary
        if isinstance(value, dict):
            for k in all_keys(value, indent=indent+1, parent = key):
                yield parent + ('-') + k
        elif isinstance(value, list):
            for t in value:
                for key_lst in all_keys(t, indent=indent+1, parent=key):
                    yield parent + ('-') + key_lst

# Iterate over all keys of a nested dictionary
# and print them one by one.

every_key = []
for fname in os.listdir(os.getcwd() + '\\data')[:]:
    with open(os.getcwd() + '\\data\\' + fname, 'r', encoding='utf-8') as file:
        tmp_json = json.loads(*file.readlines())
        for i in all_keys(tmp_json):
            if i not in every_key:
                every_key.append(i)

print(*sorted(every_key), sep='\n')
####################################################################################
taken = [
    '-feed-payload-body-text',
    '-reason-category-city_object-id',
    '-reason-category-id',
    '-reason-category-name',
    "-reason-id",
    "-reason-name",
    "-sidebar-full_address",
    "-sidebar-building",
    "-sidebar-latitude",
    "-sidebar-longitude",
    "-sidebar-municipality-name",
    "-sidebar-district-id",
    "-sidebar-responsible-data-executor-organization_name",
    "-sidebar-nearest_building-id",
    "-sidebar-nearest_building-latitude",
    "-sidebar-nearest_building-longitude",
    "-sidebar-author_name"
]
####################################################################################
def get_by_keys(keys_arr, obj):
    if obj == None:
        return None
    if len(keys_arr) == 1:
        return obj.get(keys_arr[0])
    else:
        return get_by_keys(keys_arr[1:], obj.get(keys_arr[0]))
####################################################################################
col_names = [" ".join(i.split('-')[-2:]) for i in taken]
res_ids=[]
res = list()

for fname in os.listdir(os.getcwd() + '\\data')[:]:
    with open(os.getcwd() + '\\data\\' + fname, 'r', encoding='utf-8') as file:
        tmp_json = json.loads(*file.readlines())
        res_ids.append(tmp_json['id'])
        res.append(dict())
        for key in taken:
            key = key.split('-')[1:]
            if key[0] == 'feed':
                txt = ''
                tmp_res = get_by_keys(key[1:3], get_by_keys([key[0]], tmp_json)[-1])
                if tmp_res:
                    for block in tmp_res:
                        if block['typeof'] == 1:
                            txt += block['text']
                    res[-1][" ".join(key[-2:])] = txt.strip()
                else:
                    res[-1][" ".join(key[-2:])] = None
            else:
                res[-1][' '.join(key[-2:])] = get_by_keys(key, tmp_json)
    cnt += 1


df=pd.DataFrame(res)
df
####################################################################################
df.isna().sum()
####################################################################################
to_del = [
    'sidebar building',
    'nearest_building id',
    'nearest_building latitude',
    'nearest_building longitude',
]

df = df.drop(columns=to_del)
df
####################################################################################
categories = dict()
reasons = dict()
cnt = 0
for i in df['category name']:
    categories[int(df['category id'][cnt])] = i
    cnt += 1

cnt = 0
for i in df['reason name']:
    reasons[int(df['reason id'][cnt])] = i
    cnt += 1

with open('categories.json', 'w') as file:
    json.dump(categories, file)

with open('reasons.json', 'w') as file:
    json.dump(reasons, file)
####################################################################################
df = df.drop(columns=['reason name', 'category name'])
df
####################################################################################
df.to_csv('C3M1.csv')
####################################################################################
####################################################################################
import os
import json
import pprint
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import pymorphy2
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
####################################################################################
dataset = pd.read_csv('C3M1.csv', index_col=0)
dataset
####################################################################################
to_drop_empty = []
for i in range(len(dataset)):
    if type(dataset['body text'][i]) == float:
        to_drop_empty.append(i)

dataset = dataset.drop(to_drop_empty)
dataset
####################################################################################
dataset.isna().sum()
####################################################################################
dataset = dataset.drop(columns=['sidebar full_address'])


def preprocess(text, to_print=True):
    if to_print:
        print(text)
    text = text.lower()
    text = re.sub(re.compile("[:punct:\n]"), '', text)
    if to_print:
        print()
        print(text)
    text = re.sub('[^а-яА-Я0-9ёЁ]+', ' ', text)
    text = ''.join(list(text))
    if to_print:
        print()
        print(text)

    return text


preprocess(dataset['body text'][0] + "\n\n\n\n")
####################################################################################
dataset.to_csv("tmp.csv")
dataset = pd.read_csv("tmp.csv")
dataset = dataset.drop(columns = "Unnamed: 0")
dataset
####################################################################################
for i in range(len(dataset)):
    try:
        dataset['body text'][i] = preprocess(dataset['body text'][i], to_print=False)
        dataset['municipality name'][i] = preprocess(dataset['municipality name'][i], to_print=False)
        if type(dataset['executor organization_name'][i]) == str:
            dataset['executor organization_name'][i] = preprocess(dataset['executor organization_name'][i], to_print=False)
    except KeyError:
        pass

dataset
####################################################################################
for i in range(len(dataset)):
    if type(dataset['executor organization_name'][i]) == float:
        dataset['executor organization_name'][i] = ''
####################################################################################
for i in ['municipality name', 'executor organization_name']:
    le = LabelEncoder()
    res = le.fit_transform(dataset[i])
    dataset.insert(loc = len(dataset.columns), column = i+' id', value=res, allow_duplicates =True)
dataset
####################################################################################
morph = pymorphy2.MorphAnalyzer()
for i in range(len(dataset))[:]:
    res = []
    for word in dataset["body text"][i].split(' '):
        res.append(morph.parse(word)[0].normal_form)
    dataset["body text"][i] = res
dataset
####################################################################################
stop_words = set(stopwords.words('russian'))
for i in range(len(dataset))[:]:
    res = [word for word in dataset['body text'][i] if word not in stop_words]
    dataset["body text"][i] = res
dataset
####################################################################################
sentences = []
for i in dataset['body text']:
    sentences.append(i)

bmodel = Phrases(sentences)
bsentences = []
for sentence in sentences:
    bsentences.append(bmodel[sentence])

tmodel = Phrases(bsentences)
tsentences = []
for sentence in bsentences:
    tsentences.append(tmodel[sentence])

for i in range(len(tsentences)):
    dataset['body text'][i] = set(tsentences[i]) | set(bsentences[i]) | set(sentences[i])
dataset
####################################################################################
corpus = ''
for i in sentences:
    corpus += ' '.join(i) + '. '

tfidf = TfidfVectorizer(ngram_range=(1, 3))
tfs = tfidf.fit_transform(corpus.split(' '))
feature_names = np.array(tfidf.get_feature_names())

with open('tfidf.pickle', 'wb') as file:
    pickle.dump(tfidf, file)

n = 3

# dataset.insert(loc = len(dataset.columns), column = 'keyword_1', value = [''] * len(dataset))
# dataset.insert(loc = len(dataset.columns), column = 'keyword_2', value = [''] * len(dataset))
# dataset.insert(loc = len(dataset.columns), column = 'keyword_3', value = [''] * len(dataset))

"""
#cnt = 0
#for i in dataset['body text'][:]:
#    if cnt % 5000 == 0:
#        print(cnt)

    response = tfidf.transform(' '.join(i).split(' '))
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = feature_names[tfidf_sorting][:n]
    dataset['keyword_1'][cnt] = top_n[0]
    dataset['keyword_2'][cnt] = top_n[1]
    dataset['keyword_3'][cnt] = top_n[2]
    cnt += 1


dataset
"""
####################################################################################
dataset.to_csv('preprocessed.csv')
####################################################################################
####################################################################################
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas_profiling
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn.cluster
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
####################################################################################
file = pd.read_csv('preprocessed.csv', index_col = 0)

file
####################################################################################
pandas_profiling.ProfileReport(file)
####################################################################################
file.describe()
####################################################################################
corpus = ''
import random

for i in file['body text']:
    i = i.replace('\'', '')
    i = i.split(', ')
    i = i[1:]
    # Воспользуемся не всем корпусом текстов, а только 100 случайных
    tmp = random.randint(1, 100)
    if tmp % 100 <= 10:
        try:

            i[-1] = i[-1][:-1]
        except IndexError:
            pass
        corpus += ' ' + ' '.join(i) + '.'


tfidf = TfidfVectorizer(ngram_range=(1,3))
tfs =tfidf.fit_transform(corpus.split('. '))

tfs = tfs.todense()

pca = PCA(n_components=2).fit(tfs)
data2D = pca.transform(tfs)
plt.scatter(data2D[:,0], data2D[:,1])
####################################################################################
fig, axs = plt.subplots(4, 1, figsize = (20,20))

axs[0].scatter(file['reason id'], file['category id'], alpha=0.01)
axs[0].set_xlabel("Причина обращения")
axs[0].set_ylabel("Категория обращения")
axs[1].scatter(file['city_object id'], file['category id'], alpha=0.01)
axs[1].set_xlabel("Объект города")
axs[1].set_ylabel("Категория обращения")
axs[2].scatter(file['district id'], file['category id'], alpha=0.01)
axs[2].set_xlabel("Район города")
axs[2].set_ylabel("Категория обращения")
axs[3].scatter(file['executor organization_name id'], file['category id'], alpha=0.01)
axs[3].set_xlabel("Регулирующая организация")
axs[3].set_ylabel("Категория обращения")
####################################################################################
import json

form_sentences = []
file.astype({'body text': 'object'})
for i in file['body text'][:]:
    i = i.replace('\'', '')
    i = i.split(', ')
    i = i[1:]
    try:

        i[-1] = i[-1][:-1]
    except IndexError:
        pass
    form_sentences.append([' '.join(i)])

model = Word2Vec(sentences=sent, vector_size=50, window=3, min_count=1, workers=4)
####################################################################################
model.save("word2vec.model")
words = []
sent = []
for i in form_sentences:
    sent.append(i.split(' '))
for i in range(len(sent)):
    try:
        sent[i].remove('')
    except ValueError:
        pass
sent
####################################################################################
id2word = corpora.Dictionary(sent)
texts = form_sentences
corpus = [id2word.doc2bow(texts)]
# Это была подготовка данных для алгоритма
# Создаем модель данных

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=len(set(file['category id'])),
                                           random_state=42,
                                           update_every=1,
                                           chunksize=1000,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
####################################################################################
sorted(lda_model.print_topics(), key=lambda x: x[0])
####################################################################################
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds')
vis
####################################################################################
kmeans_cluster = sklearn.cluster.KMeans(n_clusters=len(set(file['category id'])))
spectral_cluster = sklearn.cluster.SpectralClustering(n_clusters=len(set(file['category id'])))
dbscan = sklearn.cluster.DBSCAN()

to_cluster = file.copy()
y = to_cluster['category id']
x = to_cluster.drop(columns='category id')
####################################################################################
pca2 = PCA(1)
cols = []
for i in x.columns:
    if x.dtypes[i] == 'object':
        cols.append(i)
print(cols)
x = x.drop(columns = cols)
x = x[:10000]
y = y[:10000]
####################################################################################
y_k = kmeans_cluster.fit_predict(pca2.fit_transform(x))
y_s = spectral_cluster.fit_predict(pca2.fit_transform(x))
y_d = dbscan.fit_predict(pca2.fit_transform(x))
####################################################################################
print(f"Metric for KMeans: {sklearn.metrics.mutual_info_score(y, y_k)}")
print(f"Metric for DBSCAN: {sklearn.metrics.mutual_info_score(y, y_d)}")
print(f"Metric for Spectral: {sklearn.metrics.mutual_info_score(y, y_s)}")
####################################################################################
sc = sklearn.cluster.SpectralClustering(n_clusters=len(set(file['category id'])),eigen_solver='arpack')
print(f"Metric for Spectral arpack: {sklearn.metrics.mutual_info_score(y, sc.fit_predict(x)}")
####################################################################################
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas_profiling
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn.cluster
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle
####################################################################################
file = pd.read_csv('preprocessed.csv', index_col = 0)
file
####################################################################################
model = gensim.models.Word2Vec.load('word2vec.model')
model.wv.most_similar('реклама',topn=10)
####################################################################################
X_coded = file.copy()
for i in ['keyword_1', 'keyword_2', 'keyword_3']:
    for j in range(0, len(X_coded)):
        try:
            tmp = model.wv[X[i][j]]
            tmp = np.mean(tmp)
        except KeyError:
            tmp = 0
        print
        X_coded[i][j] = str(tmp)
X_coded
####################################################################################
X_coded = X_coded.astype({'keyword_1': 'float64',
                         'keyword_2': 'float64',
                         'keyword_3': 'float64'})
X_coded
####################################################################################
y = X_coded['category id'].copy()
X = X_coded.drop(columns='category id').copy()
X = X.drop(columns=['body text', 'municipality name', 'executor organization_name', 'sidebar author_name'])
X
####################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
####################################################################################
grid = {
    'C': [1*(10**i) for i in range(-10, 0)],
}

svc = LinearSVC(dual=False, random_state=42)

svc_best = GridSearchCV(svc, grid, verbose=3, n_jobs = -1, cv = 3)
svc_best.fit(X_train, y_train)
####################################################################################
print(f'Balanced accuracy score for the best SVC is {metrics.balanced_accuracy_score(y_test, svc_best.predict(X_test))}')
print(f'F1 score for the best SVC is {metrics.f1_score(y_test, svc_best.predict(X_test), average="macro")}')
####################################################################################
grid = {
    'var_smoothing': [i*1e-9 for i in range(1, 1001)]
}

gnb = GaussianNB()
gnb_best_f1 = GridSearchCV(gnb, grid, verbose=3, n_jobs = -1, cv = 3)
gnb_best_f1.fit(X_train, y_train)

####################################################################################
print(f'Balanced accuracy score for the best GaussianNB is {metrics.balanced_accuracy_score(y_test, gnb_best_f1.predict(X_test))}')
print(f'F1 score for the best GaussianNB is {metrics.f1_score(y_test, gnb_best_f1.predict(X_test), average="macro")}')
####################################################################################
grid = {
    'n_estimators': [i for i in range(300, 400, 20)],
    'criterion': ['gini', 'entropy'],
}

rfc = RandomForestClassifier(random_state=42)

rfc_best = GridSearchCV(rfc, grid, verbose=3, n_jobs = -1, cv = 2)
rfc_best.fit(X_train, y_train)
####################################################################################
GridSearchCV(cv=2, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,
             param_grid={'criterion': ['gini', 'entropy'],
                         'n_estimators': [300, 320, 340, 360, 380]},
             verbose=3)
####################################################################################
print(f'Balanced accuracy score for the best RandomForestClassifier is {metrics.balanced_accuracy_score(y_test, rfc_best.predict(X_test))}')
print(f'F1 score for the best RandomForestClassifier is {metrics.f1_score(y_test, rfc_best.predict(X_test), average="macro")}')
####################################################################################
with open('rfcdata.pickle', 'wb') as file:
    pickle.dump(rfc_best, file)
####################################################################################
