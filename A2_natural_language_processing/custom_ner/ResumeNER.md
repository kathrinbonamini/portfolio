# RESUME NAMED ENTITY RECOGNITION

Pedro Almodovar would say "HR al borde de un ataque de nervios". Your talent discovery group receives **hundreds of applications** every months: resumes, recommendation letters, third grade scoreboards, rivers of words.

They need some help from from their friends, aka data science team. Wouldn't it be nice (I know you read it by singing in your head) to have some sort of software that reads the docs and **extracts all relevant skills**? So that you can convert unstructured texts into ordered and sortable database? There we go with Custom Named Entity Recognition!

**Named Entity Recognition** models locate and label entities (like PERSON, ORGANIZATION, TIME, ...) into non-structured text.
My favourite NER library of all times is [**spaCy**](https://spacy.io/). Not only it provides state-of-the art models for all languages, extracting most common entities, but it also:
+ lets you train your own model on your own peculiar entities
+ offers a straight-to-the-point visualizer to test the results: [displacy](https://explosion.ai/demos/displacy)

Now, every story has a villain; our villain is **manual labeling**. Usual approach would involve:
+ gather something like 2-300 documents
+ manually identify all skills of interest from all these texts
+ train your own model, hoping for the best but expecting the worst
+ test with some sample
+ manually tag another 2-300 bloody documents to increase poor model performance

In my not requested opinion, manual tag is a fee that a junior needs to pay at least once in a lifetime, so that he/she can understand the importance of being creative and find escape routes; exactly my story.

When manual labeling is the only way possible, I strongly recommend using [**LabelStudio**](https://labelstud.io/): I spent hours on it, but it saved me weeks and joy to live.

In this case, I want to propose my personal escape route: **minimum viable list & incremental approach**.
+ gather something like 2-300 documents, since I am a fancy data scientist and not an HR guy, I will scrape the internet
+ get/write a list of most common skills usually referenced
+ automatically label the documents by tracking skills position
+ train my first model
+ test with some sample
+ enrich my skills list with missing ones and reiterate the process until I am satisfied.

It will take some time as well, but hours, not all the best years of our lives.


```python
import os
import random
import re
import json
import time
import requests
from datetime import datetime

import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch
from spacy.training import Example
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

import multiprocessing as mp

import warnings
```

### 1. Scrape the resumes


```python
resume_list = []
for page in range(1,6,1):
    for job_role in ['data scientist', 'software developer']:
        job_role = job_role.replace(' ', '+')
        listing_page = f'https://www.jobspider.com/job/resume-search-results.asp/words_{job_role}/searchtype_3/page_{page}'
        response = requests.get(listing_page).content
        soup = BeautifulSoup(response, 'html.parser')
        time.sleep(5)
        resume_list += soup.find_all('td', class_='StandardRow')
len(resume_list)
```




    1608




```python
# Each resume corresponds to 6 table columns, hence 6 elements
for el in resume_list[:6]:    
    print(str(el))
```

    <td align="center" class="StandardRow">1</td>
    <td align="center" class="StandardRow">3/10/2022</td>
    <td align="center" class="StandardRow">DATA SCIENTIST</td>
    <td align="center" class="StandardRow">Engineering</td>
    <td align="center" class="StandardRow" nowrap="">Richmond Hill, ON</td>
    <td align="center" class="StandardRow"><a href="/job/view-resume-83579.html"><img alt="View Ad" border="0" height="16" src="/job/images/Preview-16x16.png" width="16"/></a></td>
    


```python
# Use regex to parse resume href, then build resume url
resume_url_prefix = 'https://www.jobspider.com'
pattern = '<a href="(.*)">'

resume_url_list = []
for el in resume_list:
    el = str(el)
    match = re.search(pattern, el, re.DOTALL)
    if match:
        resume_url = resume_url_prefix + match.group(1)
        resume_url_list.append(resume_url)
print("Total number of resume urls: ", len(resume_url_list))
print("Sample resume url", resume_url_list[0])
```

    Total number of resume urls:  268
    Sample resume url https://www.jobspider.com/job/view-resume-83579.html
    


```python
# Get resumes
def segment_paragraph(whole_text, paragraph_name):
    paragraph_content = ''
    pattern = f'<b>{paragraph_name}:</b><br/><font color="#000000">(.*?)</font>'
    match = re.search(pattern, whole_text, re.DOTALL)
    if match:
        paragraph_content = clean_text(match.group(1))  
    return paragraph_content

def clean_text(text):
    text = text.replace('&amp;', '&')
    text = text.replace('\n', ' ')
    text = text.replace('<br>', ' ')
    text = text.replace('<br/>', ' ')
    text = text.replace('•', ' ')
    text = text.replace('»', ' ')
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text

def get_resume(resume_url):
    resume = {}
    try:
        response = requests.get(resume_url).content
        soup = BeautifulSoup(response, 'html.parser')

        # job title
        resume['Title'] = clean_text(soup.find('table', {'id':'Table3'}).text)

        # segmenting into: objective, experience, education, skills
        whole_text = str(soup)
        for paragraph_name in ['Objective', 'Experience', 'Education', 'Skills']:
            resume[paragraph_name] = segment_paragraph(whole_text, paragraph_name)

        resume['URL'] = resume_url
    except Exception as e:
        print(f"Cannot scrape {resume_url}: {e}")
        time.sleep(120)
    return resume

resumes = []
for i, resume_url in enumerate(resume_url_list):
    if i%15 == 0:
        print(f"Scraped {i+1} resumes")
    resumes.append(get_resume(resume_url))
    time.sleep(10)
resumes = pd.DataFrame(resumes)
print("Total number of resumes scraped: ", len(resumes))
resumes.head()
```

    Scraped 1 resumes
    Scraped 16 resumes
    Scraped 31 resumes
    Scraped 46 resumes
    Scraped 61 resumes
    Cannot scrape https://www.jobspider.com/job/view-resume-77319.html: HTTPSConnectionPool(host='www.jobspider.com', port=443): Max retries exceeded with url: /job/view-resume-77319.html (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1123)')))
    Scraped 76 resumes
    Scraped 91 resumes
    Scraped 106 resumes
    Scraped 121 resumes
    Scraped 136 resumes
    Scraped 151 resumes
    Cannot scrape https://www.jobspider.com/job/view-resume-50349.html: HTTPSConnectionPool(host='www.jobspider.com', port=443): Max retries exceeded with url: /job/view-resume-50349.html (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1123)')))
    Cannot scrape https://www.jobspider.com/job/view-resume-49814.html: HTTPSConnectionPool(host='www.jobspider.com', port=443): Max retries exceeded with url: /job/view-resume-49814.html (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1123)')))
    Scraped 166 resumes
    Scraped 181 resumes
    Scraped 196 resumes
    Scraped 211 resumes
    Scraped 226 resumes
    Scraped 241 resumes
    Scraped 256 resumes
    Total number of resumes scraped:  268
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Objective</th>
      <th>Experience</th>
      <th>Education</th>
      <th>Skills</th>
      <th>URL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Data Scientist Resume</td>
      <td>I believe that my skill and interested in Syst...</td>
      <td>Professional Experience Foreign Currency -Main...</td>
      <td>Data Science Certificate 2020-2021 University ...</td>
      <td>10+ years management experience as material pl...</td>
      <td>https://www.jobspider.com/job/view-resume-8357...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Data Scientist Resume</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>https://www.jobspider.com/job/view-resume-8270...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Data Scientist Resume</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>https://www.jobspider.com/job/view-resume-8270...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Data Science Manager Resume</td>
      <td>Innovative IT professional with extensive expe...</td>
      <td>PROFESSIONAL EXPERIENCE FlyLiz (Formerly Conse...</td>
      <td>EDUCATION Coursework toward Doctor of Philosop...</td>
      <td>SOFTWARE/APPLICATIONS:TensorFlow | ASP.net MVC...</td>
      <td>https://www.jobspider.com/job/view-resume-8263...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data Scientist / Data Analyst Resume</td>
      <td>Highly organized, innovative, and dedicated pr...</td>
      <td>J.P. Morgan Chase &amp; Co.  Chicago, IL Onboardi...</td>
      <td>Master of Science in Predictive Analytics, Nor...</td>
      <td>Business Operations Competitive Analysis Proce...</td>
      <td>https://www.jobspider.com/job/view-resume-8198...</td>
    </tr>
  </tbody>
</table>
</div>




```python
resumes.dropna(inplace=True)
resumes.to_csv('data/resumes_corpus.csv', sep='\t', index=False)
```


```python
print(f"Number of successfully scraped resumes: {len(resumes)}")
```

    Number of successfully scraped resumes: 265
    

### 2. Get the starting skillset
Look for online inspiration and put together a decent amount of skills to start training your own model. We are working with data scientist & software developers resumes, you can adapt to your own case (if you are hiring the principal dancer for Teatro alla Scala di Milano, skills might be a little different)
+ https://novoresume.com/career-blog/most-important-skills-to-put-on-your-resume
+ https://career.nichols.edu/blog/2021/08/25/top-skills-to-list-on-linkedin/
+ https://www.simplilearn.com/what-skills-do-i-need-to-become-a-data-scientist-article
+ https://www.simplilearn.com/best-programming-languages-start-learning-today-article
+ ...

Let's say you spend an hour and make a list as follows (mine has more than 700 skills so far):


```python
skills = open('skills.txt', 'r').readlines()
skills = [s.replace('\n', '') for s in skills]
skills = skills[1:]
for s in skills[:10]:
    print(s)
```

    BI
    BUSINESS INTELLIGENCE
    DASHBOARDING
    DASHBOARDS
    DATA REPORTING
    VISUAL ANALYTICS
    VISUAL REPORTING
    VISUAL REPORTS
    CRM
    RELATIONSHIP MANAGEMENT TOOLS
    


```python
resumes = pd.read_csv('data/resumes_corpus.csv', sep='\t')
```

### 3. Tag the dataset


```python
class Span:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label
        
        
def tag_skill(text, skill, entities_list):
    if isinstance(text, str):
        pattern = r'\b{}\b'.format(re.escape(skill))
        occurences = re.finditer(pattern, text, re.I)
        for occ in occurences:
            tup = (occ.span()[0], occ.span()[1], 'SKILL')
            entities_list.append(tup)
    return entities_list

def get_complete_token_annotations(text, entities_list):
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)
    new_entities_list = []
    for start, end, label in entities_list:
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        annot = tuple([span.start_char, span.end_char, label])
        new_entities_list.append(annot)
    return new_entities_list

def filter_overlapping_entities(entities_list):
    filtered_entities = []
    spans = [Span(tup[0], tup[1], tup[2]) for tup in entities_list]
    filtered_spans = spacy.util.filter_spans(spans)
    for span in filtered_spans:
        tup = tuple([span.start, span.end, span.label])
        filtered_entities.append(tup)

    return {'entities': filtered_entities}


def process_single_job(i, text, approved_skills, N_docs):
    if i % 50 == 0:
        print("Files to process: {}".format(N_docs-i))
    if not isinstance(text, str):
        return None
    entities_list = []
    for skill in approved_skills:
        entities_list = tag_skill(text, skill, entities_list)
        
    entities_list = get_complete_token_annotations(text, entities_list)
    annotations = filter_overlapping_entities(entities_list)
    sample = (text, annotations)
    return sample


def tag_dataset(approved_skills):
    dataset = []

    n = len(resumes)
    resumes['text'] = resumes['Objective'] + resumes['Experience'] + resumes['Skills']
    texts = list(resumes['text'].values)
    for i, t in enumerate(texts):
        sample = process_single_job(i, t, approved_skills, n)
        if sample:
            dataset.append(sample)

    return dataset

def split_train_test(dataset):
    n = 10
    random.shuffle(dataset)
    testset = dataset[:n]
    trainset = dataset[n:]
    return trainset, testset

def reformat_dataset(dataset):
    nlp = spacy.blank('en')
    examples = []
    for text, annotations in dataset:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    return examples
```


```python
# tag all texts --> multiprocessing
dataset = tag_dataset(skills)

# split train test
trainset, testset = split_train_test(dataset)

# create spacy3 dataset
trainset = reformat_dataset(trainset)

```

    Files to process: 265
    Files to process: 215
    Files to process: 165
    Files to process: 115
    Files to process: 65
    Files to process: 15
    

### 3. Train & Validate


```python
def train(model_path, examples, epochs=10):
    # Load existing model to fine tune if exists, otherwise load blank
    if os.path.exists(model_path):
        nlp = spacy.load(model_path)
        print("Loaded model '%s'" % model_path)
        to_be_initialized = False
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
        to_be_initialized = True

    # create the built-in pipeline components and add them to the pipeline
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe('ner', last=True)
    ner = nlp.get_pipe("ner")

    # add labels
    ner.add_label("SKILL")

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

   # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # initialize only for new models, not for retrain
        if to_be_initialized:
            nlp.initialize(lambda: examples)
        min_loss = -1
        for i in range(epochs):
            random.shuffle(examples)
            losses = {}
            for batch in minibatch(examples, size=64):
                nlp.update(batch, drop=0.5, losses=losses)

            if (min_loss == -1) or (losses["ner"] < min_loss):
                min_loss = losses["ner"]
                nlp.to_disk(model_path)
                print("Epoch {}, Losses {}".format(i, losses))
    return nlp

def validate(testset, nlp):
    for text, annotations in testset:
        doc = nlp(text)
        print("Predicted: {}".format(", ".join([ent.text for ent in doc.ents])))
        print("Expected: {}".format(", ".join([text[start:end] for start, end, _ in annotations['entities']])))
        print("="*100)
```


```python
# retrain ner model
modelpath = 'model'
nlp = train(modelpath, trainset, 50)
validate(testset, nlp)
```

    Loaded model 'model'
    Epoch 0, Losses {'ner': 6869.152222156525}
    Checkpoint saved!
    Epoch 1, Losses {'ner': 6219.446724295616}
    Checkpoint saved!
    Epoch 2, Losses {'ner': 5699.395666003227}
    Checkpoint saved!
    Epoch 3, Losses {'ner': 5083.990793228149}
    Checkpoint saved!
    Epoch 4, Losses {'ner': 4768.362592697144}
    Checkpoint saved!
    Epoch 5, Losses {'ner': 4184.893102169037}
    Checkpoint saved!
    Epoch 6, Losses {'ner': 3906.537247776985}
    Checkpoint saved!
    Epoch 7, Losses {'ner': 3674.5477122291923}
    Checkpoint saved!
    Epoch 8, Losses {'ner': 3239.5830332264304}
    Checkpoint saved!
    Epoch 9, Losses {'ner': 3211.525919497013}
    Checkpoint saved!
    Epoch 10, Losses {'ner': 2885.967892497778}
    Checkpoint saved!
    Epoch 11, Losses {'ner': 2769.530272036791}
    Checkpoint saved!
    Epoch 12, Losses {'ner': 2605.3046573996544}
    Checkpoint saved!
    Epoch 13, Losses {'ner': 2532.2996921241283}
    Checkpoint saved!
    Epoch 14, Losses {'ner': 2340.875946752727}
    Checkpoint saved!
    Epoch 15, Losses {'ner': 2362.375263363123}
    Epoch 16, Losses {'ner': 2375.121852643788}
    Epoch 17, Losses {'ner': 2302.014750123024}
    Checkpoint saved!
    Epoch 18, Losses {'ner': 2040.4346980564296}
    Checkpoint saved!
    Epoch 19, Losses {'ner': 1926.8231866061687}
    Checkpoint saved!
    Epoch 20, Losses {'ner': 1839.8659416073933}
    Checkpoint saved!
    Epoch 21, Losses {'ner': 1785.4338593482971}
    Checkpoint saved!
    Epoch 22, Losses {'ner': 1692.0062393695116}
    Checkpoint saved!
    Epoch 23, Losses {'ner': 1449.3121076636016}
    Checkpoint saved!
    Epoch 24, Losses {'ner': 1381.6330345980823}
    Checkpoint saved!
    Epoch 25, Losses {'ner': 1330.8420183584094}
    Checkpoint saved!
    Epoch 26, Losses {'ner': 1295.0078788232058}
    Checkpoint saved!
    Epoch 27, Losses {'ner': 1300.9110557064414}
    Epoch 28, Losses {'ner': 1185.2791234934703}
    Checkpoint saved!
    Epoch 29, Losses {'ner': 1090.9097124161199}
    Checkpoint saved!
    Epoch 30, Losses {'ner': 1014.7743434824515}
    Checkpoint saved!
    Epoch 31, Losses {'ner': 973.4412163924426}
    Checkpoint saved!
    Epoch 32, Losses {'ner': 1004.1525040594861}
    Epoch 33, Losses {'ner': 1014.4852016530931}
    Epoch 34, Losses {'ner': 878.5016890512779}
    Checkpoint saved!
    Epoch 35, Losses {'ner': 868.2451607307885}
    Checkpoint saved!
    Epoch 36, Losses {'ner': 937.1425466910005}
    Epoch 37, Losses {'ner': 835.27580251568}
    Checkpoint saved!
    Epoch 38, Losses {'ner': 902.4641309333965}
    Epoch 39, Losses {'ner': 883.2197586877737}
    Epoch 40, Losses {'ner': 759.1573268948123}
    Checkpoint saved!
    Epoch 41, Losses {'ner': 734.198234513402}
    Checkpoint saved!
    Epoch 42, Losses {'ner': 722.5213028270518}
    Checkpoint saved!
    Epoch 43, Losses {'ner': 669.8364305291325}
    Checkpoint saved!
    Epoch 44, Losses {'ner': 787.3951246414508}
    Epoch 45, Losses {'ner': 634.517794879619}
    Checkpoint saved!
    Epoch 46, Losses {'ner': 687.0373506112956}
    Epoch 47, Losses {'ner': 723.7901685080433}
    Epoch 48, Losses {'ner': 649.5439371973043}
    Epoch 49, Losses {'ner': 572.9764542681805}
    Checkpoint saved!
    Predicted: software engineering, HTML, JavaScript, statistics, Eclipse, Java, XML, EDM, EDM, XML, XML, xml, Visual Studio, EDM software, JavaScript, Visual Studio, Visual Studio, Microsoft Office, Java, HTML, JavaScript, XML, Microsoft Office, Visual Studio, Visio, Git, Databases, SQL Server
    Expected: software engineering, HTML, JavaScript, statistics, Eclipse, Java, XML, Document Management, XML, XML, xml, Visual Studio, JavaScript, Visual Studio, Visual Studio, Microsoft Office, Antivirus, Java, HTML, JavaScript, XML, Microsoft Office, Visual Studio, Visio, Git, Databases, SQL Server
    ====================================================================================================
    Predicted: SDLC, Data Analysis, ASP.NET, VB.NET, JavaScript, JQuery, OOP, UML, WCF, SQL Server, SQL, Oracle, SQL, OOP, SQL, HTML, JavaScript, JQuery, UI, Asp.net, WCF, SQL server, SQL, Visual studio.net, ASP.Net, WCF, JavaScript, JQuery, XML, Visual studio, SQL, SSIS, SSRS, UML, ASP.net, SQL, SQL, SQL, Visual Studio, ASP.Net, ADO.Net, Sql, ASP.NET, VB.NET, ASP.NET, VB.NET, Visual Studio, Java, UI, SQL Server, VB.NET, Java, Ruby, ASP.NET, WCF, ADO.NET, JavaScript, JQuery, HTML, SQL Server, Oracle, SQL, UML, XML, XML, Web Servers, SSRS, Visual Studio, SQL, SQL, TOAD, SQL, Unix
    Expected: SDLC, Data Analysis, ASP.NET, VB.NET, JavaScript, JQuery, Object Oriented Programming, OOP, UML, WCF, SQL Server, SQL, Oracle, SQL, OOP, SQL, C#2008, HTML, JavaScript, JQuery, UI, Asp.net, WCF, SQL server, SQL, Visual studio.net, ASP.Net, WCF, JavaScript, JQuery, XML, Visual studio, SQL, SSIS, SSRS, UML, ASP.net, SQL, SQL, SQL, Visual Studio, ASP.Net, ADO.Net, Sql, ASP.NET, VB.NET, ASP.NET, VB.NET, Visual Studio, Java, sql server, UI, sql server, SQL Server, VB.NET, Java, Ruby, ASP.NET, WCF, ADO.NET, JavaScript, JQuery, HTML, SQL Server, Oracle, SQL, UML, XML, XML, Web Servers, SSRS, Visual Studio, SQL, SQL, TOAD, SQL, Unix
    ====================================================================================================
    Predicted: Microsoft Word, Excel
    Expected: track and trace, software troubleshooting, Microsoft Word, Excel
    ====================================================================================================
    Predicted: data analysis, software development, software development lifecycle, SDLC, SAP, SAP, databases, SAP, SAP, SAP, Visio
    Expected: data analysis, software development, software development lifecycle, SDLC, SAP, SAP, databases, SAP, SAP, SAP, Visio
    ====================================================================================================
    Predicted: Oracle, SQL, T-SQL, Oracle, Java, SDLC, SQL, Java, software development, SQL Server, SDLC, OOP, Oracle, Server development, database development, Oracle, SQL, SQL, T-SQL, SQL Server, SQL, Java, Java, Application Development, Eclipse, JSP, JavaScript, XML, JSP Server, SQL
    Expected: Oracle, SQL, T-SQL, Oracle, Java, SDLC, SQL, Java, software development, SQL Server, SDLC, develop software, OOP, Oracle, Oracle, SQL, SQL, T-SQL, SQL Server, SQL, Java, Java, Application Development, Eclipse, JSP, JavaScript, XML, JSP, SQL
    ====================================================================================================
    Predicted: software development, UNIX, Web development, Software testing, OOP, Java, SQL Server, HTML, XML, Javascript, SQL Server, Oracle, databases, HTML, XML, Javascript, SQL Server, Web development, Software testing, OOP, Java, SQL Server, Oracle, HTML, XML, Javascript, SQL Server
    Expected: software development, UNIX, Web development, Software testing, Object Oriented Programming, OOP, Object Oriented Programming, Java, SQL Server, HTML, XML, Javascript, SQL Server, Oracle, databases, HTML, XML, Javascript, SQL Server, Web development, Software testing, Object Oriented Programming, OOP, Object Oriented Programming, Java, SQL Server, Oracle, HTML, XML, Javascript, SQL Server
    ====================================================================================================
    Predicted: Excel, software development, SQL, Oracle, data modeling, data analysis, UNIX, Data Analysis
    Expected: Excel, software development, SQL, Oracle, data modeling, data analysis, UNIX, Data Analysis
    ====================================================================================================
    Predicted: Cybersecurity, SharePoint, Confluence, MS Word, PowerPoint, Excel, SharePoint, Visio, Spark
    Expected: Cybersecurity, SharePoint, Confluence, MS Word, PowerPoint, Excel, SharePoint, Visio, Spark
    ====================================================================================================
    Predicted: Computer Science, software development
    Expected: Computer Science, software development
    ====================================================================================================
    Predicted: UI, UX, MS Office, SASS
    Expected: UI, UX, MS Office, SASS
    ====================================================================================================
    


```python

```
