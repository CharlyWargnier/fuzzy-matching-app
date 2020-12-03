
import streamlit as st

from streamlit_echarts import st_echarts
from pyecharts import options as opts

from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd

import base64
import csv


#import streamlit as st
#from numpy.core.fromnumeric import size
#from urllib.parse import urlparse
#import plotly.express as px
#import matplotlib.pyplot as plt
#from PIL import Image
#import os
#import json
#import time
#from pyecharts.charts import Tree
#st.text_area("label", value='asdasdasd, asddas, asdasd', height=None, max_chars=None, key=None)


#region Size of layout  ############################################################

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True, 
    )

_max_width_()

#endregion
c30, c31, c32 = st.beta_columns(3)

with c30:
#    st.title("SEO Forecast App")
#    with c30:
    st.image('logo.png', width = 425)


    #st.image('StreamSuggestLogo.png', width = 325)
    #st.image('GoogleButton.png', width = 325)

    st.header('')

with c32:
  st.header('')
  st.header('')
  st.markdown('###### Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://twitter.com/DataChaz) &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)')

with st.beta_expander("ℹ️ - To-do's ", expanded=False):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
-   Deploy the app!!!!!!!!
	    """)


with st.beta_expander("ℹ️ - Fixed ", expanded=False):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   create a list for low scores - otherwise won't appear in charts
-   remove index in last table
-   biggest issue: created lists don't add up properly and incorrect reading in charts - paste URLs to find out    
-   check -> https://colab.research.google.com/drive/1GZKKPtOCjjic5tbnr9mPzFKGMHnzrEvp#scrollTo=enTISRhUvSK1 
-   issue with labels - invert toekn sort and set!
-   put medium scores on the left of the chart, to be visible
-   Add duplicate in Add search box forain token columns, otherwise biased!

	    """)

with st.beta_expander("ℹ️ - Maybe ", expanded=False):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   add one more ratio
-   box 1 and 2 - Add search box - Needs to do 'or' not 'and'
-   Add score class in dupe columns

	    """)

with st.beta_expander("ℹ️ - About this app ", expanded=False):
  st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
-   You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast 😵 
-   The tool is in Beta. Feedback & bug spotting are welcome. [DMs are open!](https://twitter.com/DataChaz)
-   This app is free. If it's useful to you, you can [buy me a coffee](https://www.buymeacoffee.com/cwar05) to support my work! 🙏

	    """)

c29, c30 = st.beta_columns(2)

with c29:

    with st.beta_expander("ℹ️ - Keywords to add in box 1", expanded=False):
    #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

        st.write("""  
            
        -   fuzz bear
        -   fuzz bears
        -   fuzzy bear
        -   fuzzy bear

                """)

with c30:

    with st.beta_expander("ℹ️ - Keywords to add in box 2", expanded=False):
    #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

        st.write("""  
            
        -   fuzz bear
        -   fuzz bearss
        -   fuzzy bearssss
        -   bear fuzzy 

                """)

c29, c30 = st.beta_columns(2)

with c29:

    with st.beta_expander("ℹ️ - URLs to add in box 1", expanded=False):
    #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

        st.write("""  
            
        -   https://www.manh.com/extended-enterprise/services
        -   http://www.manh.com/pt-br/politica-de-privacidade
        -   http://www.manh.com/es/politica-de-privacidad
        -   http://www.manh.com/nl-nl/gebruiksvoorwaarden
        -   http://www.manh.com/platforms/manhattan-scale
        -   https://www.manh.com/de-de/resources/articles
        -   http://www.manh.com/en-gb/privacy-policy
        -   http://www.manh.com/en-au/privacy-policy
        -   http://www.manh.com/nl-nl/privacybeleid
        -   http://www.manh.com/es/terminos-de-uso
        -   http://www.manh.com/en-au/terms-of-use
        -   http://www.manh.com/en-gb/terms-of-use
        -   http://www.manh.com/privacy-policyIf
        -   http://www.manh.com/en-au/contact-us
        -   http://www.manh.com/en-gb/contact-us
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/point-of-sale
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   http://www.manh.com/privacy-policy
        -   https://www.manh.com/es-es/nowhere
        -   https://www.manh.com/ja-jp/nowhere
        -   https://www.manh.com/en-au/nowhere
        -   https://www.manh.com/zh-cn/nowhere
        -   https://www.manh.com/pt-br/nowhere
        -   http://www.manh.com/es/contactenos
        -   https://www.manh.com/de-de/nowhere
        -   https://www.manh.com/fr-fr/nowhere
        -   https://www.manh.com/en-nl/nowhere
        -   https://www.manh.com/nl-nl/nowhere
        -   http://www.manh.com/nl-nl/contact
        -   https://www.manh.com/es/nowhere
        -   http://www.manh.com/active/omni
        -   http://www.manh.com/contact-us
        -   https://www.manh.com/exchanges
        -   https://www.manh.com/exchange
        -   http://www.manh.com/exchange
        -   https://www.manh.com/en-gb/
        -   http://www.manh.com/en-gb/
        -   http://www.manh.com/zh-cn/
        -   http://www.manh.com/es-pa
        -   http://www.manh.com/nl-nl
        -   http://www.manh.com/en-au
        -   http://www.manh.com/en-ae
        -   http://www.manh.com/th-th
        -   http://www.manh.com/pt-br
        -   http://www.manh.com/is-is
        -   http://www.manh.com/zh-cn
        -   http://www.manh.com/ms-my
        -   http://www.manh.com/en-hk
        -   http://www.manh.com/en-in
        -   http://www.manh.com/ro-ro
        -   http://www.manh.com/pl-pl
        -   http://www.manh.com/de-de
        -   http://www.manh.com/en-nz
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/ja-jp
        -   http://www.manh.com/pt-pt
        -   http://www.manh.com/nl-be
        -   http://www.manh.com/it-it
        -   http://www.manh.com/en-nl
        -   http://www.manh.com/en-za
        -   http://www.manh.com/no-no
        -   http://www.manh.com/en-sg
        -   http://www.manh.com/fi-fi
        -   http://www.manh.com/es-es
        -   http://www.manh.com/es
        -   http://manh.com/es

                """)


with c30:

    with st.beta_expander("ℹ️ - URLs to add in box 2", expanded=False):
    #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

        st.write("""  
            
        -   https://www.manh.com/products/extended-enterprise/services
        -   https://www.manh.com/pt-br/politica-de-privacidade
        -   https://www.manh.com/es/politica-de-privacidad
        -   https://www.manh.com/nl-nl/gebruiksvoorwaarden
        -   https://www.manh.com/platforms/manhattan-scale
        -   https://www.manh.com/de-de/quellen/artikel
        -   https://www.manh.com/en-gb/privacy-policy
        -   https://www.manh.com/en-au/privacy-policy
        -   https://www.manh.com/nl-nl/privacybeleid
        -   https://www.manh.com/es/terminos-de-uso
        -   https://www.manh.com/en-au/terms-of-use
        -   https://www.manh.com/en-gb/terms-of-use
        -   https://www.manh.com/privacy-policyIf
        -   https://www.manh.com/en-au/contact-us
        -   https://www.manh.com/en-gb/contact-us
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/products/point-of-sale
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/en-gb/nowhere
        -   https://www.manh.com/privacy-policy
        -   https://www.manh.com/es-es
        -   https://www.manh.com/ja-jp
        -   https://www.manh.com/en-au
        -   https://www.manh.com/zh-cn
        -   https://www.manh.com/pt-br
        -   https://www.manh.com/es/contactenos
        -   https://www.manh.com/de-de
        -   https://www.manh.com/fr-fr
        -   https://www.manh.com/en-nl
        -   https://www.manh.com/nl-nl
        -   https://www.manh.com/nl-nl/contact
        -   https://www.manh.com/es
        -   https://www.manh.com/active/omni
        -   https://www.manh.com/contact-us
        -   http://info.manh.com/Exchange-2018.html
        -   http://info.manh.com/EMEA-Exchange.html
        -   https://www.manh.com/exchange
        -   https://www.manh.com/en-gb
        -   https://www.manh.com/en-gb/
        -   https://www.manh.com/zh-cn/
        -   https://www.manh.com/es-pa
        -   https://www.manh.com/nl-nl
        -   https://www.manh.com/en-au
        -   https://www.manh.com/en-ae
        -   https://www.manh.com/th-th
        -   https://www.manh.com/pt-br
        -   https://www.manh.com/is-is
        -   https://www.manh.com/zh-cn
        -   https://www.manh.com/ms-my
        -   https://www.manh.com/en-hk
        -   https://www.manh.com/en-in
        -   https://www.manh.com/ro-ro
        -   https://www.manh.com/pl-pl
        -   https://www.manh.com/de-de
        -   https://www.manh.com/en-nz
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   http://www.manh.com/fr-fr
        -   https://www.manh.com/ja-jp
        -   https://www.manh.com/pt-pt
        -   https://www.manh.com/nl-be
        -   https://www.manh.com/it-it
        -   https://www.manh.com/en-nl
        -   https://www.manh.com/en-za
        -   https://www.manh.com/no-no
        -   https://www.manh.com/en-sg
        -   https://www.manh.com/fi-fi
        -   https://www.manh.com/es-es
        -   https://www.manh.com/es
        -   http://www.manh.com/es


                """)

st.markdown('## **① Paste some keywords **')

c29, c30, c31 = st.beta_columns([8,1,8])

MAX_LINES = 100

#region text_area_01  ############################################################

with c29:

    text = st.text_area("One URL per line (100 max)", height=175)  
    lines = text.split("\n")  # A list of lines   
  
    linesDeduped = []
    
    for x in lines:    
        linesDeduped.append(x)

    #linesDeduped = list(dict.fromkeys(linesDeduped))
   
    if len(linesDeduped) > MAX_LINES:
        st.warning(f"⚠️ Only the 5 first URLs will be reviewed. Increased allowance  is coming - Stay tuned! 😊)")
        linesDeduped = linesDeduped[:MAX_LINES]
        #st.stop()

    identifier_list = ('https://', 'http://')  # tuple, not list
    notHTTP = [elem for elem in linesDeduped if not elem.startswith(identifier_list)]

    data1 = pd.DataFrame({'box1':linesDeduped})
    data1 = data1.reset_index()
    c = st.beta_container()     

#endregion text_area_01  ############################################################

#region text_area_02  ############################################################

with c31:

    text2 = st.text_area("One URL per line (100 max)", height=175, key = 2)  
   
    #if not text2:
        #st.stop()    
    lines2 = text2.split("\n")  # A list of lines2

    #if '' in lines2:
    #    st.warning('⚠️ Remove empty lines2')
    #    #st.stop()
  
    linesDeduped2 = []
    
    for x in lines2:    
        linesDeduped2.append(x)

    #linesDeduped2 = list(dict.fromkeys(linesDeduped2))
   
    if len(linesDeduped2) > MAX_LINES:
        st.warning(f"⚠️ Only the 5 first URLs will be reviewed. Increased allowance  is coming - Stay tuned! 😊)")
        #st.warning(f"You've exceeded the allowed number of URLs (5 max). We're planning to #increase the allowance soon, so stay tuned! 😊)")
        linesDeduped2 = linesDeduped2[:MAX_LINES]
        #st.stop()

    identifier_list = ('https://', 'http://')  # tuple, not list
    notHTTP = [elem for elem in linesDeduped2 if not elem.startswith(identifier_list)]

    data2 = pd.DataFrame({'box2':linesDeduped2})
    
    #st.write('data2')
    data2 = data2.reset_index()

    c = st.beta_container()     
        
#endregion text2_area_02  ############################################################

mergedDF = data1.merge(data2, on='index', how='right')

@st.cache(suppress_st_warning=True)
def ratio(row):
    name = row['box1']
    name1 = row['box2']
    return fuzz.ratio(name, name1)

@st.cache(suppress_st_warning=True)
def partial_ratio(row):
    name = row['box1']
    name1 = row['box2']
    return fuzz.partial_ratio(name, name1)

@st.cache(suppress_st_warning=True)
def token_sort_ratio(row):
    name = row['box1']
    name1 = row['box2']
    return fuzz.token_sort_ratio(name, name1)

@st.cache(suppress_st_warning=True)
def token_set_ratio(row):
    name = row['box1']
    name1 = row['box2']
    return fuzz.token_set_ratio(name, name1)

#endregion

#region [Table] Apply ratios to the dataframe ############################

#mergedDF

dfRatio = mergedDF.copy()

#partial_ratio
dfRatio['partial_ratio'] = dfRatio.apply(partial_ratio, axis = 1)
dfRatio['partial_ratio'] = dfRatio['partial_ratio'].astype(np.float64)

#2nd ratio
dfRatio['token_sort_ratio'] = dfRatio.apply(token_sort_ratio, axis = 1)
dfRatio['token_sort_ratio'] = dfRatio['token_sort_ratio'].astype(np.float64)

#3rd ratio
dfRatio['token_set_ratio'] = dfRatio.apply(token_set_ratio, axis = 1)
dfRatio['token_set_ratio'] = dfRatio['token_set_ratio'].astype(np.float64)


PartialRatio = dfRatio.copy()
PartialRatio = PartialRatio.drop(['token_set_ratio','token_sort_ratio'], axis=1)
PartialRatio.rename(columns={'partial_ratio':'ratio'}, inplace=True)

PartialRatio['ratioClass'] = np.where(PartialRatio['box1'] == PartialRatio['box2'],
                                'Duplicate',
                                np.where(PartialRatio['box2'] == '/'
                                , 'Redirects to Home P.',
                                np.where(PartialRatio.ratio >= 80,
                                'HighScore',
                                np.where(PartialRatio.ratio >= 70,
                                'MediumScore',
                                'Low score'))))

PartialRatio['algoType'] = 'partial_ratio'


#region [Table] token_sort_ratio table ########################################################################

dfToken = dfRatio.copy()
dfToken = dfToken.drop(['partial_ratio','token_set_ratio'], axis=1)
dfToken.rename(columns={'token_sort_ratio':'ratio'}, inplace=True)

dfToken['ratioClass'] = np.where(dfToken['box1'] == dfToken['box2'],
                                'Duplicate',
                                np.where(dfToken['box2'] == '/',
                                'Redirects to Home P.',
                                np.where(dfToken.ratio >= 80,
                                'HighScore',
                                np.where(dfToken.ratio >= 70,
                                'MediumScore',
                                'Low score'))))

dfToken['algoType'] = 'token_sort_ratio'

  #endregion

  #region [Table] token_set_ratio table ########################################################################

dfTokenSet = dfRatio.copy()
dfTokenSet = dfTokenSet.drop(['partial_ratio','token_sort_ratio'], axis=1)
dfTokenSet.rename(columns={'token_set_ratio':'ratio'}, inplace=True)

dfTokenSet['ratioClass'] = np.where(dfTokenSet['box1'] == dfTokenSet['box2'],
                                'Duplicate',
                                np.where(dfTokenSet['box2'] == '/',
                                'Redirects to Home P.',
                                np.where(dfTokenSet.ratio >= 80,
                                'HighScore',
                                np.where(dfTokenSet.ratio >= 70,
                                'MediumScore',
                                'Low score'))))

dfTokenSet['algoType'] = 'token_set_ratio'

#endregion

#region [Table] Append all tables #################################################################

dfNew = PartialRatio.append([dfToken,dfTokenSet])

#region [Table] Pivot table #######################################################################


dfNew
#df.loc[df.col1 == 'Yes', 'col2'] = ''

dfPivot1 = pd.pivot_table(dfNew, values='ratio', index=['ratioClass', 'algoType'], aggfunc=len).reset_index('algoType')
dfPivot1['algoType'] = dfPivot1.replace({'algoType' : { 'token_set_ratio' : "Token Set ratio count", 'token_sort_ratio' : "Token Sort ratio count", 'partial_ratio' : "Partial ratio count"}})
dfPivot1['ratioName'] = dfPivot1.index

with st.beta_expander("ℹ️ - Display dfPivot", expanded=True):

    c29, c30, c31 = st.beta_columns([8,1,8])


    with c29:
        st.write('dfPivot1')
        dfPivot1
        #dfPivot1.to_csv('dfPivot1.csv')
    
    with c31:
        dfTemplate = pd.DataFrame({
            #'index':[1,2,3,4,5,6,7,8,9],
            'algoType':['Partial ratio count','Token Set ratio count','Token Sort ratio count','Partial ratio count','Token Set ratio count','Token Sort ratio count','Partial ratio count','Token Set ratio count','Token Sort ratio count','Partial ratio count','Token Set ratio count','Token Sort ratio count'],
            'ratio':[0,0,0,0,0,0,0,0,0,0,0,0],
            'ratioName':['Duplicate','Duplicate','Duplicate','HighScore','HighScore','HighScore','MediumScore','MediumScore','MediumScore','Low score','Low score','Low score']})

        st.write('dfTemplate')
        dfTemplate

        #dfTemplate.to_csv('dfTemplate.csv')
    
        dfMerged3 = pd.concat([dfTemplate,dfPivot1])     
        dfMerged3.drop_duplicates(subset=['algoType','ratioName'], inplace=True, keep='last')       
        dfMerged3 = dfMerged3.sort_values(['ratioName','algoType' ], ascending=[True, True])

        st.write('dfMerged3')
        dfMerged3


        dfPivotFiltered1 = dfMerged3.loc[dfMerged3['ratioName'] == 'HighScore']
        dfPivotFiltered2 = dfMerged3.loc[dfMerged3['ratioName'] == 'MediumScore']
        dfPivotFiltered3 = dfMerged3.loc[dfMerged3['ratioName'] == 'Duplicate']
        dfPivotFiltered4 = dfMerged3.loc[dfMerged3['ratioName'] == 'Low score']


listHighScore = dfPivotFiltered1['ratio'].tolist()
listMediumScore = dfPivotFiltered2['ratio'].tolist()
listLow = dfPivotFiltered4['ratio'].tolist()
listDuplicate = dfPivotFiltered3['ratio'].tolist()


st.write("listLow")
listLow
st.write("listDuplicate")
listDuplicate
st.write("listMediumScore")
listMediumScore
st.write("listHighScore")
listHighScore

Pivot_dict = dict(zip(dfPivot1.ratio,  dfPivot1.algoType))
Pivot_dict

st.markdown('## **① Results Overview **')

Algolist = []

st.markdown('## **- Tabular View **')

c1000 = st.beta_container()   

with st.beta_expander("ℹ️ - More info about algos", expanded=False):

    st.write("""        
    -   Algo 1
    -   Algo 2
            """)

c0, c0a, c1, c1a, c2, c2a, c3 = st.beta_columns([6,0.2,6,1,6,1,6])

with c1:

    two  = st.checkbox("partial_ratio", value=True)
    c1 = st.beta_container()
    if two:
        Algolist.append("partial_ratio") 

with c3:
    three = st.checkbox("token_sort_ratio", value=True)
    c3 = st.beta_container()
    if three:
        Algolist.append("token_sort_ratio")  

with c2:
    one = st.checkbox("token_set_ratio", value=True)
    c2 = st.beta_container()
    if one:
        Algolist.append("token_set_ratio")  


Algolist


scores = ['Duplicate','High_Score','Medium_Score','listLow']

opts = {
    "tooltip": {
        "trigger": 'axis',
        "axisPointer": {            
            "type": 'shadow'
        }
    },
    "color":['#ae1029','#57904b','#fb8649', '#0065c2'],
    "legend": {
        
        "data": scores
    },
    "grid": {
        "left": '13%',
        "right": '14%',
        "bottom": '13%',
        "containLabel": False
    },
    "xAxis": {
        "type": 'value'
    },
    "yAxis": {
        "type": 'category',
        "data": Algolist
    },
    "series": [
        {
            "name": 'Duplicate',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listDuplicate
        },
        {
            "name": 'High_Score',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listHighScore
        },
        {
            "name": 'Medium_Score',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listMediumScore
        },
        {
            "name": 'listLow',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listLow
        },

    ]
};



dfFiltered = mergedDF.copy()

with c1000.beta_container():
    #st_echarts(opts, width = 1500, height= 400)
    st_echarts(opts, width = 1500, height= 400)

#region multiselect ############################################################

if two == True:

    # THEN add related slider
    maxValuePartial = dfRatio['partial_ratio'].max()
    minValuePartial = dfRatio['partial_ratio'].min()

    #maxValuePartial
    #minValuePartial

    if minValuePartial < maxValuePartial:
        Partial_slider = c1.slider('', minValuePartial, maxValuePartial, value = float(maxValuePartial), step = 1.00)
        dfFiltered['partial_ratio'] = dfFiltered.apply(partial_ratio, axis = 1)
        dfFiltered['partial_ratio'] = dfFiltered['partial_ratio'].astype(np.int64)
        dfFiltered.rename(columns = {'partial_ratio':'Partial Ratio'}, inplace = True)
        dfFiltered = dfFiltered[(dfFiltered['Partial Ratio'] <= Partial_slider)]
    else:
        pass
        #st.markdown("No Partial slider as `minValue` = `maxValue`")

if three == True:

        # THEN add related slider
        maxValueToken_sort = dfRatio['token_sort_ratio'].max()
        minValueToken_sort = dfRatio['token_sort_ratio'].min()

        if minValueToken_sort < maxValueToken_sort:
            
            Token_sort_slider = c3.slider('', minValueToken_sort, maxValueToken_sort, value = float(maxValueToken_sort), step = 1.00, key = 1)
            dfFiltered['token_sort_ratio'] = dfFiltered.apply(token_sort_ratio, axis = 1)
            dfFiltered['token_sort_ratio'] = dfFiltered['token_sort_ratio'].astype(np.int64)
            dfFiltered.rename(columns = {'token_sort_ratio':'Token Sort'}, inplace = True)
            dfFiltered = dfFiltered[(dfFiltered['Token Sort'] <= Token_sort_slider)]
        else:   
            pass
            #st.markdown("No T-Sort slider as `minValue` = `maxValue`")

if one == True:

    maxValueToken_set = dfRatio['token_set_ratio'].max()
    minValueToken_set = (dfRatio['token_set_ratio'].min())

    if minValueToken_set < maxValueToken_set:
        Token_set_slider = c2.slider('', minValueToken_set, maxValueToken_set, value = float(maxValueToken_set), step = 1.00, key = 10)
        dfFiltered['token_set_ratio'] = dfFiltered.apply(token_set_ratio, axis = 1)
        dfFiltered['token_set_ratio'] = dfFiltered['token_set_ratio'].astype(np.int64)
        dfFiltered.rename(columns = {'token_set_ratio':'Token Set'}, inplace = True)
        dfFiltered = dfFiltered[(dfFiltered['Token Set'] <= Token_set_slider)]
    else:
        pass
        #st.markdown("No T-Set slider as `minValue` = `maxValue`")

dfFiltered['Duplicate'] = np.where(dfFiltered['box1'] == dfFiltered['box2'],
                                'Duplicate',
                                np.where(dfFiltered['box2'] == '/',
                                'Redirects to Home P.',
                                'OK'))

AllUniqueValues = list(dfFiltered['Duplicate'].unique())

loopBox = c0.checkbox('Show only duplicates', key = 100) 

if loopBox:
    dfFiltered = dfFiltered[dfFiltered['Duplicate'] == "Duplicate"]


del dfFiltered['index']


def where(x):
    bg = ['red', 'peachpuff']
    fg = ['white', 'black']  
    ls = ['Duplicate', 'Redirects to Home P.']
    for i, y in enumerate(ls):
        if y in x:
            return f"background-color: {bg[i]}; color: {fg[i]}"
    return ''

try:

    st.table(dfFiltered.style.background_gradient(axis=None, cmap='Reds_r')\
                                    .applymap(where, subset=['Duplicate']))
except ValueError:
    st.warning ('👈 -  Select at least 1 fuzzy algo in section 2️⃣ to see the table!')


try:
    csv = dfFiltered.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown('## ** ③ Download CSV **')
    
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_table.csv">**Click here to get the prize!**</a>'
    st.markdown(href, unsafe_allow_html=True)

except NameError:

    print ('wait')

#endregion

except NameError:
    st.markdown ('NameError')

