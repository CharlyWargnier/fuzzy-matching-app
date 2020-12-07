
import streamlit as st

from streamlit_echarts import st_echarts
from pyecharts import options as opts

from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import base64
import csv

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


c30, c31, c32 = st.beta_columns(3)

with c30:
    st.image('logo.png', width = 425)
    st.header('')

with c32:
  st.header('')
  st.header('')
  st.markdown('###### Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://twitter.com/DataChaz) &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)')


with st.beta_expander("ℹ️ - About this app ", expanded=True):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   This data app uses _________________ to ___________________________.
-   The tool is in Beta. Feedback & bug spotting are welcome. [DMs are open!](https://twitter.com/DataChaz)
-   This app is free. If it's useful to you, you can [buy me a coffee](https://www.buymeacoffee.com/cwar05) to support my work! 🙏

	    """)


###############################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# Fixing random state for reproducibility
np.random.seed(19680801)
N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = np.random.randn(N_points)
y = .4 * x + np.random.randn(100000) + 5

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(x, bins=n_bins)
axs[1].hist(y, bins=n_bins)

###############################################

#st.markdown('## **① ▼ Paste some keywords **')
st.markdown('## ** ▼ Paste some keywords **')

c29, c30, c31 = st.beta_columns([8,1,8])

MAX_LINES = 1000

#region text_area_01  ############################################################

with c29:

    text = st.text_area("One Keyword or URL per line (1000 max)", height=175)  
    lines = text.split("\n")  # A list of lines   
  
    linesDeduped = []
    
    for x in lines:    
        linesDeduped.append(x)

    #linesDeduped = list(dict.fromkeys(linesDeduped))
   
    if len(linesDeduped) > MAX_LINES:
        st.warning(f"⚠️ Only the first 1000 elements will be reviewed. More coming, stay tuned! 😊")
        linesDeduped = linesDeduped[:MAX_LINES]
        #st.stop()

    identifier_list = ('https://', 'http://')  # tuple, not list
    notHTTP = [elem for elem in linesDeduped if not elem.startswith(identifier_list)]

    data1 = pd.DataFrame({'Box #01':linesDeduped})
    data1 = data1.reset_index()
    c = st.beta_container()     

#endregion text_area_01  ############################################################

#region text_area_02  ############################################################

with c31:

    text2 = st.text_area("One Keyword or URL per line (1000 max)", height=175, key = 2)  
   
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
        st.warning(f"⚠️ Only the first 1000 elements will be reviewed. More coming, stay tuned! 😊")
        #st.warning(f"You've exceeded the allowed number of URLs (5 max). We're planning to #increase the allowance soon, so stay tuned! 😊")
        linesDeduped2 = linesDeduped2[:MAX_LINES]
        #st.stop()

    identifier_list = ('https://', 'http://')  # tuple, not list
    notHTTP = [elem for elem in linesDeduped2 if not elem.startswith(identifier_list)]

    data2 = pd.DataFrame({'Box #02':linesDeduped2})
    
    #st.write('data2')
    data2 = data2.reset_index()

    c = st.beta_container()     
        
#endregion text2_area_02  ############################################################

mergedDF = data1.merge(data2, on='index', how='right')

@st.cache(suppress_st_warning=True)
def ratio(row):
    name = row['Box #01']
    name1 = row['Box #02']
    return fuzz.ratio(name, name1)

@st.cache(suppress_st_warning=True)
def partial_ratio(row):
    name = row['Box #01']
    name1 = row['Box #02']
    return fuzz.partial_ratio(name, name1)

@st.cache(suppress_st_warning=True)
def token_sort_ratio(row):
    name = row['Box #01']
    name1 = row['Box #02']
    return fuzz.token_sort_ratio(name, name1)

@st.cache(suppress_st_warning=True)
def token_set_ratio(row):
    name = row['Box #01']
    name1 = row['Box #02']
    return fuzz.token_set_ratio(name, name1)

#endregion

#region [Table] Apply ratios to the dataframe ############################

#mergedDF

dfRatio = mergedDF.copy()

#partial_ratio
dfRatio['Partial Ratio'] = dfRatio.apply(partial_ratio, axis = 1)
dfRatio['Partial Ratio'] = dfRatio['Partial Ratio'].astype(np.float64)

#2nd ratio
dfRatio['token_sort_ratio'] = dfRatio.apply(token_sort_ratio, axis = 1)
dfRatio['token_sort_ratio'] = dfRatio['token_sort_ratio'].astype(np.float64)

#3rd ratio
dfRatio['Token Set Ratio'] = dfRatio.apply(token_set_ratio, axis = 1)
dfRatio['Token Set Ratio'] = dfRatio['Token Set Ratio'].astype(np.float64)


PartialRatio = dfRatio.copy()
PartialRatio = PartialRatio.drop(['Token Set Ratio','token_sort_ratio'], axis=1)
PartialRatio.rename(columns={'Partial Ratio':'ratio'}, inplace=True)

PartialRatio['ratioClass'] = np.where(PartialRatio['Box #01'] == PartialRatio['Box #02'],
                                'Duplicate',
                                np.where(PartialRatio['Box #02'] == '/'
                                , 'Redirects to Home P.',
                                np.where(PartialRatio.ratio >= 80,
                                'HighScore',
                                np.where(PartialRatio.ratio >= 40,
                                'MediumScore',
                                'Low score'))))

PartialRatio['algoType'] = 'Partial Ratio'


#region [Table] token_sort_ratio table ########################################################################

dfToken = dfRatio.copy()
dfToken = dfToken.drop(['Partial Ratio','Token Set Ratio'], axis=1)
dfToken.rename(columns={'token_sort_ratio':'ratio'}, inplace=True)

dfToken['ratioClass'] = np.where(dfToken['Box #01'] == dfToken['Box #02'],
                                'Duplicate',
                                np.where(dfToken['Box #02'] == '/',
                                'Redirects to Home P.',
                                np.where(dfToken.ratio >= 80,
                                'HighScore',
                                np.where(dfToken.ratio >= 40,
                                'MediumScore',
                                'Low score'))))

dfToken['algoType'] = 'token_sort_ratio'

  #endregion

  #region [Table] token_set_ratio table ########################################################################

dfTokenSet = dfRatio.copy()
dfTokenSet = dfTokenSet.drop(['Partial Ratio','token_sort_ratio'], axis=1)
dfTokenSet.rename(columns={'Token Set Ratio':'ratio'}, inplace=True)

dfTokenSet['ratioClass'] = np.where(dfTokenSet['Box #01'] == dfTokenSet['Box #02'],
                                'Duplicate',
                                np.where(dfTokenSet['Box #02'] == '/',
                                'Redirects to Home P.',
                                np.where(dfTokenSet.ratio >= 80,
                                'HighScore',
                                np.where(dfTokenSet.ratio >= 40,
                                'MediumScore',
                                'Low score'))))


dfTokenSet['algoType'] = 'Token Set Ratio'


#endregion

#region [Table] Append all tables #################################################################

dfNew = PartialRatio.append([dfToken,dfTokenSet])

#region [Table] Pivot table 
########################################################################


########################################################################

#if not (text and text2):
#    st.success('aaa')
#    st.stop()

########################################################################

dfPivot1 = pd.pivot_table(dfNew, values='ratio', index=['ratioClass', 'algoType'], aggfunc=len).reset_index('algoType')
dfPivot1['algoType'] = dfPivot1.replace({'algoType' : { 'Token Set Ratio' : "Token Set ratio count", 'token_sort_ratio' : "Token Sort ratio count", 'Partial Ratio' : "Partial count"}})
dfPivot1['ratioName'] = dfPivot1.index

#with st.beta_expander("ℹ️ - Display dfPivot", expanded=True):
#
#    c29, c30, c31 = st.beta_columns([8,1,8])
#
#
#    #with c29:
#    #    st.write('test')
#    #    #st.write('dfPivot1')
#    #    #dfPivot1
#    #    #dfPivot1.to_csv('dfPivot1.csv')
#    
#    with c31:


dfTemplate = pd.DataFrame({
    #'index':[1,2,3,4,5,6,7,8,9],
    'algoType':['Partial count','Token Set ratio count','Token Sort ratio count','Partial count','Token Set ratio count','Token Sort ratio count','Partial count','Token Set ratio count','Token Sort ratio count','Partial count','Token Set ratio count','Token Sort ratio count'],
    'ratio':[0,0,0,0,0,0,0,0,0,0,0,0],
    'ratioName':['Duplicate','Duplicate','Duplicate','HighScore','HighScore','HighScore','MediumScore','MediumScore','MediumScore','Low score','Low score','Low score']})

#st.write('dfTemplate')
#dfTemplate

#dfTemplate.to_csv('dfTemplate.csv')

dfMerged3 = pd.concat([dfTemplate,dfPivot1])     
dfMerged3.drop_duplicates(subset=['algoType','ratioName'], inplace=True, keep='last')       
dfMerged3 = dfMerged3.sort_values(['ratioName','algoType' ], ascending=[True, True])

#st.write('dfMerged3')
#dfMerged3


dfPivotFiltered1 = dfMerged3.loc[dfMerged3['ratioName'] == 'HighScore']
dfPivotFiltered2 = dfMerged3.loc[dfMerged3['ratioName'] == 'MediumScore']
dfPivotFiltered3 = dfMerged3.loc[dfMerged3['ratioName'] == 'Duplicate']
dfPivotFiltered4 = dfMerged3.loc[dfMerged3['ratioName'] == 'Low score']


listHighScore = dfPivotFiltered1['ratio'].tolist()
listMediumScore = dfPivotFiltered2['ratio'].tolist()
listLow = dfPivotFiltered4['ratio'].tolist()
listDuplicate = dfPivotFiltered3['ratio'].tolist()

#st.write("listLow")
#listLow
#st.write("listDuplicate")
#listDuplicate
#st.write("listMediumScore")
#listMediumScore
#st.write("listHighScore")
#listHighScore

Pivot_dict = dict(zip(dfPivot1.ratio,  dfPivot1.algoType))
#Pivot_dict




st.markdown('## ** ▼ Check results or download CSV **')

c1, c2 = st.beta_columns(2)
   
with c1:
        with st.beta_expander("ℹ️ - More info about scores", expanded=False):

            st.write("""        
            -   high score means anything above
            -   medium score means anything above
            -   low score means anything above
                    """)  

    #st.markdown('## **① Check results or download CSV **')

with c2:

        with st.beta_expander("ℹ️ - More info about algos", expanded=False):

            st.write("""        
            -   Algo 1
            -   Algo 2
                    """)

        #st.markdown('## **① Check results or download CSV **')


st.markdown('')

Algolist = []

#st.markdown('## **- Tabular View **')

c1000 = st.beta_container()   





c0, c0a, c1, c1a, c2, c2a, c3 = st.beta_columns([6,0.2,6,1,6,1,6])

with c1:

    two  = st.checkbox("Partial Ratio", value=True)
    c1 = st.beta_container()
    if two:
        Algolist.append("Partial Ratio") 

with c3:
    three = st.checkbox("Token Sort Ratio", value=True)
    c3 = st.beta_container()
    if three:
        Algolist.append("Token Sort Ratio")  

with c2:
    one = st.checkbox("Token Set Ratio", value=True)
    c2 = st.beta_container()
    if one:
        Algolist.append("Token Set Ratio")  


#Algolist


scores = ['High score (80-100)','Medium score (40-80)','Low score (0-40)', 'Duplicate']

opts = {
    "tooltip": {
        "trigger": 'axis',
        "axisPointer": {            
            "type": 'shadow'
        }
    },
    #"color":['#ae1029','#57904b','#fb8649', '#0065c2'],
    "color":['#808080','#6AB155','#FDBF02', '#FF0000'],
    "legend": {
        
        "data": scores
    },
    #legend: {
    #orient: 'vertical',
    #x: 'right',
    #y: 'bottom',
    #"padding": 100,
    #"itemGap": 100,
    "textStyle": {
      "fontSize": '13',
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
            "name": 'High score (80-100)',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listHighScore
        },
        {
            "name": 'Medium score (40-80)',
            "type": 'bar',
            "stack": '01',
            "label": {
                "show": True,
                "position": 'insideLeft'
            },
            "data": listMediumScore
        },
        {
            "name": 'Low score (0-40)',
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
    st_echarts(opts, width = 1600, height= 375)

#region multiselect ############################################################

if one == True:

    maxValueToken_set = int(dfRatio['Token Set Ratio'].max())
    minValueToken_set = int(dfRatio['Token Set Ratio'].min())

    if minValueToken_set < maxValueToken_set:
        Token_set_slider = c2.slider('', minValueToken_set, maxValueToken_set, value = int(maxValueToken_set), step = 1, key = 10)
        dfFiltered['Token Set Ratio'] = dfFiltered.apply(token_set_ratio, axis = 1)
        dfFiltered['Token Set Ratio'] = dfFiltered['Token Set Ratio'].astype(np.int64)
        dfFiltered.rename(columns = {'Token Set Ratio':'Token Set'}, inplace = True)
        dfFiltered = dfFiltered[(dfFiltered['Token Set'] <= Token_set_slider)]
    else:
        pass
        #st.markdown("No T-Set slider as `minValue` = `maxValue`")

if two == True:

    maxValuePartial = int(dfRatio['Partial Ratio'].max())
    minValuePartial = int(dfRatio['Partial Ratio'].min())

    if minValuePartial < maxValuePartial:
        Partial_slider = c1.slider('', minValuePartial, maxValuePartial, value = int(maxValuePartial), step = 1)
        dfFiltered['Partial Ratio'] = dfFiltered.apply(partial_ratio, axis = 1)
        dfFiltered['Partial Ratio'] = dfFiltered['Partial Ratio'].astype(np.int64)
        dfFiltered.rename(columns = {'Partial Ratio':'Partial'}, inplace = True)
        dfFiltered = dfFiltered[(dfFiltered['Partial'] <= Partial_slider)]
    else:
        pass


if three == True:

        # THEN add related slider
        maxValueToken_sort = int(dfRatio['token_sort_ratio'].max())
        minValueToken_sort = int(dfRatio['token_sort_ratio'].min())

        if minValueToken_sort < maxValueToken_sort:
            
            Token_sort_slider = c3.slider('', minValueToken_sort, maxValueToken_sort, value = int(maxValueToken_sort), step = 1, key = 1)
            dfFiltered['token_sort_ratio'] = dfFiltered.apply(token_sort_ratio, axis = 1)
            dfFiltered['token_sort_ratio'] = dfFiltered['token_sort_ratio'].astype(np.int64)
            dfFiltered.rename(columns = {'token_sort_ratio':'Token Sort'}, inplace = True)
            dfFiltered = dfFiltered[(dfFiltered['Token Sort'] <= Token_sort_slider)]
        
        else:   
            pass
            #st.markdown("No T-Sort slider as `minValue` = `maxValue`")

dfFiltered['Duplicate'] = np.where(dfFiltered['Box #01'] == dfFiltered['Box #02'],
                                'Duplicate',
                                np.where(dfFiltered['Box #02'] == '/',
                                'Redirects to Home P.',
                                'OK'))

AllUniqueValues = list(dfFiltered['Duplicate'].unique())

loopBox = c0.checkbox('Only show duplicates', key = 100) 

#c0.markdown('---')

c0.write('')
c0.write('')

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

    # tips from Arran - remove index from st.table tables
    #st.table(dfFiltered.assign(hack='').set_index('hack'))

    dfFiltered.index = dfFiltered.index + 1

    st.markdown('---')

    #dfFiltered = dfFiltered[['Box #02','Box #01']]
    
    #dfFiltered = dfFiltered[['Box #01','Box #02','Partial','Token Set','Token Sort','Duplicate']]
    #dfFiltered
    #st.stop()

    st.table(dfFiltered.style.background_gradient(axis=None, cmap='Reds_r')\
                                    .applymap(where, subset=['Duplicate']))
except ValueError:
    #st.warning ('👈 -  Select at least 1 fuzzy algo in section 2️⃣ to see the table!')
    pass


try:
    csv = dfFiltered.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    #st.markdown('## ** ⯈⯈⯈ Download filtered table 🎁 **')
    #st.markdown('## ** ③ Download CSV ⯈ Download link 🎁 **')   
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_table.csv">**   ⯈ Download filtered table 🎁**</a>'
    c0.markdown(href, unsafe_allow_html=True)
    #st.markdown(href, unsafe_allow_html=True)

except NameError:

    print ('wait')

#endregion

except NameError:
    st.markdown ('NameError')


with st.beta_expander("ℹ️ - To-do's ", expanded=True):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  

-   Cached NOT WORKING! (too long to get data in table when sliders are changed)
-   Add cache to score classifiers 
-   Add cache to sliders -if one == True etc...
-   Remove dupe in 'Only the first 1000 elements'
-   Add a toggle button for text areas
-   Change yellow to orange?
-   change color codes in table to do a shade from green to red
-   Add many URLs - from logs
-   Add exception 01: add content in both boxes otherwise error
-   Add Info about algos
-   Add Use cases about fuzz matching

	    """)


with st.beta_expander("ℹ️ - Fixed ", expanded=False):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   remove error: Only the 5 first URLs will be reviewed
-   Increase allowance to 1000
-   Review scores bracket
-   Add Info about scores
-   Change colour charts
-   increase legend size
-   ratios - remove underscores in names
-   re-order columns with partial 1st
-   remove decimals in sliders
-   Amend index in last table!!!!!!!!
-   create a list for low scores - otherwise won't appear in charts
-   remove index in last table
-   biggest issue: created lists don't add up properly and incorrect reading in charts - paste URLs to find out    
-   check -> https://colab.research.google.com/drive/1GZKKPtOCjjic5tbnr9mPzFKGMHnzrEvp#scrollTo=enTISRhUvSK1 
-   issue with labels - invert toekn sort and set!
-   put medium scores on the left of the chart, to be visible
-   Add duplicate in Add search box forain token columns, otherwise biased!

	    """)

with st.beta_expander("ℹ️ - Still To-Do! ", expanded=False):
  #st.markdown(" text to add - https://docs.google.com/document/d/1z9X16ZF0d-T2hc2JEp3EPpbaJfUlpfRDRO0tNsK1ZHs/edit")

  st.write("""  
    
-   Optimise speed and cached functions
-   Add percentages in chart
-   Add more than 1000 URLs
-   sort by partial or token score
-   sort high scores only
-   add one more ratio
-   add bi dirwctional sliders
-   add histogram/distribution per ratio
-   box 1 and 2 - Add search box - Needs to do 'or' not 'and'
-   Add score class in dupe columns

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


