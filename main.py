#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import warnings
import re
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

if st.checkbox('See how the scores are calculated'):
    st.subheader('Notation')
    st.write("$C:=$The list of all possible categories")
    st.write("$c:=$A category within C")
    st.write("$RC(brand):=$The relevant categories of the brand. A category is relevant if it shares the same parent category as the categories that the brand has receipts for.")
    st.write("$r(c,b):=$number of receipts for categroy c and brand")
    st.write("$s(offer):=$The semantic similarity of the search to the offer")
    st.write("$s(brand):=$The semantic similarity of the search to the brand")
    st.write("$s(retailer):=$The semantic similarity of the search to the retailer")
    st.write("$s(c):=$The semantic similarity of the search to the category c")
    st.divider()
    
    st.subheader('This is how the score is calculated when searching by category')
    st.latex(r'''
    score|offer,brand= \frac{s(offer)+\sum_{c\in RC(brand)}\sqrt{\left(\frac{r(c,brand)}{\sum_{c\in C}r(c,brand)}\right)\left(s(c
)\right)}}{2}
    ''')
    st.divider()
    
    st.subheader('This is how the score is calculated when searching by brand')
    st.latex(r'''
    score|offer,brand= \frac{s(offer)+\sqrt{\left(s(brand)\right)\sum_{c\in C}\sqrt{\left(\frac{r(c,search)}{\sum_{c\in C}r(c,search)}\right)\left(\frac{r(c,brand)}{\sum_{c\in RC(brand)}r(c,brand)}\right)}}}{2} 
    ''')
    st.write("where")
    st.latex(r'''
    \sum_{c\in C}\sqrt{\left(\frac{r(c,search)}{\sum_{c\in C}r(c,search)}\right)\left(\frac{r(c,brand)}{\sum_{c\in RC(brand)}r(c,brand)}\right)}
    ''')
    st.write("represents the similarity of the originaly searched brand to some other brand by taking into account the Receipts column. For example if the searched brand was Dr. Pepper, the receipts column shows us that Dr. Pepper is 87% carbonated soft drinks and 13% cooking & baking. Therefor some other brand has a value of 1 for this expression if that brand also is 87% carbonated soft drinks and 13% cooking & baking, and somewhere between 0 and 1 if it has a different percentage of carbonated soft drinks and cooking & baking")
    st.divider()
    
    st.subheader('This is how the score is calculated when searching by retailer')
    st.latex(r'''
    score|offer,brand,retailer=\frac{s(offer)+s(brand)+s(retailer)}{3}
    ''')
    st.divider()

    st.subheader('Notes')
    st.write("-One assumption I made is the the receipts column coresponds to the number of times an offer within that category for the brand has been used.")
    st.write("-For the cosine similarity score I used the sentence transformer \"multi-qa-mpnet-base-cos-v1\" on the offers, brands, retailer, and search strings. I then measured the similarity of vectors from the sentence transformer by using cosine similarity.")
    st.write("-The use of the square roots and dividing by 2 or 3 is put in place so that all scores have a range of -1 to 1.")
    st.write("-Each search will score every possible offer. While we would want to limit the number of offers to just show a few of the top scoring offers, by scoring every offer it is possible for the user to keep scrolling through offers untill they find one they like.")
    st.write("-For each score I am using multiple metrics and combining them. For example in the retailer score I just combine all three and weigh them equally. While this seems to be effective, I think if there was more data and true score metric that it would be a good idea to work on developing a model to treat each metric as a seperate feature and find how they can be combined in a more thoughtful way to reach a final score metric.")
    st.write("-Brands are treated equally regardless of their total number of receipts. The benefit of this is that the offers presented are based on relevance so we should always be getting the most relevant offer possible. However this can be bad since people are less likely to shop at small brands, therefor having the list be flooded with small brands compared to large brands makes the offers less relevant to the person. I choose to assume people would be just as interested in offers from small brands compared to large brands, but in reality I think it would make sense to assume people prefer offers from large brands since more people already shop there.")
    
    
    
progress_text = "Loading in data and tools"
my_bar = st.progress(0, text=progress_text)

@st.cache_data
def loadmodel():
    model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    return model
model=loadmodel()



# # Data cleaning. Remove capitalization, special characters, and duplicate rows
#Remove capitalization
pattern = r'[^\w\s]'
@st.cache_data
def loadbrand():
    brand=pd.read_csv("brand_category.csv")
    brand["BRAND_BELONGS_TO_CATEGORY"]=brand["BRAND_BELONGS_TO_CATEGORY"].str.lower()
    brand["BRAND"]=brand["BRAND"].str.lower()
    brand['BRAND']=brand['BRAND'].astype(str)
    brand["BRAND_BELONGS_TO_CATEGORY"]=brand["BRAND_BELONGS_TO_CATEGORY"].apply(lambda x: re.sub(pattern, '', x))
    brand['BRAND']=brand['BRAND'].apply(lambda x: re.sub(pattern, '', x))
    return brand
brand=loadbrand()

@st.cache_data
def loadcat():
    cat=pd.read_csv("categories.csv")
    cat["IS_CHILD_CATEGORY_TO"]=cat["IS_CHILD_CATEGORY_TO"].str.lower()
    cat["PRODUCT_CATEGORY"]=cat["PRODUCT_CATEGORY"].str.lower()
    cat["IS_CHILD_CATEGORY_TO"]=cat["IS_CHILD_CATEGORY_TO"].apply(lambda x: re.sub(pattern, '', x))
    cat["PRODUCT_CATEGORY"]=cat["PRODUCT_CATEGORY"].apply(lambda x: re.sub(pattern, '', x))
    cat["IS_CHILD_CATEGORY_TO"]=cat["IS_CHILD_CATEGORY_TO"].apply(lambda x: re.sub(pattern, '', x))
    cat["PRODUCT_CATEGORY"]=cat["PRODUCT_CATEGORY"].apply(lambda x: re.sub(pattern, '', x))
    return cat
cat=loadcat()
    
@st.cache_data
def loadoffer():
    offer=pd.read_csv("offer_retailer.csv")
    offer["OFFER"]=offer["OFFER"].str.lower()
    offer["RETAILER"]=offer["RETAILER"].str.lower()
    offer["BRAND"]=offer["BRAND"].str.lower()
    offer["RETAILER"]=offer["RETAILER"].astype(str)
    offer["OFFER"]=offer["OFFER"].apply(lambda x: re.sub(pattern, '', x))
    offer["RETAILER"]=offer["RETAILER"].apply(lambda x: re.sub(pattern, '', x))
    offer["BRAND"]=offer["BRAND"].apply(lambda x: re.sub(pattern, '', x))
    offer=offer.drop_duplicates()
    offer=offer.replace('nan','')
    return offer
offer=loadoffer()



#adding in a multiplier for the brand df. This represents the proportion of receipts come from that category for that brand.
def find_brand_multiplier(b):
    df=brand[brand["BRAND"]==b]
    total_receipts=df["RECEIPTS"].sum()
    df["MULTIPLIER"]=df["RECEIPTS"]/total_receipts
    df=df[["BRAND_BELONGS_TO_CATEGORY","MULTIPLIER"]]
    return df

my_bar.progress(40, text=progress_text)

@st.cache_data
def data_preprocessing():
    df=pd.DataFrame()
    for b in brand.BRAND.unique():
        df_temp=find_brand_multiplier(b)
        df_temp["BRAND"]=b
        df = pd.concat([df, df_temp])
    return df.merge(brand,how='right', on=['BRAND','BRAND_BELONGS_TO_CATEGORY'])
brand=data_preprocessing()


my_bar.progress(60, text=progress_text)

#create a dictionary with parent categories and a list of their children categories
categories={}
parents=cat.IS_CHILD_CATEGORY_TO.unique()
for parent in parents:
    categories[parent]=list(cat[cat["IS_CHILD_CATEGORY_TO"]==parent].PRODUCT_CATEGORY.values)

#create a dictionary with categories and a list of the brands that have receipts for that category.
brands={}
parents=brand.BRAND_BELONGS_TO_CATEGORY.unique()
for parent in parents:
    brands[parent]=list(brand[brand["BRAND_BELONGS_TO_CATEGORY"]==parent].BRAND.values)

my_bar.progress(70, text=progress_text)

# # 1. searches by category
parents=cat.IS_CHILD_CATEGORY_TO.unique()
children=cat.PRODUCT_CATEGORY.unique()

@st.cache_data
def search_category(search):
    #create a similarity df for the search and all category name's
    vectors=model.encode(list(cat['PRODUCT_CATEGORY']))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim=cat.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    
    #Of the categories that are similar to the search, find all of the categories that also share a parent category.
    top_sim=list(sim.nlargest(3, 'Cosine')['PRODUCT_CATEGORY'].values)
    possible_cats=[]
    for c in top_sim:
        if c in parents:
            possible_cats.append(categories[c].copy())
            possible_cats.append([c])

        if c in children:
            for possible_cat in categories:
                if c in categories[possible_cat]:
                    c2=possible_cat
            possible_cats.append(categories[c2].copy())
            possible_cats.append([c2])
    possible_cats = [item for sublist in possible_cats for item in sublist] 

    #find the brands associated with these categories and weigh the brand by the number of their recipts are from these categories
    possible_brands=brand.copy(deep=True)
    possible_brands.loc[~possible_brands["BRAND_BELONGS_TO_CATEGORY"].isin(possible_cats), "MULTIPLIER"]=0
    possible_brands=possible_brands.merge(sim,left_on="BRAND_BELONGS_TO_CATEGORY",right_on="PRODUCT_CATEGORY")
    #Then multiply the weights to get a score that also takes into account how similar the brand is to the original search
    possible_brands["cat_Score"]=possible_brands["MULTIPLIER"]*possible_brands["Cosine"]
    possible_brands=possible_brands.groupby("BRAND").sum("cat_Score")#This score represents how likely the brand has offers related to the search term
    #find the offers from the brands above
    possible_offers=offer.merge(possible_brands,how="left", left_on='BRAND', right_on='BRAND')
    possible_offers=possible_offers.drop(columns=["MULTIPLIER","Cosine","RECEIPTS"])
    possible_offers['cat_Score']=possible_offers['cat_Score'].fillna(0)#if there is a brand thats in the offer df, but not the brand df, we fill the cat_score with 0 for that brand
    
    #similarity of search to offers
    vectors=model.encode(list(offer['OFFER']))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim=offer.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    possible_offers=possible_offers.merge(sim,how="left",on=["OFFER","RETAILER","BRAND"])
    possible_offers["Score"]=(possible_offers["cat_Score"]+possible_offers["Cosine"])/2
    possible_offers=possible_offers.drop(columns=["cat_Score","Cosine"])
    
    possible_offers=possible_offers.sort_values(by=['Score'],ascending=False)
    
    return possible_offers.reset_index(drop=True)



my_bar.progress(80, text=progress_text)


# # 2. Searches by Brand
@st.cache_data
def search_brand(search):
    #create a similarity df for the search and all brand names
    vectors=model.encode(list(set(brand["BRAND"].values)))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim=pd.DataFrame(list(set(brand["BRAND"].values)),columns=["BRAND"])
    sim['Cosine']=cosine.reshape(-1, 1)
    
    if sim.nlargest(1, 'Cosine')["Cosine"].values[0]>.7:#if there is a good match we treat the most similar brand to the search as the new search
        search=sim.nlargest(1, 'Cosine')["BRAND"].values[0]
    old_search=search#saving the old search to be used for finding the similarity between offer and search
    
    possible_brands=[]
    ca=[]
    for c in brands:
        if search in brands[c]:
            ca.append(c)
            possible_brands=brands[c]
    possible_brands=brand[brand["BRAND_BELONGS_TO_CATEGORY"].isin(ca)]
    possible_brands=possible_brands.merge(find_brand_multiplier(search),left_on="BRAND_BELONGS_TO_CATEGORY",right_on="BRAND_BELONGS_TO_CATEGORY")
    possible_brands["MULTIPLIER"]=(possible_brands["MULTIPLIER_x"]*possible_brands["MULTIPLIER_y"])**.5
    possible_brands=possible_brands.groupby("BRAND").sum("MULTIPLIER")
    possible_offers=offer.merge(possible_brands,how='left',left_on='BRAND', right_on='BRAND')
    
    possible_offers=possible_offers.merge(sim,how='left',left_on='BRAND', right_on='BRAND')
    
    possible_offers["Brand_Score"]=(possible_offers["MULTIPLIER"]*possible_offers["Cosine"])**.5
    possible_offers.loc[possible_offers["BRAND"]==search,"Brand_Score"]=1

    possible_offers["Brand_Score"]=possible_offers["Brand_Score"].fillna(0)
    possible_offers=possible_offers.drop(columns=["MULTIPLIER","Cosine"])

    #similarity of search to offers
    vectors=model.encode(list(offer['OFFER']))
    cosine=cosine_similarity(model.encode([old_search]), vectors)
    sim=offer.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    possible_offers=possible_offers.merge(sim,on=["OFFER","RETAILER","BRAND"])

    possible_offers["Score"]=(possible_offers["Brand_Score"]+possible_offers["Cosine"])/2
    possible_offers=possible_offers.drop(columns=["Brand_Score","Cosine","MULTIPLIER_x","MULTIPLIER_y","RECEIPTS"])
    
    possible_offers=possible_offers.sort_values(by=['Score'],ascending=False)
    
    return possible_offers.reset_index(drop=True)


my_bar.progress(90, text=progress_text)


# # 3. Searches by Retailer
@st.cache_data
def search_Retailer(search):
    vectors=model.encode(list(offer['OFFER']))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim=offer.copy(deep=True)
    sim['Cosine1']=cosine.reshape(-1, 1)
    
    vectors=model.encode(list(offer['RETAILER']))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim['Cosine2']=cosine.reshape(-1, 1)
    
    vectors=model.encode(list(offer['BRAND']))
    cosine=cosine_similarity(model.encode([search]), vectors)
    sim['Cosine3']=cosine.reshape(-1, 1)
    
    possible_offers=sim
    
    possible_offers["Score"]=(possible_offers["Cosine1"]+possible_offers["Cosine2"]+possible_offers["Cosine3"])/3
    possible_offers=possible_offers.drop(columns=["Cosine1","Cosine2","Cosine3"])
    possible_offers=possible_offers.sort_values(by=['Score'],ascending=False)

    return possible_offers.reset_index(drop=True)

my_bar.progress(100, text="Data and tools sucessfully loaded.")
my_bar.empty()


search_type = st.selectbox(
    'How would you like to search for an offer?',
    ('Category', 'Brand', 'Retailer'))

if search_type=='Category':
    search = st.text_input('Category name', '')
    if search!='':
        n=st.number_input('How many offers would you like?',step=1)
        df=search_category(search)
        if n:
            st.dataframe(df.head(n))
        else:
            st.dataframe(df)
    
if search_type=='Brand':
    search = st.text_input('Brand name', '')
    if search!='':
        n=st.number_input('How many offers would you like?',step=1)
        st.write('This may take a minute or two for Brand search.')
        df=search_brand(search)
        if n:
            st.dataframe(df.head(n))
        else:
            st.dataframe(df)
            
if search_type=='Retailer':
    search = st.text_input('Retailer name', '')
    if search!='':
        n=st.number_input('How many offers would you like?',step=1)
        df=search_Retailer(search)
        if n:
            st.dataframe(df.head(n))
        else:
            st.dataframe(df)
            
            


