#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import warnings
import re
import numpy as np
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
    score|offer,brand= \frac{s(offer)+\sum_{c\in RC(brand)}\left[\left(\frac{r(c,brand)}{\sum_{c\in C}r(c,brand)}\right)\left(s(c
)\right)\right]}{2}
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
    
    


def loadmodel():
    model=SentenceTransformer('multi-qa-mpnet-base-cos-v1')
    return model
model=loadmodel()


# # Load in cleaned data. Remove capitalization, special characters, and duplicate rows as seen in "Create processed data.ipynb"
#Remove capitalization

def loadbrand():
    brand=pd.read_csv("Data/brand_category_clean.csv")
    brand["BRAND"]=brand["BRAND"].apply(str)
    return brand
brand=loadbrand()

def loadcat():
    cat=pd.read_csv("Data/categories_clean.csv")
    return cat
cat=loadcat()
    
def loadoffer():
    offer=pd.read_csv("Data/offer_retailer_clean.csv")
    return offer
offer=loadoffer()


def loadbrand_vectors():
    brand_vectors=np.load('Data/brand_vectors.npy')
    return brand_vectors
brand_vectors=loadbrand_vectors()


def loadoffer_vectors():
    offer_vectors=np.load('Data/offer_vectors.npy')
    return offer_vectors
offer_vectors=loadoffer_vectors()


def loadcategory_vectors():
    category_vectors=np.load('Data/category_vectors.npy')
    return category_vectors
category_vectors=loadcategory_vectors()


def loadretailer_vectors():
    retailer_vectors=np.load('Data/retailer_vectors.npy')
    return retailer_vectors
retailer_vectors=loadretailer_vectors()


def loadoffer_brand_vectors():
    offer_brand_vectors=np.load('Data/offer_brand_vectors.npy')
    return offer_brand_vectors
offer_brand_vectors=loadoffer_brand_vectors()



#adding in a multiplier for the brand df. This represents the proportion of receipts come from that category for that brand.
def find_brand_multiplier(b):
    df=brand[brand["BRAND"]==b]
    total_receipts=df["RECEIPTS"].sum()
    df["MULTIPLIER"]=df["RECEIPTS"]/total_receipts
    df=df[["BRAND_BELONGS_TO_CATEGORY","MULTIPLIER"]]
    return df



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


# # 1. searches by category
parents=cat.IS_CHILD_CATEGORY_TO.unique()
children=cat.PRODUCT_CATEGORY.unique()


def search_category(search):
    #create a similarity df for the search and all category name's
    cosine=cosine_similarity(model.encode([search]), category_vectors)
    sim=cat.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    
    #Of the categories that are similar to the search, find all of the categories that also share a parent category.
    top_sim=list(sim.nlargest(1, 'Cosine')['PRODUCT_CATEGORY'].values)
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
    possible_brands.loc[possible_brands["BRAND_BELONGS_TO_CATEGORY"].isin(possible_cats), "MULTIPLIER"]=1
    possible_brands=possible_brands.merge(sim,left_on="BRAND_BELONGS_TO_CATEGORY",right_on="PRODUCT_CATEGORY")
    #Then multiply the weights to get a score that also takes into account how similar the brand is to the original search
    possible_brands["cat_Score"]=possible_brands["MULTIPLIER"]*possible_brands["Cosine"]
    possible_brands=possible_brands.groupby("BRAND").mean("cat_Score")#This score represents how likely the brand has offers related to the search term
    #find the offers from the brands above
    possible_offers=offer.merge(possible_brands,how="left", left_on='BRAND', right_on='BRAND')
    possible_offers=possible_offers.drop(columns=["MULTIPLIER","Cosine","RECEIPTS"])
    possible_offers['cat_Score']=possible_offers['cat_Score'].fillna(0)#if there is a brand thats in the offer df, but not the brand df, we fill the cat_score with 0 for that brand
    
    #similarity of search to offers
    cosine=cosine_similarity(model.encode([search]), offer_vectors)
    sim=offer.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    possible_offers=possible_offers.merge(sim,how="left",on=["OFFER","RETAILER","BRAND"])
    possible_offers["Score"]=(possible_offers["cat_Score"]+possible_offers["Cosine"])/2
    possible_offers=possible_offers.drop(columns=["cat_Score","Cosine"])
      
    return possible_offers.reset_index(drop=True)




# # 2. Searches by Brand
def search_brand(search):
    #create a similarity df for the search and all brand names
    
    cosine=cosine_similarity(model.encode([search]), brand_vectors)
    sim=pd.DataFrame(sorted(list(set(brand["BRAND"].values))),columns=["BRAND"])
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
    possible_brands=possible_brands.groupby("BRAND").sum()
    possible_offers=offer.merge(possible_brands,how='left',left_on='BRAND', right_on='BRAND')
    
    possible_offers=possible_offers.merge(sim,how='left',left_on='BRAND', right_on='BRAND')
    
    possible_offers["Brand_Score"]=(possible_offers["MULTIPLIER"]*possible_offers["Cosine"])**.5
    possible_offers.loc[possible_offers["BRAND"]==search,"Brand_Score"]=1

    possible_offers["Brand_Score"]=possible_offers["Brand_Score"].fillna(0)
    possible_offers=possible_offers.drop(columns=["MULTIPLIER","Cosine"])

    #similarity of search to offers
    cosine=cosine_similarity(model.encode([old_search]), offer_vectors)
    sim=offer.copy(deep=True)
    sim['Cosine']=cosine.reshape(-1, 1)
    possible_offers=possible_offers.merge(sim,on=["OFFER","RETAILER","BRAND"])

    possible_offers["Score"]=(possible_offers["Brand_Score"]+possible_offers["Cosine"])/2
    possible_offers=possible_offers.drop(columns=["Brand_Score","Cosine","BRAND_BELONGS_TO_CATEGORY","MULTIPLIER_x","MULTIPLIER_y","RECEIPTS"])
        
    return possible_offers.reset_index(drop=True)




# # 3. Searches by Retailer
def search_Retailer(search):
    cosine=cosine_similarity(model.encode([search]), offer_vectors)
    sim=offer.copy(deep=True)
    sim['Cosine1']=cosine.reshape(-1, 1)
    
    cosine=cosine_similarity(model.encode([search]), retailer_vectors)
    sim['Cosine2']=cosine.reshape(-1, 1)
    
    cosine=cosine_similarity(model.encode([search]), offer_brand_vectors)
    sim['Cosine3']=cosine.reshape(-1, 1)
    
    possible_offers=sim
    
    possible_offers["Score"]=(possible_offers["Cosine1"]+possible_offers["Cosine2"]+possible_offers["Cosine3"])/3
    possible_offers=possible_offers.drop(columns=["Cosine1","Cosine2","Cosine3"])

    return possible_offers.reset_index(drop=True)



search = st.text_input('Search by product, brand or category', '')
if search!='':
    result = pd.concat([search_Retailer(search),search_brand(search),search_category(search)]).sort_values(by=['Score'],ascending=False)
    result=result.drop_duplicates(subset=["OFFER","RETAILER","BRAND"],keep='first')
    result=result.reset_index(drop=True)
    st.dataframe(result)
    

            
            


