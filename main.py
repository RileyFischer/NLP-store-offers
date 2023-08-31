#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import warnings
import re
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st



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


@st.cache_data
def data_preprocessing():
    df=pd.DataFrame()
    for b in brand.BRAND.unique():
        df_temp=find_brand_multiplier(b)
        df_temp["BRAND"]=b
        df = pd.concat([df, df_temp])
    return df.merge(brand,how='right', on=['BRAND','BRAND_BELONGS_TO_CATEGORY'])
brand=data_preprocessing()



#create a dictionary with parent categories and a list of their children categories
@st.cache_data
def loadcategories():
    categories={}
    parents=cat.IS_CHILD_CATEGORY_TO.unique()
    for parent in parents:
        categories[parent]=list(cat[cat["IS_CHILD_CATEGORY_TO"]==parent].PRODUCT_CATEGORY.values)
    return categories
categories=loadcategories()

#create a dictionary with categories and a list of the brands that have receipts for that category.
@st.cache_data
def loadbrands():
    brands={}
    parents=brand.BRAND_BELONGS_TO_CATEGORY.unique()
    for parent in parents:
        brands[parent]=list(brand[brand["BRAND_BELONGS_TO_CATEGORY"]==parent].BRAND.values)
    return brands
brands=loadbrands()

    


# # 1. searches by category
parents=cat.IS_CHILD_CATEGORY_TO.unique()
children=cat.PRODUCT_CATEGORY.unique()


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
    possible_brands["cat_Score"]=(possible_brands["MULTIPLIER"]*possible_brands["Cosine"])
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






# # 2. Searches by Brand

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





# # 3. Searches by Retailer

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




search_type = st.selectbox(
    'How would you like to search for an offer?',
    ('Category', 'Brand', 'Retailer'))
n=None
if search_type=='Category':
    search = st.text_input('Category name', '')
    n=st.number_input('How many offers would you like?',step=1)
    if n:
        if search!='':
            df=search_category(search)
            st.dataframe(df.head(n))
            

if search_type=='Brand':
    search = st.text_input('Brand name', '')
    n=st.number_input('How many offers would you like?',step=1)
    if n:
        if search!='':
            st.write('This may take a minute or two for Brand search.')
            df=search_brand(search)
            st.dataframe(df.head(n))
            
            
if search_type=='Retailer':
    search = st.text_input('Retailer name', '')
    n=st.number_input('How many offers would you like?',step=1)
    if n:
        if search!='':
            df=search_Retailer(search)
            st.dataframe(df.head(n))
        
            
            


