# NLP-store-offers
This is my work for the Fetch Rewards Data Scientist(NLP) take home exam I did. Here is the instructions for the assignment:

For this assignment, you will build a tool that allows users to intelligently search for offers via text input from the user.

You will be provided with a dataset of offers and some associated metadata around the retailers and brands that are sponsoring the offer. You will also be provided with a dataset of some brands that we support on our platform, and the categories that those products belong to.

Acceptance Criteria:

- **If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category.**

- **If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.**

- **If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.**

- **The tool should also return the score that was used to measure the similarity of the text input with each offer**

# How to run 


### Streamlit app:
1. Go to directory:
   ```
   cd (directory location)
   ```
2. Clone repository:
    ```
   git clone https://github.com/RileyFischer/NLP-store-offers
   ```
3. install libraries:
   ```
   pip install -r requirements.txt
   ```
4. run streamlit app:
   ```
   streamlit run main.py
   ```
### Explore code:
1. Open main.ipynb to see the code and explore

# Features

- **Streamlit based web interface**
- **Data Cleaning**
- **Semantic similarity**: Using pretrained models to find the semantic similarity of text.
- - **Feature engineering**: making full use of all of the data present for more relevant offers.
  

# Notation
$C:=$ The list of all possible categories

$c:=$ A category within C

$RC(brand):=$ The relevant categories of the brand. A category is relevant if it shares the same parent category as the categories that the brand has receipts for.

$r(c,b):=$ number of receipts for categroy c and brand.

$s(offer):=$ The semantic similarity of the search to the offer.

$s(brand):=$ The semantic similarity of the search to the brand.

$s(retailer):=$ The semantic similarity of the search to the retailer.

$s(c):=$ The semantic similarity of the search to the category c.
    
    
# This is how the score is calculated when searching by category
    
$$score|offer,brand= \frac{s(offer)+\sum_{c\in RC(brand)}\left[\left(\frac{r(c,brand)}{\sum_{c\in C}r(c,brand)}\right)\left(s(c
)\right)\right]}{2}$$
    
# This is how the score is calculated when searching by brand
    
$$score|offer,brand= \frac{s(offer)+\sqrt{\left(s(brand)\right)\sum_{c\in C}\sqrt{\left(\frac{r(c,search)}{\sum_{c\in C}r(c,search)}\right)\left(\frac{r(c,brand)}{\sum_{c\in RC(brand)}r(c,brand)}\right)}}}{2}$$
where
$$\sum_{c\in C}\sqrt{\left(\frac{r(c,search)}{\sum_{c\in C}r(c,search)}\right)\left(\frac{r(c,brand)}{\sum_{c\in RC(brand)}r(c,brand)}\right)}$$
represents the similarity of the originaly searched brand to some other brand by taking into account the Receipts column. For example if the searched brand was Dr. Pepper, the receipts column shows us that Dr. Pepper is 87% carbonated soft drinks and 13% cooking & baking. Therefor some other brand has a value of 1 for this expression if that brand also is 87% carbonated soft drinks and 13% cooking & baking, and somewhere between 0 and 1 if it has a different percentage of carbonated soft drinks and cooking & baking
    
    
# This is how the score is calculated when searching by retailer
$$score|offer,brand,retailer=\frac{s(offer)+s(brand)+s(retailer)}{3}$$
    

# Notes
- **One assumption I made is the the receipts column coresponds to the number of times an offer within that category for the brand has been used.**

- **For the cosine similarity score I used the sentence transformer \"multi-qa-mpnet-base-cos-v1\" on the offers, brands, retailer, and search strings. I then measured the similarity of vectors from the sentence transformer by using cosine similarity.**

- **The use of the square roots and dividing by 2 or 3 is put in place so that all scores have a range of -1 to 1.**

- **Each search will score every possible offer. While we would want to limit the number of offers to just show a few of the top scoring offers, by scoring every offer it is possible for the user to keep scrolling through offers untill they find one they like.**

- **For each score I am using multiple metrics and combining them. For example in the retailer score I just combine all three and weigh them equally. While this seems to be effective, I think if there was more data and true score metric that it would be a good idea to work on developing a model to treat each metric as a seperate feature and find how they can be combined in a more thoughtful way to reach a final score metric.**
  
- **Brands are treated equally regardless of their total number of receipts. The benefit of this is that the offers presented are based on relevance so we should always be getting the most relevant offer possible. However this can be bad since people are less likely to shop at small brands, therefor having the list be flooded with small brands compared to large brands makes the offers less relevant to the person. I choose to assume people would be just as interested in offers from small brands compared to large brands, but in reality I think it would make sense to assume people prefer offers from large brands since more people already shop there.**
