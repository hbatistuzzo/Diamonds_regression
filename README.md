# Diamonds_Henrique

![GitHub top language](https://img.shields.io/github/languages/top/hbatistuzzo/Diamonds_Henrique)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hbatistuzzo/Diamonds_Henrique)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/hbatistuzzo/Diamonds_Henrique)
![GitHub last commit](https://img.shields.io/github/last-commit/hbatistuzzo/Diamonds_Henrique)

## Project objective

<img src="images/diamonds.jpg" align="right" width="45%"/>
This project is based on a [somewhat classic kaggle dataset from 2016](https://www.kaggle.com/datasets/shivam2503/diamonds) used to explain introductory level machine learning.
Given a historic dataset with over 54,000 diamonds prices (`diamonds.csv`) and their characteristics, we are tasked by our client (Rick Harrison from _Pawn Stars_) to estimate the price of his own list (`rick_diamonds.csv`)
of 5,000 diamonds, thus setting up a classic regression problem. Specificaly, the goals are:


- to infer which characteristics are more likely to influence a diamond's price
- to progressively train and test a regression model until its accuracy meet a certain standard (defined by the RMSE)


<p align="center"><img src="images/challenge_objectives.png" alt="full"  width="60%"></p>

---

## Technologies
- Python 3.8.3
	- Pandas 1.4.4
	- Numpy 1.20.3
	- Pycaret 2.3.10
	- Seaborn 0.11.2
	- Matplotlib 3.5.3
	- SQLAlchemy 1.4.42
	- Scikit-learn 1.1

---

## Dataset Description

The list of diamonds contains the following information:1:41 PM 11/24/2022

- Price: Price in US dollars
- Carat: Weight of the diamond
- Cut: Quality of the cut (Fair,Good,Very Good,Premium,Ideal)
- Color: Diamond colour,from J(worst) to D(best)
- Clarity: A measurement of how clear the diamond is(I1(worst),SI2,SI1,VS2,VS1,VVS2,VVS1,IF(best))
- x:Length in mm
- y:Width in mm
- z:Depth in mm
- Depth:Total depth percentage = z/mean(x,y) = 2*z/(x+y)(43-79)
- Table: Width of top of diamond relative to widest point(43-95)

<p align="center"><img src="images/diamonds.jfif" alt="full"  width="60%"></p>

A pairplot of these attributes yields:

<img src="/images/pairplot.png" align="center" width="75%"/>

Which can also be visualized of this heatmap of correlations:

<img src="/images/heatmap.png" align="center" width="75%"/>

---

## Steps
1 - Price predicted as the mean of prices from diamonds.csv(3980)


2 - Price predicted using carat as the only variable from diamonds.csv(1605)


3 - Price predicted using carat and depth variables from diamonds.csv(1598)


4 - Price predicted using carat and table variables from diamonds.csv(1595)


5 - Price predicted using carat,table and depth variables from diamonds.csv(1583)


6 - Price predicted using carat,table,depth and clarity variables from diamonds.csv(1217); Cut does not seen to influence the model


7 - Price predicted using carat,table,depth, clarity and color variables from diamonds.csv(987); Cut does not seen to influence the model


8 - Price predicted using carat,table,depth,x, clarity and color variables from diamonds.csv(709); Cut does not seen to influence the model


X was the missing piece in this linear regression model


9 - Price predicted using carat,table,depth,x, clarity , color and cut(grouped by Fair and Good) variables from diamonds.csv(688)

---

# Conclusion
Using most of the data of the original DataFrame significantly improved the preciseness of the model.