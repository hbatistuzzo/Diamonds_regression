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
- to progressively train and test a regression model until its accuracy meet a certain standard (defined by the RMSE). Rick’s goal is to obtain an average error below 900 dollars.


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

## Dataset Description and Inspection:

The list of diamonds contains the following information:

- carat (0.2-5.01): The carat is the diamond’s physical weight measured in metric carats. One carat equals 0.20 gram and is subdivided into 100 points.
- cut (Fair, Good, Very Good, Premium, Ideal): The quality of the cut. The more precise the diamond is cut, the more captivating the diamond is to the eye thus of high grade.
- color (from J (worst) to D (best)): The colour of gem-quality diamonds occurs in many hues. In the range from colourless to light yellow or light brown. Colourless diamonds are the rarest. Other natural colours (blue, red, pink for example) are known as "fancy,” and their colour grading is different than from white colorless diamonds.
- clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)): Diamonds can have internal characteristics known as inclusions or external characteristics known as blemishes. Diamonds without inclusions or blemishes are rare; however, most characteristics can only be seen with magnification.
- depth (43-79): It is the total depth percentage which equals to z / mean(x, y) = 2 * z / (x + y). The depth of the diamond is its height (in millimetres) measured from the culet (bottom tip) to the table (flat, top surface) as referred in the labelled diagram above.
- table (43-95): It is the width of the top of the diamond relative to widest point. It gives diamond stunning fire and brilliance by reflecting lights to all directions which when seen by an observer, seems lustrous.
- price ($$326 - $18826): It is the price of the diamond in US dollars. It is our very target column in the dataset.
- x (0 - 10.74): Length of the diamond (in mm)
- y (0 - 58.9): Width of the diamond (in mm)
- z (0 - 31.8): Depth of the diamond (in mm)

<p align="center"><img src="images/diamonds.jfif" alt="fuller"  width="60%"></p>

The dataset itself doesn't need any cleaning other than the removal of a few lines where dimensions (y or x) are set to zero, which is physically impossible.

- A first modelling atempt will be performed by exploring the relationship between price and the physical dimensions of the diamonds. An Univariate Analysis for statistical description yields

|       |        carat |        depth |        table |        price |            x |            y |            z |
|------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| count | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 | 48940.000000 |
|  mean |     0.797817 |    61.751931 |    57.451161 |  3934.409644 |     5.730712 |     5.734333 |     3.538648 |
|   std |     0.474126 |     1.430026 |     2.233450 |  3989.333861 |     1.121920 |     1.145344 |     0.706817 |
|   min |     0.200000 |    43.000000 |    43.000000 |   326.000000 |     0.000000 |     0.000000 |     0.000000 |
|   25% |     0.400000 |    61.000000 |    56.000000 |   949.000000 |     4.710000 |     4.720000 |     2.910000 |
|   50% |     0.700000 |    61.800000 |    57.000000 |  2401.000000 |     5.690000 |     5.710000 |     3.520000 |
|   75% |     1.040000 |    62.500000 |    59.000000 |  5331.250000 |     6.540000 |     6.540000 |     4.040000 |
|   max |     5.010000 |    79.000000 |    95.000000 | 18823.000000 |    10.740000 |    58.900000 |    31.800000 |

- "Price", as expected, is skewed. There are few diamonds which are worth too much and a lot of diamonds with reasonably small prices.

<p align="center"><img src="images/prices.png" alt="prices"  width="100%"></p>

Bivariate Analysis: a first tentative step is to analyze the correlation matrix between the variables, which yields:

<p align="center">

|       |    carat |     depth |     table |     price |         x |         y |        z |
|------:|---------:|----------:|----------:|----------:|----------:|----------:|---------:|
| carat | 1.000000 |  0.027074 |  0.181688 |  0.922186 |  0.975152 |  0.949687 | 0.951824 |
| depth | 0.027074 |  1.000000 | -0.297123 | -0.012037 | -0.025858 | -0.029903 | 0.094344 |
| table | 0.181688 | -0.297123 |  1.000000 |  0.127832 |  0.195367 |  0.183362 | 0.150646 |
| price | 0.922186 | -0.012037 |  0.127832 |  1.000000 |  0.885019 |  0.864059 | 0.860247 |
|     x | 0.975152 | -0.025858 |  0.195367 |  0.885019 |  1.000000 |  0.972447 | 0.969336 |
|     y | 0.949687 | -0.029903 |  0.183362 |  0.864059 |  0.972447 |  1.000000 | 0.948768 |
|     z | 0.951824 |  0.094344 |  0.150646 |  0.860247 |  0.969336 |  0.948768 | 1.000000 |

</p>

Which can also be visualized as a heatmap of correlations:

<p align="center"><img src="images/heatmap.png" alt="heat"  width="75%"></p>

The price of a diamond has a direct correlation with its dimensions (and hence with the carat, since the weight of the diamonds is itself a function of its dimensions). It is not a straight linear correlation but an exponential one.
There are other relevant features which also influence its price, such as color, clarity and cut. A pairplot of these attributes can be useful in inspecting these relations:

<p align="center"><img src="images/output.png" alt="pp"  width="75%"></p>

---

## Modelling



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