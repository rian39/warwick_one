# In the data: interdisciplinary modes of machine learning


Adrian Mackenzie

Sociology Department, Lancaster University

a.mackenzie@lancaster.ac.uk

---------------------------

Perhaps our knowledge is distorted unless we can comprehend its essential connection with happenings which involve spatial relationships of fifteen dimensions (A.N. Whitehead, *Modes of Thought*, 78)


Imagine you are walking through a forest of interarticulated branches. Some are covered with ice or snow, and the suns melts their touching tips to reveal space between. Some are so thickly brambled they seem solid; others are oddly angular in nature, like esplanaded trees.  Some of the trees are wild, some have been cultivated ... Helicopters flying overhead can quickly tell your many types of each, even each leaf, there are in the world, but they cannot yet give you a guidebook for bird-watching or forestry conservation. There is a lot of underbrush and a complex ecology of soil bacteria, flora and fauna.  ... Now imagine that the forest is a huge information space and each of the trees and bushes are classification systems.  ... Your job is to describe this forest. [@bowker_sorting_1999,31-32]

----------------------------------------

* Edward Snowdon, GCHQ, NSA, Tesco ClubCard, dunnhumby, Beyond Analysis, Amazon, Facebook, Walmart, ...
* Trevor Hastie, Rob Tibshirani, Jerome Friedman, Andrew Ng, Toby Segharan, Hilary Mason, Rachel Schutt,  Heather Arthur  ...
* Linear regression model, decision tree, Random Forest (TM), *k*-nearest neighbours, neural network, support vector machine ...

-----------------------------------------------

"If most things that could happen don't happen, then we are far better  off trying first to find local patterns in data and only then looking for regularities among those patterns. Indeed, it is for this reason that cluster analysis and scaling, not regression, dominate big-money social science -- market research -- where the aim is to find, understand, and exploit strong local patterns.  For these are methods that seek clumps and partitions of data and make no attempt to write general transformations" [@abbott_time_2001, 241]


"The sheer density of the collisions of classification schemes in our lives calls for a new kind of science, a new set of metaphors, linking traditional social science and computer and information science. We need a topography of things such as the distribution of ambiguity.  ...  It will also use the best of object-oriented programming and other areas of computer science to describe this territory" [@bowker_sorting_1999, 31].

----------------------------------

## Key argument about *machine learning* based on 2 word plays:

1. people *vectorize* techniques  - to carry (vector); to flatten a matrix

```r
rectangular_matrix = matrix(seq(1:30), ncol = 6)
rectangular_matrix
```

```
##      [,1] [,2] [,3] [,4] [,5] [,6]
## [1,]    1    6   11   16   21   26
## [2,]    2    7   12   17   22   27
## [3,]    3    8   13   18   23   28
## [4,]    4    9   14   19   24   29
## [5,]    5   10   15   20   25   30
```

```r
as.vector(rectangular_matrix)
```

```
##  [1]  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
## [24] 24 25 26 27 28 29 30
```

2. techniques *optimize* people - 'make better'; engender optimism

--------------------------------------------

## People vectorize techniques

* [Hastie, Tibshirani & Friedman](figure/hastie.jpg), ~ 17,000 citations, #3 best seller on Amazon 'data-mining', and 'machine learning' sales lists
* Andrew Ng (Coursera co-founder), CS229 'Machine Learning' course at Stanford University, ~500,000 views since 2008
* Toby Segharan, [*Programming Collective Intelligence*](figure/segaran.jpg) (2007),  #1 best seller on Amazon 'machine learning' sales list
* [Hilary Mason at Bacon](http://www.hilarymason.com/presentations-2/devs-love-bacon-everything-you-need-to-know-about-machine-learning-in-30-minutes-or-less/) 
* [Cath O'Neill & Rachel Schutt from Johnson Research Labs](http://columbiadatascience.com/blog/)
*  Heather Arthur, ['essentially machine learning algorithms are better programmers than you'](http://www.youtube.com/watch?v=uZqXc1E91mE&feature=youtu.be); 00:03:35

------------------------------------------

## Techniques optimize people

### classic dataset: 'iris' R.A. Fisher (1936)

[1] 5
<TABLE border=1>
<TR> <TH>  </TH> <TH> Sepal.Length </TH> <TH> Sepal.Width </TH> <TH> Petal.Length </TH> <TH> Petal.Width </TH> <TH> Species </TH>  </TR>
  <TR> <TD align="right"> 1 </TD> <TD align="right"> 5.10 </TD> <TD align="right"> 3.50 </TD> <TD align="right"> 1.40 </TD> <TD align="right"> 0.20 </TD> <TD> setosa </TD> </TR>
  <TR> <TD align="right"> 2 </TD> <TD align="right"> 4.90 </TD> <TD align="right"> 3.00 </TD> <TD align="right"> 1.40 </TD> <TD align="right"> 0.20 </TD> <TD> setosa </TD> </TR>
  <TR> <TD align="right"> 3 </TD> <TD align="right"> 4.70 </TD> <TD align="right"> 3.20 </TD> <TD align="right"> 1.30 </TD> <TD align="right"> 0.20 </TD> <TD> setosa </TD> </TR>
  <TR> <TD align="right"> 4 </TD> <TD align="right"> 4.60 </TD> <TD align="right"> 3.10 </TD> <TD align="right"> 1.50 </TD> <TD align="right"> 0.20 </TD> <TD> setosa </TD> </TR>
  <TR> <TD align="right"> 5 </TD> <TD align="right"> 5.00 </TD> <TD align="right"> 3.60 </TD> <TD align="right"> 1.40 </TD> <TD align="right"> 0.20 </TD> <TD> setosa </TD> </TR>
   </TABLE>

```r

data(iris)


library(rpart)
rpart(Species ~ ., iris)
```

n= 150 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 150 100 setosa (0.33333 0.33333 0.33333)  
  2) Petal.Length< 2.45 50   0 setosa (1.00000 0.00000 0.00000) *
  3) Petal.Length>=2.45 100  50 versicolor (0.00000 0.50000 0.50000)  
    6) Petal.Width< 1.75 54   5 versicolor (0.00000 0.90741 0.09259) *
    7) Petal.Width>=1.75 46   1 virginica (0.00000 0.02174 0.97826) *

```r
ir.rp = rpart(Species ~ ., iris)
ir.rp
```

n= 150 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 150 100 setosa (0.33333 0.33333 0.33333)  
  2) Petal.Length< 2.45 50   0 setosa (1.00000 0.00000 0.00000) *
  3) Petal.Length>=2.45 100  50 versicolor (0.00000 0.50000 0.50000)  
    6) Petal.Width< 1.75 54   5 versicolor (0.00000 0.90741 0.09259) *
    7) Petal.Width>=1.75 46   1 virginica (0.00000 0.02174 0.97826) *

```r

par(mfrow = c(1, 2), xpd = NA)
plot(ir.rp, main = "tree recursive partitioning of iris")
text(ir.rp, cex = 0.8, use.n = TRUE)
table(iris$Species)  # is data.frame with 'Species' factor
```

    setosa versicolor  virginica 
        50         50         50 

```r
iS <- iris$Species == "setosa"
iV <- iris$Species == "versicolor"
op <- par(bg = "bisque")
matplot(c(1, 8), c(0, 4.5), type = "n", xlab = "Length", ylab = "Width", main = "Petal and Sepal Dimensions in Iris Blossoms")
matpoints(iris[iS, c(1, 3)], iris[iS, c(2, 4)], pch = "sS", col = c(2, 4))
matpoints(iris[iV, c(1, 3)], iris[iV, c(2, 4)], pch = "vV", col = c(2, 4))
legend(1, 4, c("    Setosa Petals", "    Setosa Sepals", "Versicolor Petals", 
    "Versicolor Sepals"), pch = "sSvV", col = rep(c(2, 4), 2))
```

![plot of chunk iris_tree](figure/iris_tree.png) 

"Of all the well-known learning methods, decision trees comes closest to meeting the requirements for serving as an off-the-shelf procedure for data-mining" [@hastie_elements_2009, 352]

----------------------------------------

### Building high-dimensional classifiers

#### Golub's blood pressure dataset, 1999

Many more dimensions (columns) in the data 

columns in Golub dataset: 102 

<TABLE border=1>
<TR> <TH>  </TH> <TH> \begin{sideways} RS2495368 \end{sideways} </TH> <TH> \begin{sideways} RS2292857 \end{sideways} </TH> <TH> \begin{sideways} RS6685064 \end{sideways} </TH> <TH> \begin{sideways} RS10907175 \end{sideways} </TH> <TH> \begin{sideways} RS6680471 \end{sideways} </TH> <TH> \begin{sideways} RS4648360 \end{sideways} </TH> <TH> \begin{sideways} RS3107146 \end{sideways} </TH> <TH> \begin{sideways} RS34012 \end{sideways} </TH> <TH> \begin{sideways} RS2843130 \end{sideways} </TH> <TH> \begin{sideways} RS2645072 \end{sideways} </TH> <TH> \begin{sideways} RS1107910 \end{sideways} </TH> <TH> \begin{sideways} RS17391750 \end{sideways} </TH> <TH> \begin{sideways} RS1869972 \end{sideways} </TH> <TH> \begin{sideways} RS16823103 \end{sideways} </TH> <TH> \begin{sideways} RS4648515 \end{sideways} </TH> <TH> \begin{sideways} RS16825081 \end{sideways} </TH> <TH> \begin{sideways} RS3766180 \end{sideways} </TH> <TH> \begin{sideways} RS10907187 \end{sideways} </TH> <TH> \begin{sideways} RS6603781 \end{sideways} </TH> <TH> \begin{sideways} RS385039 \end{sideways} </TH> <TH> \begin{sideways} RS7512269 \end{sideways} </TH> <TH> \begin{sideways} RS7519837 \end{sideways} </TH> <TH> \begin{sideways} RS2840528 \end{sideways} </TH> <TH> \begin{sideways} RS3122922 \end{sideways} </TH> <TH> \begin{sideways} RS2985862 \end{sideways} </TH> <TH> \begin{sideways} RS12045693 \end{sideways} </TH> <TH> \begin{sideways} RS2272908 \end{sideways} </TH> <TH> \begin{sideways} RS626479 \end{sideways} </TH> <TH> \begin{sideways} RS9786963 \end{sideways} </TH> <TH> \begin{sideways} RS11260562 \end{sideways} </TH> <TH> \begin{sideways} RS307378 \end{sideways} </TH> <TH> \begin{sideways} RS10910093 \end{sideways} </TH> <TH> \begin{sideways} RS6684865 \end{sideways} </TH> <TH> \begin{sideways} RS10909855 \end{sideways} </TH> <TH> \begin{sideways} RS10910097 \end{sideways} </TH> <TH> \begin{sideways} RS6603803 \end{sideways} </TH> <TH> \begin{sideways} RS6672353 \end{sideways} </TH> <TH> \begin{sideways} RS7527871 \end{sideways} </TH> <TH> \begin{sideways} RS897635 \end{sideways} </TH> <TH> \begin{sideways} RS4075116 \end{sideways} </TH> <TH> \begin{sideways} RS2905036 \end{sideways} </TH> <TH> \begin{sideways} RS16823335 \end{sideways} </TH> <TH> \begin{sideways} RS10910060 \end{sideways} </TH> <TH> \begin{sideways} RS12119163 \end{sideways} </TH> <TH> \begin{sideways} RS4245756 \end{sideways} </TH> <TH> \begin{sideways} RS2132303 \end{sideways} </TH> <TH> \begin{sideways} RS12049543 \end{sideways} </TH> <TH> \begin{sideways} RS4486391 \end{sideways} </TH> <TH> \begin{sideways} RS2246732 \end{sideways} </TH> <TH> \begin{sideways} RS2843129 \end{sideways} </TH> <TH> \begin{sideways} RS16823350 \end{sideways} </TH> <TH> \begin{sideways} RS4648402 \end{sideways} </TH> <TH> \begin{sideways} RS3753242 \end{sideways} </TH> <TH> \begin{sideways} RS6659552 \end{sideways} </TH> <TH> \begin{sideways} RS1980789 \end{sideways} </TH> <TH> \begin{sideways} RS3855951 \end{sideways} </TH> <TH> \begin{sideways} RS2296442 \end{sideways} </TH> <TH> \begin{sideways} RS2842933 \end{sideways} </TH> <TH> \begin{sideways} RS2377041 \end{sideways} </TH> <TH> \begin{sideways} RS7513222 \end{sideways} </TH> <TH> \begin{sideways} RS729045 \end{sideways} </TH> <TH> \begin{sideways} RS3737628 \end{sideways} </TH> <TH> \begin{sideways} RS2980300 \end{sideways} </TH> <TH> \begin{sideways} RS262683 \end{sideways} </TH> <TH> \begin{sideways} RS6603791 \end{sideways} </TH> <TH> \begin{sideways} RS10909872 \end{sideways} </TH> <TH> \begin{sideways} RS12023660 \end{sideways} </TH> <TH> \begin{sideways} RS3890745 \end{sideways} </TH> <TH> \begin{sideways} RS2027264 \end{sideways} </TH> <TH> \begin{sideways} RS6688000 \end{sideways} </TH> <TH> \begin{sideways} RS3107157 \end{sideways} </TH> <TH> \begin{sideways} RS2887286 \end{sideways} </TH> <TH> \begin{sideways} RS4648633 \end{sideways} </TH> <TH> \begin{sideways} RS2281173 \end{sideways} </TH> <TH> \begin{sideways} RS10910099 \end{sideways} </TH> <TH> \begin{sideways} RS4474198 \end{sideways} </TH> <TH> \begin{sideways} RS7545940 \end{sideways} </TH> <TH> \begin{sideways} RS10910061 \end{sideways} </TH> <TH> \begin{sideways} RS9442385 \end{sideways} </TH> <TH> \begin{sideways} RS7548727 \end{sideways} </TH> <TH> \begin{sideways} RS3736330 \end{sideways} </TH> <TH> \begin{sideways} RS262641 \end{sideways} </TH> <TH> \begin{sideways} RS897632 \end{sideways} </TH> <TH> \begin{sideways} RS3128309 \end{sideways} </TH> <TH> \begin{sideways} RS2645081 \end{sideways} </TH> <TH> \begin{sideways} RS2606414 \end{sideways} </TH> <TH> \begin{sideways} RS897620 \end{sideways} </TH> <TH> \begin{sideways} RS1973906 \end{sideways} </TH> <TH> \begin{sideways} RS7540231 \end{sideways} </TH> <TH> \begin{sideways} RS2803285 \end{sideways} </TH> <TH> \begin{sideways} RS12084736 \end{sideways} </TH> <TH> \begin{sideways} RS12031614 \end{sideways} </TH> <TH> \begin{sideways} RS262680 \end{sideways} </TH> <TH> \begin{sideways} RS7511905 \end{sideways} </TH> <TH> \begin{sideways} RS4040617 \end{sideways} </TH> <TH> \begin{sideways} RS12063033 \end{sideways} </TH> <TH> \begin{sideways} RS1496555 \end{sideways} </TH> <TH> \begin{sideways} RS16824948 \end{sideways} </TH> <TH> \begin{sideways} RS897631 \end{sideways} </TH> <TH> \begin{sideways} RS9782915 \end{sideways} </TH> <TH> \begin{sideways} height \end{sideways} </TH> <TH> \begin{sideways} bloodpressure \end{sideways} </TH>  </TR>
  <TR> <TD align="right"> 1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right"> 171.19 </TD> <TD> normal </TD> </TR>
  <TR> <TD align="right"> 2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   2 </TD> <TD align="right">  </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">  </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">  </TD> <TD align="right">   0 </TD> <TD align="right"> 179.25 </TD> <TD> normal </TD> </TR>
  <TR> <TD align="right"> 3 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right"> 188.23 </TD> <TD> normal </TD> </TR>
  <TR> <TD align="right"> 4 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">  </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">  </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right"> 175.44 </TD> <TD> high </TD> </TR>
  <TR> <TD align="right"> 5 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   2 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   1 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right">   0 </TD> <TD align="right"> 177.98 </TD> <TD> normal </TD> </TR>
   </TABLE>

- the Golub blood pressure data from 1999 has over 100 columns rather than 5. This is not especially wide, but radically changes the structures that might be found in the data. 

![plot of chunk name](figure/name.png) 


#### Infrastructural re-dimensioning: RF-ACE

For hundreds of thousands or millions of dimensional data?

[Google Compute Engine + Institute of Systems Biology, June 2012](http://www.youtube.com/watch?v=3igX-ebL-PY)

> The world's 3rd largest supercomputer *learns associations* between genomic features' [@anthony_google_2012]

[RF-ACE; Random Forest- Artificial Contrasts with Ensembles](https://code.google.com/p/rf-ace/)

Urs Hölzle, Senior Vice President of Infrastructure at Google 'then went even further and scaled the application to run on 600,000 cores across Google’s global data centers' [@google_inc_behind_2012]. 

---------------------------------------

### Partial observers and dimensional explosion

Ways of moving through high-dimensional spaces:

#### A: Stay very local?

*k-nn*: *k*-nearest neighbours



```r
library(animation)
oopt = ani.options(interval = 2, nmax = ifelse(interactive(), 10, 2))
x = matrix(c(rnorm(80, mean = -1), rnorm(80, mean = 1)), ncol = 2, byrow = TRUE)
y = matrix(rnorm(20, mean = 0, sd = 1.2), ncol = 2)
knn.ani(train = x, test = y, cl = rep(c("first class", "second class"), each = 40), 
    k = 30)
```

![plot of chunk ml-animation](figure/ml-animation1.png) ![plot of chunk ml-animation](figure/ml-animation2.png) ![plot of chunk ml-animation](figure/ml-animation3.png) ![plot of chunk ml-animation](figure/ml-animation4.png) ![plot of chunk ml-animation](figure/ml-animation5.png) ![plot of chunk ml-animation](figure/ml-animation6.png) ![plot of chunk ml-animation](figure/ml-animation7.png) ![plot of chunk ml-animation](figure/ml-animation8.png) 


#### B: Optimize a 'loss' function (a.k.a 'cost function') that 'weights' some dimensions


[kittydar](http://harthur.github.io/kittydar/): neural network cat face classifier

![plot of chunk gradient_desc](figure/gradient_desc1.png) ![plot of chunk gradient_desc](figure/gradient_desc2.png) ![plot of chunk gradient_desc](figure/gradient_desc3.png) ![plot of chunk gradient_desc](figure/gradient_desc4.png) ![plot of chunk gradient_desc](figure/gradient_desc5.png) ![plot of chunk gradient_desc](figure/gradient_desc6.png) ![plot of chunk gradient_desc](figure/gradient_desc7.png) ![plot of chunk gradient_desc](figure/gradient_desc8.png) ![plot of chunk gradient_desc](figure/gradient_desc9.png) ![plot of chunk gradient_desc](figure/gradient_desc10.png) ![plot of chunk gradient_desc](figure/gradient_desc11.png) ![plot of chunk gradient_desc](figure/gradient_desc12.png) 


Weights of nodes in neural network: optimize a loss function using gradient descent

----------------------------------------------
## Machine learning as 'cruel optimisation' or 'repetition of originary subordination'?

Unstable circuits of vectorization and optimisation

Whitehead: machine learning doesn't eschew pluri-dimensionality


Bowker & Star: can  'new kind of science' of 'distribution of ambiguity' acknowledge machine learning already in us?

The terms by which we are hailed are rarely the ones we choose (and even when we try to impose protocols on how we are to be named, they usually fail); but these terms we never really choose are the occasion for something we might still call agency, the repetition of an originary subordination for another purpose, one whose future is partially open. (Judith Butler, _Excitable Speech_, 1997, 38)

Cruel optimism is the condition of maintaining an attachment to a problematic object in advance of its loss. ... One makes affective bargains about the costliness of one’s attachments, usually unconscious ones, most of which keep one in proximity to the scene of desire/attrition (Lauren Berlant, _Cruel Optimism_, 2007, 21) 



