# In the data: interdisciplinary modes of machine learning

Adrian Mackenzie

Sociology Department, Lancaster University

a.mackenzie@lancaster.ac.uk

## Abstract

This paper explores  ways of thinking about digital data that lie somewhere between blithe faith and critical dismissal. It focuses on the machine learning, an increasingly widespread bundle of techniques and approaches that lies at the centre of contemporary data processing. Machine learning is used to program computers to find patterns, associations, and correlations, to classify events and make predictions on a large scale. As a set of techniques for classifying and predicting, machine learning lies close to centre of calculation in social network media, finance markets, robotics, and contemporary sciences such as genomics and epidemiology. This paper will discuss who is doing machine learning, who could do machine learning, and how they might do it differently. 

## The problem of the forest

Writing in the late 1990s, as the internet and Web were transforming library and information science, Geoffrey Bowker and Susan Leigh Star presented a 'topography of the distribution of ambiguity':

> Imagine you are walking through a forest of interarticulated branches. Some are covered with ice or snow, and the suns melts their touching tips to reveal space between. Some are so thickly brambled they seem solid; others are oddly angular in nature, like esplanaded trees.  Some of the trees are wild, some have been cultivated ... Helicopters flying overhead can quickly tell your many types of each, even each leaf, there are in the world, but they cannot yet give you a guidebook for bird-watching or forestry conservation. There is a lot of underbrush and a complex ecology of soil bacteria, flora and fauna.  ... Now imagine that the forest is a huge information space and each of the trees and bushes are classification systems.  ... Your job is to describe this forest. [@bowker_sorting_1999,31-32]

In *Sorting Things Out: Classification and its Consequences,* they describe some of the texture of the interlocking systems of classifications that densely grid contemporary lifeworlds. The trees -- the classification systems -- are hybrid physical-logical entities, criss-crossed by standards, practices and work-arounds, and deeply embedded in infrastructures. In the years since, the density of classification systems has only increased in many different domains. One powerful dynamic here has been the automation of classification systems. Not only is classification done by devices, classification systems -- the trees -- are produced by devices or machines. 

This is the horizon of the contemporary lifeworld. Someone, perhaps not us, not yet, has helicopters. Probably they are a business like Walmart, amazon, google or Facebook. They might be a State, like the United States or the United Kingdom. Perhaps we will have the data. There is, after all, a lot of data already available, and more all the time. It may be that civil society, non-governmental groups like Wikileaks, or various forms of massive data leak will render valuable data available. Or maybe just the increasingly powerful techniques of record linkage will allow comprehensive classification systems to emerge.   
- we on the other hand, are mostly in the forest of classification devices that generate and organise data. What do we do there? 
- the problem is that at the moment we have few  ways of thinking concretely about what you might do, or ways of articulating optimism about data with understandings of what it might cost to have the data. A couple of possibilities present themselves.  

A first possibility is that we link traditional social science in its studies of identities, differences, membership and belonging  via utterances, practices and images to systems of classification. This is what Bowker and Star themselves recommend:
 
> The sheer density of the collisions of classification schemes in our lives calls for a new kind of science, a new set of metaphors, linking traditional social science and computer and information science. We need a topography of things such as the distribution of ambiguity.  ...  It will also use the best of object-oriented programming and other areas of computer science to describe this territory [@bowker_sorting_1999, 31].

While they advocate a mixture of methods to describe the forest, they don't themselves don't explicitly include machine-made classifications in this science. This is partly because machine learning classifiers were less common in the late 1990s than they are today, and also because it was not possible for either social scientists or computer scientists to easily build those classifiers. 

The newly available capacity to build automatic classification systems suggests a second possibility. A second possibility is that we could look for patterns in the trees, branches and paths of information using automated classification systems. Andrew Abbott, whose work has received much more attention in the wake of [@savage_contemporary_2009], argued more than  that a decade ago  that we should adopt pattern-based approaches to working with data:
	
> If most things that could happen don’t happen, then we are far better  off trying first to find local patterns in data and only then looking for regularities among those patterns. Indeed, it is for this reason that cluster analysis and scaling, not regression, dominate big-money social science — market research — where the aim is to find, understand, and exploit strong local patterns.  For these are methods that seek clumps and partitions of data and make no attempt to write general transformations [@abbott_time_2001, 241]
<<<<<<< HEAD

So Abbott is advocating, effectively, building 'classifiers' for the data.  Pattern discovery or cluster analysis are effectively automated classification systems. Patterns, for instance, are spatio-temporal segmentations of the world, and thus support classification into categories (fits the pattern vs. does not fit the pattern). 
- But there are some problems with Abbott's diagnosis. The first is an empirical one: 'big money social science' such as business analytics intimately and increasingly uses regression models (especially in the form of logistic regression models that classify). In any case, the practices of linear modelling have been heavily renovated, and proliferated in a number of different directions especially in the field of machine learning. So the contrast that Abbott sees between pattern-based approaches and linear model or regression-based approaches to data is not so empirically well-grounded. The largely North American social sciences that Abbott criticises for their linear modelling of social data may or may not continue in their practices (augmented by social network analysis and heavy doses of with Bayesian statistics). The broader point here is that the simple opposition between pattern or cluster-based approaches and linear models of reality does not sufficiently to my mind orient us to the shifts in data practice that have been happening in the last decade.


There are problems in moving between these two different images of the forest of classification systems and the locally patterned fields of data. But it might not be too ambitious to try to see if the shifts in technique that Abbott proposes could help perform the descriptive task that Bowker and Star envisaged.  Ironically, growth in tree-based methods of classification, and the emergence of techniques such as random forests [@breiman_random_2001] around the time Bowker and Star were writing suggest that interest in growing tree (that is, classification systems) has continued to be strong. 


## Learning from data -- if you had all the data, what then?

I want to present several vignettes that illustrate what statisticians and computer scientists do when they search for local patterns in data, and when they make classifiers, whether it is for business, science, medicine or anything else. 

At the time Abbott was writing and much more so today, classification systems relied on what is called data mining, knowledge discovery or currently, _machine learning_. These terms reflect different scientific, govermental and commercial interests in data and patterns. Mathematically, they are all linked by the idea that they are *finding an approximation to the function that generated the data.* Nearly all of the models and techniques used, whether it is clustering, linear modelling, neural networks, support vector machines, random forests, topic modelling, etc., etc., can be seen as attempts to find the function that generated the data. Alternately, the variety of techniques are framed in terms of 'pattern recognition,' a term that stems from artificial intelligence and computer science. 

For present purposes, the first issue here is the shape of the data. The shape of datasets is widely discussed in machine learning, and shape perhaps more than the size (as in 'big') matters to how algorithms go about finding structures or patterns. It is shape too that occasions much of the infrastructural re-dimensioning we have just seen displayed. 

## data is wide, dirty and mixed: 3 million features

- The shape of contemporary datasets is sometimes described as 'wide, dirty and mixed'. Each of these terms carries interesting connotations, and a figurative analysis of data talk, which often uses frontier language, would be quite useful. I'm not going to offer that here, but just point to some examples of what they mean practically.

- Statistics textbooks and statistical practice are full of datasets like this one: classic dataset: 'iris' R.A. Fisher (1936)


```
## [1] 5
```

```
## \begin{table}[ht]
## \centering
## \begin{tabular}{rrrrrl}
##   \hline
##  & Sepal.Length & Sepal.Width & Petal.Length & Petal.Width & Species \\ 
##   \hline
## 1 & 5.10 & 3.50 & 1.40 & 0.20 & setosa \\ 
##   2 & 4.90 & 3.00 & 1.40 & 0.20 & setosa \\ 
##   3 & 4.70 & 3.20 & 1.30 & 0.20 & setosa \\ 
##   4 & 4.60 & 3.10 & 1.50 & 0.20 & setosa \\ 
##   5 & 5.00 & 3.60 & 1.40 & 0.20 & setosa \\ 
##    \hline
## \end{tabular}
## \end{table}
```

- Fisher's widely cited 'iris' data is often used to demonstrate various statistical techniques.
- In machine learning textbooks such as [@hastie_elements_2009], iris is mentioned, and often used to demonstrate basic techniques such as trees, for instance.


```r

data(iris)


library(rpart)
rpart(Species ~ ., iris)
```

```
n= 150 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 150 100 setosa (0.33333 0.33333 0.33333)  
  2) Petal.Length< 2.45 50   0 setosa (1.00000 0.00000 0.00000) *
  3) Petal.Length>=2.45 100  50 versicolor (0.00000 0.50000 0.50000)  
    6) Petal.Width< 1.75 54   5 versicolor (0.00000 0.90741 0.09259) *
    7) Petal.Width>=1.75 46   1 virginica (0.00000 0.02174 0.97826) *
```

```r
ir.rp = rpart(Species ~ ., iris)
ir.rp
```

```
n= 150 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

1) root 150 100 setosa (0.33333 0.33333 0.33333)  
  2) Petal.Length< 2.45 50   0 setosa (1.00000 0.00000 0.00000) *
  3) Petal.Length>=2.45 100  50 versicolor (0.00000 0.50000 0.50000)  
    6) Petal.Width< 1.75 54   5 versicolor (0.00000 0.90741 0.09259) *
    7) Petal.Width>=1.75 46   1 virginica (0.00000 0.02174 0.97826) *
```

```r

par(mfrow = c(1, 2), xpd = NA)
plot(ir.rp, main = "tree recursive partitioning of iris")
text(ir.rp, cex = 0.8, use.n = TRUE)
table(iris$Species)  # is data.frame with 'Species' factor
```

```

    setosa versicolor  virginica 
        50         50         50 
```

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

- Here is an example of the kind of tree we might find in Bowker and Star's forest. It is a classification tree produced by 'recursive partitioning.' This tree classifies species of iris on the basis of the data on the length and width of petal and sepals. What is interesting about this classifier is that it knows nothing much about irises, but manages to classify them by finding associations between species and petals/sepals.  The decision tree shown here is in many ways a classic Aristotlean classification system. It is monothetic in that at every point in a classification problem, one only needs to make a binary choice. Is the length of the petal less that 2.45 cm? If so, it is a setosa. If not, is the petal width less than 1.75cm? If so, it is a versicolor. Otherwise, it is a virginica.

Decision trees are perhaps the most widely used machine learning technique in commercial data mining and perhaps also in biomedical research.  They can handle many different kinds of data and they are easy to understand. Hastie et. al. suggest that 'of all the well-known learning methods, decision trees comes closest to meeting the requirements for serving as an off-the-shelf procedure for data-mining' [@hastie_elements_2009, 352]. The popular random forest technique that lies behind the Google Compute demonstration constructs many decision trees, uses them to classify the data, and then uses a majority voting system to produce a predicted classification. 

## From irises to genomes: infrastructural re-dimensioning

But the iris dataset is different to  most contemporary datasets. They often have many more columns. For instance, in the late 1990s, when biomedical data from genechips or microarrays started to become widely available, machine learning techniques were applied to their analysis. A typical biological dataset today  would be Golub's c.1999 'blood pressure' dataset, or today, a cancer genome. 

### Golub's blood pressure dataset, 1999

```
## columns in Golub dataset: 102
```

```
## \begin{table}[ht]
## \centering
## \begin{tabular}{rrrrrrrl}
##   \hline
##  & \begin{sideways} RS2495368 \end{sideways} & \begin{sideways} RS2292857 \end{sideways} & \begin{sideways} RS6685064 \end{sideways} & \begin{sideways} RS897631 \end{sideways} & \begin{sideways} RS9782915 \end{sideways} & \begin{sideways} height \end{sideways} & \begin{sideways} bloodpressure \end{sideways} \\ 
##   \hline
## 1 &   0 &   0 &   0 &   0 &   0 & 171.19 & normal \\ 
##   2 &   0 &   0 &   0 &  &   0 & 179.25 & normal \\ 
##   3 &   0 &   0 &   0 &   1 &   0 & 188.23 & normal \\ 
##   4 &   0 &   0 &   0 &   0 &   0 & 175.44 & high \\ 
##   5 &   0 &   0 &   0 &   0 &   0 & 177.98 & normal \\ 
##    \hline
## \end{tabular}
## \end{table}
```

- the Golub blood pressure data from 1999 has over 100 columns rather than 5. This is not especially wide, but radically changes the structures that might be found in the data. 

![plot of chunk name](figure/name.png) 

- as we can see, the number of different models that you might build to look at patterns between blood pressure, height and this genetic data very quickly becomes huge when there are numerous variables or 'features' in the data. 

### Google compute/Institute for Systems Biology, 2012

- cancer genomes + google compute

- begin with someone/something that does have all the data -- Google and certain genomic scientists work together on cancer genomes;
- Let me illustrate what I mean by *infrastructural re-dimensioning*:

> [Google Compute Engine + Institute of Systems Biology, June 2012](http://www.youtube.com/watch?v=3igX-ebL-PY)

> The world's 3rd largest supercomputer *learns associations* between genomic features' [@anthony_google_2012]


> The Google Compute Engine, a globally-distributed ensemble of computers,  was briefly turned over to exploration of cancer genomics during 2012, and publicly demonstrated during the annual Google I/O conference. Midway through the demonstration, visualized as a circular genome, the speaker, Urs Hölzle, Senior Vice President of Infrastructure at Google 'then went even further and scaled the application to run on 600,000 cores across Google’s global data centers' [@google_inc_behind_2012]. The world's '3rd largest supercomputer', as it was called by TechCrunch, a prominent technology blog,  'learns associations between genomic features' [@anthony_google_2012].

The scientists are from Institute for Systems Biology (ISB), University of Washington. Its a fairly well-established and long-standing icon of big data science, with some Nobel Prizes, etc, associated with it. [SHOW Image of building]

Urs Hölzle, the Vice-President of Infrastructure at Google shows the fruits of their collaboration to an audience of developers at Google I/O in San Francisco, 2012. His title is significant -- VP of _Infrastructure_. [SHOW the video]

On the one hand, the scientists at ISB have the data. They are somewhat like us -- they are academics. They are different to us in that they have the best data possible on cancer genomes. Their data reaches down into clinics, it comes from the best scientific instruments money can buy, and they benefit from PhDs, programmers, engineers, disk drives and an assortment of other services at their beck and call.  

This example is the helicopter-style example. One thing that people do when they have all the data is build huge data structures and data processing infrastructures that allow them to fly over data. 

Yet, they need or want google. Or maybe Google wants them for something. What the Google I/O conference shows is the availability of flexible instrastructure that can be quickly, rapidly made available on demand. It can shift orders of magnitude in size in seconds. Not many infrastructures can do that. But even if it can, why are such infrastructures needed? It is as if Oxford St in London could suddenly switch between a pedestrian mall and an 8 lane motorway. What form of life or sociality would require that mutability?

In the Google Compute case, it is hard to see what machine learning is doing. We are told that it is discovering associations between molecular features on human cancer genomes, as well as gene expression, patient attributes, and mutations.  These associations measure how likely it is that features are related to each other. And we know that is using the Random Forest algorithm, which is, as I've said a typical machine learning algorithm.   

In this case, what the scientists want to do is to sift through massive numbers of associations between different parts of a genome to identify patterns and processes associated with mutagenesis. The algorithm they use is called [RF-ACE; Random Forest- Artificial Contrasts with Ensembles](https://code.google.com/p/rf-ace/), and an implementation in the R statistical programming language is freely available. While it is state-of-the-art, this algorithm has many kinship with the decision trees I showed above. It is again an example of an unsupervised learning algorithm that tries to *find structure* in a data-set without any prior knowledge about the data. 


- While the data we have seen is wide, it is not necessarily very mixed or dirty. Mixed means that the types of data are varied. A mixed dataset includes numbers, categories, text, and so forth. Dirty data, as you can imagine, describes data lacks consistency, that might have holes in it or is somehow noisy.

## Different ways of finding structure in the data

- The point here is that finding structure in data means working out which bits of data are related to which other bit. Typically a row of data in a dataset relates to one thing. The thing might be a person, an event, a sample, a place, but machine learning is rather indifferent to that. What matters is finding, amidst the many possible combinations of features or variables those combinations that carry some weight, that pattern the other data, or that stand out. 
- In machine learning broadly speaking, there are two different ways of navigating or finding these patterns. The classification trees shown above and many other so-called 'unsupervised learning' approaches generally use a Aristotlean approach to classification and prediction. They try to find ways of lumping things together so that there is less difference in the group than between the groups. They try to find local patches of similarity in the data, and classify things according to which patch they belong to. 

- Possible the simplest machine learning technique that illustrates a prototypical mode of classification would be _k-nn_ or '_k_ nearest neighbours.' This technique is widely used to build classifiers and do classification. The animated vignette uses randomly generated values that belong to one of two classes, 'first class' or 'second class.' Class membership is assigned randomly. As in many machine learning problems, the data shown here comes from two sources. The algorithm uses 'training set' to learn something about how the two variables $x, y$ relate to class membership. When the algorithm encounters new cases -the so-called 'test dataset' - it calculates the distance between the $x, y$ values of the new case and all the points in the training set, selects the $k$ nearest neighbours, counts how many of the nearest neighbours belong to each of the classes, and then assigns the new case to the majority class. This is a very simple algorithm that performs classification remarkably well, without much statistical calculation or statistical modelling. It is really just putting like with like. The intuition here is that new events are best put together with those they lie near to: the prototypes. 


```r
library(animation)
oopt = ani.options(interval = 2, nmax = ifelse(interactive(), 10, 2))
x = matrix(c(rnorm(80, mean = -1), rnorm(80, mean = 1)), ncol = 2, byrow = TRUE)
y = matrix(rnorm(20, mean = 0, sd = 1.2), ncol = 2)
knn.ani(train = x, test = y, cl = rep(c("first class", "second class"), each = 40), 
    k = 30)
```

![plot of chunk ml-animation](figure/ml-animation1.png) ![plot of chunk ml-animation](figure/ml-animation2.png) ![plot of chunk ml-animation](figure/ml-animation3.png) ![plot of chunk ml-animation](figure/ml-animation4.png) ![plot of chunk ml-animation](figure/ml-animation5.png) ![plot of chunk ml-animation](figure/ml-animation6.png) ![plot of chunk ml-animation](figure/ml-animation7.png) ![plot of chunk ml-animation](figure/ml-animation8.png) 

In this algorithm, much hinges on how likeness is equated with neighbourliness. Like many machine learning techniques, the algorithm moves a kind of 'partial observer' around in the data. In _knn_, the observer moves around from point to point in the test dataset. At each point, it measures who or what is closest, and it counts what kind of things are nearby. Points within the neighbourhood as classified as the same as the nearest prototype From the standpoint of the partial observer, much  depends on how distance is understood, and much depends on the choice of the prototypes. The metrics on which the example shown above rely are Euclidean. Effectively, it sees the data environment as Cartesian space, in which distances can be measured in terms of _x,y_ coordinates, using Pythagoras' theorem:

>idx = rank(apply(train, 1, function(x) sqrt(sum((x - 
            test[i, ])^2))), ties.method = "random") %in% seq(k)
        vote = cl[idx]
        res = c(res, factor(names(which.max(table(vote))), levels = levels(clf), 
            labels = levels(clf)))'

- While in some settings this makes sense, Euclidean space just doesn't work well for many kinds of data. In so-called categorical data, for instance, that describe what category or class things belong to (e.g. what species, what nationality, what gender), Euclidean measures of distance produce pretty meaningless results. Various non-Euclidean distance metrics have been invented to deal with these situations. Gower's distance, proposed in 1971 [@gower_general_1971], can take into account different kinds of data, and they can expresses dissimilarities between classes of things. Even if different data types can be included in the calculation of distance, distance has many different implications and entanglements. Geographic distance does not equate to cultural distance for instance. Physical proximity does not equate to emotional intimacy. There are many ways in which extended spaces or distances are folded and manifolded by other relations that do not directly appear in linear Euclidean metrics.  A plethora of techniques for dealing with more complicated spatial forms have appeared, techniques such as spectral clustering or self-organising maps, that seek to work with entangled or nested shapes. 



### learning to classify differently


But even with this sophistication, a broader problem remains with how machine learning algorithms tend to treat things as fixed in their membership. The whole tenor of social and cultural theory in recent decades has tended to see differences, including strongly marked differences such as categories, as made through the methods of their knowing. Classes or differences in themselves are *nominal* in the sense that they are energised by the power of naming that imbues bodies and things with social existence.   As we saw with the decision trees, machine learning algorithms cut the world through successive binary splits; similarly, _knn_ classifies the world according to how many of the nearest neighbours belong to each of the two classes, but in both cases, the result is a naming that interpellates bodies. 

The role of classification systems as corporeal interpellators becomes much more obvious when we begin to examine how machine learning is moving as a set of methods. There are many possible examples here, but the point about classification systems as interpellations is perhaps easiest to see in relation to images. 

[kittydar](http://harthur.github.io/kittydar/) uses machine learning to detect cats. It was written by Heather Arthur, a software developer at Mozilla Foundation.  Kittydar is typical and unusual in certain respects. It is ususual in that Arthur works as a software developer on the Mozilla Firefox browser. Her machine learning work uses the Javascript programming language, a language much more associated with web development rather than with the somewhat macho machine learning or computationally intensive languages. Kittydar is a typical machine learning classifier, albeit one with a quite restricted application domain -- detected of forward-facing cats in photographs. Image classification is a typical machine learning application. Whether it is face recognition, hand-writing recognition, pedestrian detection or medical-image processing, classification of  images or 'machine vision' is a topic of long-standing interest. Like many classifiers, kittydar uses 'supervised learning' techniques. In supervised learning, the algorithm is 'trained' on a set of labelled examples, and then tested against unlabelled images by asking it to label them. Kittydar implements two of the most powerful supervised learning techniques -- neural networks and support vector machines -- to identify areas of images likely to contain a cat's face.

Heather Arthur describes how Kittydar works as follows: 

	> Kittydar first chops the image up into many "windows" to test for the presence of a cat head. For each window, kittydar first extracts more tractable data from the image's data. Namely, it computes the [Histogram of Orient Gradients](http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) descriptor of the image, using the [hog-descriptor](http://github.com/harthur/hog-descriptor) library. This data describes the directions of the edges in the image (where the image changes from light to dark and vice versa) and what strength they are. This data is a vector of numbers that is then fed into a [neural network](https://github.com/harthur/brain) which gives a number from `0` to `1` on how likely the histogram data represents a cat. The neural network (the JSON of which is located in this repo) has been pre-trained with thousands of photos of cat heads and their histograms, as well as thousands of non-cats.

This algorithm is also a classifier. It is another tree in the forest, in Bowker and Star's terms. The algorithm in kittdar, however, does not make cuts across the data like the decision tree. Like the random forest, it tends to involve many branches or connections.  The detection of cats is not just an idle interest. Cat video and photographs are a high-value component of Youtube and photo-sharing websites. And the pursuit of cats in images preoccupies teams of researchers at Microsoft and Google [@le_building_2011; @bbc_google_2012]. It seems that the feeling of seeing  a cat, at least for quite a few people, constitutes a meaningful interpellation, a way of having a body, or being present, that matters. Cats give us our bodies in some way or other.

But how does a classifier treat an image of a cat as opposed to the measurements of petal size on a flower? An image of a cat is a lot more non-linear than the plot of iris data. At least, it is has many more lines, local patches, surfaces and curves than we can see in the iris data. Yet images of cats, somewhat re-processed, are what kittydar has to look at. How does a neural network look at the edges, lines and color patches of an image? 

Neural networks received a great deal of attention when they were first developed in the 1980s. Whole fields of science were transformed by these models (for instance, cognitive science was strongly influence by them), as well as certain fields of practical business activity, like the postal services where neural network-based handwriting recognition changed the mail sorting. Much agency in finding patterns was attributed to neural networks, before it was realised that training good neural network models is just as hard as any other kind of modelling, and that neural networks are in effect a kind of statistical model fitted to the data by combining various models together.

In contrast to the other machine learning techniques discussed so far -- decision trees, k-nn, random forests -- neural networks are now theorised as a non-linear combination of linear models. As statistical models, neural networks now have an explicit statistical underpinning. At the core of the neural network architecture, which is usually shows in terms of layers of nodes, is a complicated function-fitting that explores a complicated data topography, no longer traversed by straight cuts or bifurcations, but by walking up and down slopes trying to find lowpoints. As Hastie, Tibshirani and Friedman observe, despite the hype around neural networks, they are 'just nonlinear statistical models' [@hastie_elements_2009,302]. They take linear combinations of inputs as features and then model the target output as a non-linear combination of these features. 

## Yihuir on gradient descent

![plot of chunk gradient_desc](figure/gradient_desc1.png) ![plot of chunk gradient_desc](figure/gradient_desc2.png) ![plot of chunk gradient_desc](figure/gradient_desc3.png) ![plot of chunk gradient_desc](figure/gradient_desc4.png) ![plot of chunk gradient_desc](figure/gradient_desc5.png) ![plot of chunk gradient_desc](figure/gradient_desc6.png) ![plot of chunk gradient_desc](figure/gradient_desc7.png) ![plot of chunk gradient_desc](figure/gradient_desc8.png) ![plot of chunk gradient_desc](figure/gradient_desc9.png) ![plot of chunk gradient_desc](figure/gradient_desc10.png) ![plot of chunk gradient_desc](figure/gradient_desc11.png) ![plot of chunk gradient_desc](figure/gradient_desc12.png) 

The gradient descent algorithm is used to calculate the 'weights' that the neural network uses to fit the model to the data, and hence to predict for any new data what class it belongs. What I find interesting about this process of fitting the neural network is that it is very hard to see what the landscape that the model is exploring. At Hastie et. al. suggest, they are most useful where 'prediction without interpretation is the goal' [@hastie_elements_2009, 408]. 

It is hard to know how kittydar sees cats, and how it manages to recognise a cat it and what it is not. But it is clear that the dimensionality of data, the way in which it becomes more like a forest and less like a plain, more like an animal and less like a line.       

## machine learning as interdisciplinary method

A couple of observations about these vignettes:

1.  machine learning techniques move widely across disciplines, from astronomy to surveillance, from medical imaging to futures markets. In some ways that is not suprising because statistical technqiues  have longed roved widely across natural sciences, social sciences and various other practices. There may be, however, a kind of hypermobility associated with machine learning that is partly due to infrastructural -re-dimensioning, and the associated shifting dimensionality  of data.
2.  what is this hypermobility? I said above that machine learning tries to find the function that generated the data. Another way of viewing this is to see machine learning as a way of navigating high-dimensional spaces, spaces that are difficult for us to perceive, observe or represent. Many machine learning techniques try to address the fluxing dimensionality of patterns. We can see patterns easily when they are on surfaces, but hyperplanes or hypersurfaces can have patterns that we simply can't see, although we might be able to have some feeling of them. These spaces, generated by functions, are explored in machine learning, usually by finding lines or planes that cut through them, linking somethings together and separating others [^parisi].

[^parisi]: This is the key intuition developed by Luciana Parisi in her recent work [@parisi_contagious_2013]. Her promotion of algorithms as objects of abstract experience is consonant with my argument here. I differ from her in the importance I attribute to algorithmic information theory and its notion of randomness. She sees algorithmic processes as ruptured by non-computable bursts of randomness generated by the axiomatic undecibility (cf. [@Mackenzie, 1997] on undecibility and non-computable numbers). Importantly, I differ in the cases I draw on and how I link practice and theory.

3. Thirdly, if fluxing data dimensionality becomes navigable through machine learning, it becomes important to ask the usual kinds of science studies questions such as navigable or explorable at what cost. 

## The fluxing dimensionality of people

 - So while I haven't yet talked about how machine learning actually locates structure in data, I want to now turn to talking about people, companies and scientists doing machine learning. Who today does machine learning and at what cost? 

- Academically, the epicentre of the field lies somewhere between computer science and statistics. For instance, one of the main textbooks in the field [@hastie_elements_2009], was written by three very well-known statisticians from Stanford University. On the other hand, the Stanford computer scientist Andrew Ng is one of the most well-known figures in the field. His Youtube lectures on machine learning CS229, a postgraduate course, show viewing figures peaking at 500,000.
- SHOW [Andrew Ng, Stanford, CS229](http://www.youtube.com/watch?v=ey2PE5xi9-A) 

- The tenor of these lectures is that understanding machine learning as a mathematically grounded process of finding the function that best approximates how the data was generated offers to do things that programmers and others can't do. Ng's refers often to the Silicon Valley companies he visits, and how often they are wasting their time trying to build systems that are statistically unreliable. He adjures his students to understand the mathematical and statistical underpinnings of machine learning so that when they graduate they will be to respond to the challenges of constructing and predicting increasingly complex processes. From the perspective of Stanford or MIT,  the quintessential machine learning expert is a PhD, highly-maths literate and also able to program, who goes on to work in Silicon Valley, Wall Street or for some US government agency. 
- This does not exhaust the constituency of machine learning practitioners. Beginning around 2006, machine learning methods started appearing more broadly in software cultures. For instance, the book _Programming Collective Intelligence_ appeared in 2007 and quickly became popular as a way of thinking about how to reconfigure website and other network media platforms to deal with the greatly increased range of interactions. A series of software packages, instructional media, and events have ensued that promote machine learning technique as the way to manage, contain and optimise the flows of data not only in digital media, but in the sciences. 
- More recently, the recourse to machine learning has become more explicit. Recent books such as _Machine Learning for Hackers_, written by two social science U.S. PhD students, are symptomatic.  Many of the datascience organisations as well as hackers individually show increasing interest in machine learning as a way to program software to react in ways that cannot be anticipated in advance by programmers themselves. 
- Yet the people who do machine learning are not the same as the typical programmer or hacker. Indeed machine learning seems to coincide with a shift not only in how data is handled, but in the ways in which data-driven systems are constructed and configured. 
- Several vignettes outline this shift:
	- [Hilary Mason at Bacon](http://www.hilarymason.com/presentations-2/devs-love-bacon-everything-you-need-to-know-about-machine-learning-in-30-minutes-or-less/) 
		- [SHOW conference video]: Mason talks about the wonderful who lie someone between engineering, programming, mathematics and social science.
	- the second is the growth of data science courses that machine learning methods to the new coding classes. These courses are run at universities, industry conferences, hackathons and various online settings by statisticians as well as more conventional software industry trainers. For instance, the 'first' datascience course was  run  by by Rachel Schutt (formerly at Google) in late 2012 at Columbia University in New York. It attracted students from various departments and disciplines, and will appear as a book later this year. Not all of this course is machine learning, but substantial portions are. 
		- SHOW [Cath O'Neill & Rachel Schutt from Johnson Research Labs](http://columbiadatascience.com/blog/)
	- Heather Arthur, the software developer I showed speaking earlier makes two claims that are relevant here. [Heather Arthur on cat face detection](http://www.youtube.com/watch?v=uZqXc1E91mE&feature=youtu.be) [SHOW ] 
		- 'essentially machine learning algorithms are better programmers than you ' (00:03:35)
		- 'what is cool is that this is all running on the client side. 'A few years ago this would not have been possible' (00:17:30); 
		- Programmer's become better programmers via machine learning; at the same time, code in all settings becomes more powerful; 'all this is running on the client side' means that the machine learning algorithms (in the case of Kittydar, neural network algorithms) are much more widely distributed. They are not staying just in  the server farms, the data centres such as Google Compute. Machine learning gets into 'the clients.' 
	- We could also look at cases such as the Obama re-election data team
- While the study of recent Stanford or MIT Phds, or for that matter, the winners of machine learning competitions might not seem to be that promising as a way to respond to the complexities of machine learning techniques, it offers some really valuable short cuts for the reinvention of methods. 
	- Although machine learning models are often work as hidden layers in infrastructure, or they remain difficult to interpret precisely because they have been produced by machines, the people who learn to do machine learning are themselves less opaque. It is possible to track the kind of modelling work they do, and to impute method-technique signatures to their approach to data. People tend to keep using the same techniques they have used in the past. They invest so heavily in becoming expert in these techniques, it seems that they do not radically change course as they move between industry and academia and vice versa. So while the actual machine learning technique used in a given setting may not be obvious or openly stated, it is possible to impute techniques given then the biographical trajectory of the people involved. In certain cases, the prominence of machine learning experts or data scientists works against the obfuscation of their own methods in big data. 
		- the example of [Jimmy Lin](http://www.umiacs.umd.edu/~jimmylin/publications/index.html) - his publications on 'question answering' since 2001 give a reasonably good guide to the approach Twitter is likely to be using as it shapes responses to queries. 
	- The optimism of many of the people about data, and in particular, what I like to call their 'optimism about optimisation' takes them close to the servers (in business, government, science, finance, etc.), but at the same time is subject to its own forms of recursive erosion. When Heather Arthur says 'machine learning algorithms are better programmers that you,' she is pointing towards the horizon of machine learning in general, which is self-optimising optimisation in which programmers and data scientists' own agency is much diminished. If at the moment, these are the wonderful people, will they be the wonderful people in 10-20 years? Or will they have automated their own work out of existence? 
	- In short, people have a fluxing dimensionality, just as much as the data they produce, and this fluxing dimensionality can be reduced or augmented by machine learning. 


## What to do

-  Three observations then about the significance of machine learning:
	-  its ubiquity in science, business and government renders it a key contemporary control practice, that differs in powerful yet subtle ways from existing ways of knowing, predicting, anticipating or controlling
	- its mobility as a cluster of methods has strongly performative effects -- parts of the world are heavily re-configured through it, whether in the banal forms of intelligence apparent in smart devices (text prediction, gesture recognition, voice recognition, etc), or in the financial torsions induced by algorithmic trading. 
	- the mobility of its methods has no clear boundaries. The spread of machine learning into social sciences and humanities has started. Much of our research already implicitly depends on it (google search, etc). The question is how we move in relation to machine learning. It could be the cultural analytics route as propounded by Lev Manovich and others in the digital humanities; it could be the big data computational social science of US political scientists such as Gary King; or it could be more like the abundant hackathons that try to repurpose adn reinvest data with meaning or insights for the benefit of NGOs and charities. 
		
- Machine learning is a hard field to get into in some ways for social science and humanities researchers. There are lots of statistical, mathematical and infrastructural subleties to deal with. On the one hand, it is becoming enormously available and increasingly as Thrift would say, part of the contemporary time-space signature. Myriad mundane examples could be given. On the other hand, to get into it, to occupy the fluxing dimensionality of the data is  technically difficult, and conceptually challenging and sometimes boring. It takes very significant investments in time and attention (for instance, going through 27 hours of Youtube lectures with lots of equations written on blackboards is not trivial). 

- But I am suggesting that getting into it analytically and practically by whatever might be worthwhile, at least for some people for several reasons:
	1. It offers  a way of accompanying the fluxing dimensionality of data. In _Modes of Thought_, Alfred North Whitehead writes:

		>Perhaps our knowledge is distorted unless we can comprehend its essential connection with happenings which involve spatial relationships of fifteen dimensions [@whitehead_modes_1958, 78]
		
		In this passage, Whitehead's choice of 15 is arbitrary. I guess it just refers to a spatiality that it is hard for to imagine, even though we no doubt inhabit it from time. From the standpoint of machine learning, we often do move through high dimensional spaces. Many machine learning techniques seek to reduce the dimensionality of spatial relationships in data. These would include the many dimensional reduction strategies. But others seek to expand the dimensionality of data. And the feild as a whole tends to augment rather than diminish data dimensionality. Certain techniques artificially inject infinite dimensional spaces into the models in order to find hyperplanes that separate data. 
	
	2. Like all sciences and technologies, machine learning must contain zones of slippage, inconsistency or friction where things can happen. While it might not be us as researchers who occupy or can identify those zones most easily, in making sense of machine learning all the way down, and pointing to the structures, processes or relations at play, we help free up the possibilities of gaming the models. And actually, who else is going to do it? That is, machine learning provides a way to contest the asymmetrical distributions of agency admidst re-dimensioned infrastructures.  
	3. To some extent, as researchers we are encountering a version of the quandary I posed at the outset: what happens if you have all the data? The question is how we are going to comprehend 15D happenings, especially when a good number of those dimensions are occluded from us.  A couple of scenarios occur to me here:

		- using machine learning to impute what kind of machine learning is going on in a given setting - these could involve either labelling what techniques are likely being used, or actively experimenting with ways of perturbing the model (as for instance, the many machine learning-based attempts to do search engine optimizination do). 'Gaming the model' means figuring out how it is likely to work, and then using that to perturb it. 
		- using machine learning techniques as a way of thinking about differences, movement, shape and change
		- using machine learning
	4. Bowker and Star suggested that we needed a kind of science to deal with classification systems: 

		> The sheer density of the collisions of classification schemes in our lives calls for a new kind of science, a new set of metaphors, linking traditional social science and computer and information science. We need a topography of things such as the distribution of ambiguity.  ...  It will also use the best of object-oriented programming and other areas of computer science to describe this territory. [@bowker_sorting_1999, 31]


 
	

## Extra stuff 

- Jaron Lanier
- occupydata
- animation
- text/coding/reproducibility

- text data and shaping the world
	- James Lin, Twitter
	- e-discovery
	- leximancer

- signals -- image and sound
	- driverless car - Thrum - winner of the Darpa challenge
	- kittydar: :  vs. google cat
	- Eulerian video: how bodies encounter data
- the social scientists
		- Gary King
		- manovich -- cultural analytics
		- Savage - descriptive assemblage


## References   
