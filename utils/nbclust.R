library(NbClust)

data <- read.csv("C:/Users/Szokolics Dániel/Desktop/Business Analytics/Data Science II/Assignment 1/data/usarrests_to_clust.csv")
clusts <- NbClust(data[2:5], method = 'kmeans')
