#Set working directory

#Install jsonlite package to open json format file
install.packages("jsonlite")
library(jsonlite)

#Load training and testing dataset
train = fromJSON('train.json')
str(train)
test = fromJSON('test.json')
str(test)

#Add dependent variable cuisine with NA in test dataset
test$cuisine = NA

#Combine train and test dataset
full = rbind(train, test)
str(full)

#Install NLP and tm packages to preprocess the datas
install.packages('NLP')
library(NLP)
install.packages('tm')
library(tm)

#create corpus of ingredients
corpus = Corpus(VectorSource(full$ingredients))
corpus = tm_map(corpus, tolower)
corpus[[1]]
corpus = tm_map(corpus, removePunctuation)
corpus[[1]]
corpus = tm_map(corpus, removeWords, c(stopwords('english')))
corpus[[1]]
corpus = tm_map(corpus, stripWhitespace)                
corpus[[1]]
corpus = tm_map(corpus, stemDocument)
corpus[[1]]

#Create document term matrix
frequencies = DocumentTermMatrix(corpus)
frequencies

#explore the frequency column wise and get the ingredients with highest frequency
freq = colSums(as.matrix(frequencies))
freq

ord = order(freq)
ord

freq[head(ord)]
freq[tail(ord)]

head(table(freq), 30)
tail(table(freq), 30)

#remove ingredients which occurs less than 4 times
sparse = removeSparseTerms(frequencies, 1- 4/nrow(frequencies))
dim(sparse)

#Create dataframe from sparse
newsparse = as.data.frame(as.matrix(sparse))

colnames(newsparse) = make.names(colnames(newsparse))

#Check for the popular cuisine in train dataset
table(train$cuisine)\

#Add dependent variable cuisine to dataframe newsparse
str(newsparse)
newsparse$cuisine = as.factor(c(train$cuisine, rep('italian', nrow(test))))
str(newsparse$cuisine)

#split the dataset as train and test from newsparse
mytrain = newsparse[1:nrow(train), ]
mytest = newsparse[-(1:nrow(train)), ]

#Install packages xgboost and Matrix for building models 
install.packages("xgboost")
library(xgboost)
install.packages("Matrix")
library(Matrix)

#Create matrix to train and test the models
newtrain = xgb.DMatrix(Matrix(data.matrix(mytrain[, !colnames(mytrain) %in% c('cuisine')])), label = as.numeric(mytrain$cuisine)-1)
newtest = xgb.DMatrix(Matrix(data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')])))

#Create watchlist
watchlist = list(train = newtrain, test = newtest)

#Create xgboost models with different parameters
Model1 = xgboost(data = newtrain, max.depth = 30, eta = 0.3, nrounds = 200, objective = "multi:softmax", num_class = 20, verbose = 1)
Model2 = xgboost(data = newtrain, max.depth = 20, eta = 0.2, nrounds = 250, objective = "multi:softmax", num_class = 20)
Model3 = xgboost(data = newtrain, max.depth = 25, gamma = 2, min_child_weight = 2, eta = 0.1, nrounds = 150, objective = "multi:softmax", num_class = 20, verbose = 2)

#Made predictions using above models
Prediction1 = predict(Model1, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
Prediction1.text = levels(mytrain$cuisine)[Prediction1 + 1]

Prediction2 = predict(Model2, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
Prediction2.text = levels(mytrain$cuisine)[Prediction2 + 1]

Prediction3 = predict(Model3, newdata = data.matrix(mytest[, !colnames(mytest) %in% c('cuisine')]))
Prediction3.text = levels(mytrain$cuisine)[Prediction3 + 1]

#Create Dataframe for predictions
result1 = cbind(as.data.frame(test$id), as.data.frame(Prediction1.text))
colnames(result1) = c('id', 'cuisine')
result1 = data.table::data.table(result1, key = 'id')

result2 = cbind(as.data.frame(test$id), as.data.frame(Prediction2.text))
colnames(result2) = c('id', 'cuisine')
result2 = data.table::data.table(result2, key = 'id')

result3 = cbind(as.data.frame(test$id), as.data.frame(Prediction3.text))
colnames(result3) = c('id', 'cuisine')
result3 = data.table::data.table(result3, key = 'id')

#combine all the cuisines column into one dataframe
result3$cuisine2 = result2$cuisine
result3$cuisine1 = result1$cuisine

#Using mode fun to extract the predicted value with highest frequency per id
Mode = function(x) {
  u = unique(x)
  u[which.max(tabulate(match(x, u)))]
}

x = Mode(result3[, c("cuisine", "cuisine2", "cuisine1")])
y = apply(result3,1,Mode)

#Create dataframe for the final output
final = data.frame(id= result3$id, cuisine = y)
data.table::data.table(final)

#Save the Output as csv file
write.csv(final, 'sample_submisssion.csv', row.names = FALSE)
