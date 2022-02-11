emprunt <- read.csv("Data Projet.csv", header = TRUE, sep = ",", dec = ".",
                    stringsAsFactors = TRUE)

# Verification de la bonne lecture du fichier
names(emprunt)

# Verification des modes (types) de chaque variable
str(emprunt)

# Affichage du data frame
View(emprunt)

#----------------------------------------#
# STATISTIQUES GENERALES SUR LES DONNEES #
#----------------------------------------#

# Quartiles et moyenne des variables quantitatives et effectifs par valeur des variables qualitatives
summary(emprunt)

# Quartiles et moyenne de la variable quantitative Income pour chaque valeur de Default
tapply(emprunt$income, emprunt$default, summary)

#-------------------------------------------#
# INSTALLATION ET ACTIVATION DES LIBRAIRIES #
#-------------------------------------------#

install.packages("rpart")
install.packages("rpart.plot")
install.packages("ggplot2")
install.packages("tree")
install.packages("ROCR")
installed.packages("c50")
install.packages("randomForest")
install.packages("kknn")
install.packages("e1071")
install.packages("naivebayes")
install.packages("nnet")

library(rpart)
library(rpart.plot)
library(ggplot2)
library(tree)
library(ROCR)
library(C50)
library("randomForest")
library("kknn")
library(e1071)
library(naivebayes)
library(nnet)

#-----------------------------------#
# VISUALISATION MONODIMENSIONNELLES #
#-----------------------------------#

# Diagrammes circulaire en secteurs de variables discretes
pie(table(emprunt$ed), main = "Répartition des diplomes")
pie(table(emprunt$employ), main = "Répartition de la duree d'emploi")
pie(table(emprunt$default), main = "Répartition des remboursements")


#----------------------------------------#
# HISTOGRAMMES D'EFFECTIFS DES VARIABLES #
#----------------------------------------#

# Histogramme d'effectifs de variables discrètes
qplot(ed, data=emprunt, main="Distribution des diplomes", xlab="Valeur de ed", ylab="Nombre d'instances")
qplot(default, data=emprunt, main="Distibution des remboursements", xlab="Valeur de default", ylab="Nombre d'instances", fill=default)

# Histogramme d'effectifs de variables continues
qplot(age, data=emprunt, main="Distibution de age", xlab="Valeur de age", ylab="Nombre d'instances", binwidth=1)
qplot(employ, data=emprunt, main="Distibution de employ", xlab="Valeur de employ", ylab="Nombre d'instances", binwidth=1)
qplot(income, data=emprunt, main="Distibution de income", xlab="Valeur de income", ylab="Nombre d'instances", binwidth=10, fill=income)

#------------------#
# NUAGES DE POINTS #
#------------------#


# Nuages de points en couleurs des variables continue entre age et income
qplot(age, income, data=emprunt, main="Nuage de point de income et age", xlab="Valeur de age", ylab="Valeur de income", color=income) + geom_jitter(height = 0.3)
# Nuages de points en couleurs des variables continue entre age et employ
qplot(age, employ, data=emprunt, main="Nuage de point de employ et age", xlab="Valeur de age", ylab="Valeur de employ", color=employ) + geom_jitter(height = 0.3)
# Nuages de points en couleurs des variables continue entre employ et income
qplot(employ, income, data=emprunt, main="Nuage de point de employ et income", xlab="Valeur de employ", ylab="Valeur de income", color=income)

# Nuages de points de la variable continue income et la variable discrete ed
qplot(ed, income, data=emprunt, main="Nuage de point de income et ed", xlab="Valeur de ed", ylab="Valeur de income", color=income) + geom_jitter(height = 0.3)

# Nuages de points de la variable continue income et la variable discrete default
qplot(default, income, data=emprunt, main="Nuage de point de income et default", xlab="Valeur de default", ylab="Valeur de income", color=default) + geom_jitter(height = 0.3)

#---------------------------------------------------#
# CREATION DES ENSEMBLES D'APPRENTISSAGE ET DE TEST #
#---------------------------------------------------#

# Creation des ensembles d'apprentissage et de test
emprunt_EA <- emprunt[1:800,]
emprunt_ET <- emprunt[801:1200,]

# Suppression des variables d'identifications des ensembles d'apprentissage
emprunt_EA <- subset(emprunt_EA, select = -branch)
emprunt_EA <- subset(emprunt_EA, select = -ncust)
emprunt_EA <- subset(emprunt_EA, select = -customer)

#--------------------------------------------------------------------------#
# TEST DES ARBRES DE DECISION ET COMPARAISON AVEC LES RESULTATS PRECEDENTS #
#--------------------------------------------------------------------------#

#-------------------------------------------#
# APPRENTISSAGE DE L'ARBRE DE DECISION C5.0 #
#-------------------------------------------#

# Apprentissage arbre sur 'emprunt_EA'
tree1 <- C5.0(default~., emprunt_EA)

# Affichages graphiques 
plot(tree1, type="simple")

# Application de 'tree1' sur l'ensemble de test emprunt_ET
test_tree1 <- predict(tree1, emprunt_ET, type="class")

# Matrice de confusion des tests de 'tree1' sur l'ensemble de test emprunt_ET
mc_tree1 <- table(emprunt_ET$default, test_tree1)
print(mc_tree1)

# Generation des probabilites pour chaque exemple de test pour l'arbre 'tree1'
prob_tree1 <- predict(tree1, emprunt_ET, type="prob")

# Affichage des deux vecteurs de probabilites generes
print(prob_tree1)

# Affichage du vecteur de probabilites de prediction 'Oui'
prob_tree1[,2]

# Affichage du vecteur de probabilites de prediction 'Non'
prob_tree1[,1]

# Construction d'un data frame contenant classe reelle, prediction et probabilités des predictions
df_result1 <- data.frame(emprunt_ET$default, test_tree1, prob_tree1[,2], prob_tree1[,1])

# Renommage des colonnes afin d'en faciliter la lecture et les manipulations
colnames(df_result1) = list("Default","Prediction", "P(Oui)", "P(Non)")
View(df_result1)

# Quartiles et moyenne des probabilites des predictions 'Oui' pour l'arbre 'tree1'
summary(df_result1[df_result1$Prediction=="Oui", "P(Oui)"])

# Quartiles et moyenne des probabilites des predictions 'Non' pour l'arbre 'tree1'
summary(df_result1[df_result1$Prediction=="Non", "P(Non)"])

# Génération des probabilites de prediction sur l'ensemble de test
prob_tree1 <- predict(tree1, emprunt_ET, type="prob")

# Génération des donnees necessaires pour la courbe ROC
roc_pred1 <- prediction(prob_tree1[,2], emprunt_ET$default)

# Calcul des taux de vrais positifs (tpr) et taux de faux positifs (fpr)
roc_perf1 <- performance(roc_pred1,"tpr","fpr")

# Tracage de la  courbe ROC
plot(roc_perf1, add = TRUE, col = "blue")

# Calcul de l'AUC de l'arbre 'C5.0()'
auc_tree1 <- performance(roc_pred1, "auc")

# Affichage de la valeur de l'AUC
attr(auc_tree1, "y.values")

#--------------------------------------------#
# APPRENTISSAGE DE L'ARBRE DE DECISION RPART #
#--------------------------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rpart <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  dt <- rpart(default~., emprunt_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))
  
  # Tests du classifieur : classe predite
  dt_class <- predict(dt, emprunt_ET, type="class")
  
  # Matrice de confusion
  print(table(emprunt_ET$default, dt_class))
  
  # Tests du classifieur : probabilites pour chaque prediction
  dt_prob <- predict(dt, emprunt_ET, type="prob")
  
  # Courbes ROC
  dt_pred <- prediction(dt_prob[,2], emprunt_ET$default)
  dt_perf <- performance(dt_pred,"tpr","fpr")
  plot(dt_perf, main = "Arbres de décision rpart()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  dt_auc <- performance(dt_pred, "auc")
  cat("AUC = ", as.character(attr(dt_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

# Construction de l'arbre de decision 'tree2'
tree2 <- rpart(default~., emprunt_EA, parms = list(split = "gini"), control = rpart.control(minbucket = 10))

# Affichage de l'arbre par 'tree2' par plot.rpart() et text.rpart() 
plot(tree2)
text(tree2, pretty=0)

# Affichage textuel de l'arbre de decision
print(tree2)

# Application de 'tree2' sur l'ensemble de test emprunt_ET
test_tree2 <- predict(tree2, emprunt_ET, type="class")

# Matrice de confusion des tests de 'tree2' sur l'ensemble de test emprunt_ET
mc_tree2 <- table(emprunt_ET$default, test_tree2)
print(mc_tree2)

# Generation des probabilites pour chaque exemple de test pour l'arbre 'tree2'
prob_tree2 <- predict(tree2, emprunt_ET, type="prob")

# Affichage des deux vecteurs de probabilites generes
print(prob_tree2)

# Affichage du vecteur de probabilites de prediction 'Oui'
prob_tree2[,2]

# Affichage du vecteur de probabilites de prediction 'Non'
prob_tree2[,1]

# Construction d'un data frame contenant classe reelle, prediction et probabilités des predictions
df_result2 <- data.frame(emprunt_ET$default, test_tree2, prob_tree2[,2], prob_tree2[,1])

# Renommage des colonnes afin d'en faciliter la lecture et les manipulations
colnames(df_result2) = list("Default","Prediction", "P(Oui)", "P(Non)")
View(df_result2)

# Quartiles et moyenne des probabilites des predictions 'Oui' pour l'arbre 'tree2'
summary(df_result2[df_result2$Prediction=="Oui", "P(Oui)"])

# Quartiles et moyenne des probabilites des predictions 'Non' pour l'arbre 'tree2'
summary(df_result2[df_result2$Prediction=="Non", "P(Non)"])

# Génération des probabilites de prediction sur l'ensemble de test
prob_tree2 <- predict(tree2, emprunt_ET, type="prob")
print(prob_tree2)

# Génération des donnees necessaires pour la courbe ROC
roc_pred2 <- prediction(prob_tree2[,2], emprunt_ET$default)
print(roc_pred2)
roc_pred2 <- prediction(prob_tree2[,2], emprunt_ET$default, label.ordering = c("Non", "Oui"))
roc_pred2 <- prediction(prob_tree2[,2], emprunt_ET$default, labels = "Oui")


# Calcul des taux de vrais positifs (tpr) et taux de faux positifs (fpr)
roc_perf2 <- performance(roc_pred2,"tpr","fpr")
print(roc_perf2)

# Tracage de la  courbe ROC
plot(roc_perf2, col = "green")

#-------------------------------------------#
# APPRENTISSAGE DE L'ARBRE DE DECISION TREE #
#-------------------------------------------#

# Apprentissage arbre
tree3 <- tree(default~., data=emprunt_EA)

# Affichage graphique par plot.tree() et text.tree()
plot(tree3)
text(tree3, pretty=0)

# Génération des probabilites de prediction sur l'ensemble de test
prob_tree3 <- predict(tree3, emprunt_ET, type="vector")

# Génération des donnees necessaires pour la courbe ROC
roc_pred3 <- prediction(prob_tree3[,2], emprunt_ET$default)

# Calcul des taux de vrais positifs (tpr) et taux de faux positifs (fpr)
roc_perf3 <- performance(roc_pred3,"tpr","fpr")

# Ajout de la courbe ROC au precedent graphique
plot(roc_perf3, add = TRUE, col = "red")

# Calcul de l'AUC de l'arbre 'tree()'
auc_tree3 <- performance(roc_pred3, "auc")

# Affichage de la valeur de l'AUC
attr(auc_tree3, "y.values")

#----------------#
# RANDOM FORESTS #
#----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rf <- randomForest(default~., emprunt_EA, ntree = arg1, mtry = arg2)
  
  # Test du classifeur : classe predite
  rf_class <- predict(rf,emprunt_ET, type="response")
  
  # Matrice de confusion
  print(table(emprunt_ET$default, rf_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  rf_prob <- predict(rf, emprunt_ET, type="prob")
  
  # Courbe ROC
  rf_pred <- prediction(rf_prob[,2], emprunt_ET$default)
  rf_perf <- performance(rf_pred,"tpr","fpr")
  plot(rf_perf, main = "Random Forests randomForest()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  rf_auc <- performance(rf_pred, "auc")
  cat("AUC = ", as.character(attr(rf_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#---------------------#
# K-NEAREST NEIGHBORS #
#---------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanés du classifeur de type k-nearest neighbors
  knn <- kknn(default~., emprunt_EA, emprunt_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(emprunt_ET$default, knn$fitted.values))
  
  # Courbe ROC
  knn_pred <- prediction(knn$prob[,2], emprunt_ET$default)
  knn_perf <- performance(knn_pred,"tpr","fpr")
  plot(knn_perf, main = "Classifeurs K-plus-proches-voisins kknn()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knn_auc <- performance(knn_pred, "auc")
  cat("AUC = ", as.character(attr(knn_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(default~., emprunt_EA, probability=TRUE, kernel = arg1)
  
  # Test du classifeur : classe predite
  svm_class <- predict(svm, emprunt_ET, type="response")
  
  # Matrice de confusion
  print(table(emprunt_ET$default, svm_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, emprunt_ET, probability=TRUE)
  
  # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities")
  
  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,1], emprunt_ET$default)
  svm_perf <- performance(svm_pred,"tpr","fpr")
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  svm_auc <- performance(svm_pred, "auc")
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-------------#
# NAIVE BAYES #
#-------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nb <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur 
  nb <- naive_bayes(default~., emprunt_EA, laplace = arg1, usekernel = arg2)
  
  # Test du classifeur : classe predite
  nb_class <- predict(nb, emprunt_ET, type="class")
  
  # Matrice de confusion
  print(table(emprunt_ET$default, nb_class))
  
  # Test du classifeur : probabilites pour chaque prediction
  nb_prob <- predict(nb, emprunt_ET, type="prob")
  
  # Courbe ROC
  nb_pred <- prediction(nb_prob[,2], emprunt_ET$default)
  nb_perf <- performance(nb_pred,"tpr","fpr")
  plot(nb_perf, main = "Classifieurs bayésiens naïfs naiveBayes()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  nb_auc <- performance(nb_pred, "auc")
  cat("AUC = ", as.character(attr(nb_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
}

#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nn <- nnet(default~., emprunt_EA, size = arg1, decay = arg2, maxit=arg3)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nn_class <- predict(nn, emprunt_ET, type="class")
  
  # Matrice de confusion
  print(table(emprunt_ET$default, nn_class))
  
  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, emprunt_ET, type="raw")
  
  # Courbe ROC 
  nn_pred <- prediction(nn_prob[,1], emprunt_ET$default)
  nn_perf <- performance(nn_pred,"tpr","fpr")
  plot(nn_perf, main = "Réseaux de neurones nnet()", add = arg4, col = arg5)
  
  # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("AUC = ", as.character(attr(nn_auc, "y.values")))
  
  # Return ans affichage sur la console
  invisible()
}

#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Arbres de decision
test_rpart("gini", 10, FALSE, "red")
test_rpart("gini", 5, TRUE, "blue")
test_rpart("information", 10, TRUE, "green")
test_rpart("information", 5, TRUE, "orange")

# Forets d'arbres decisionnels aleatoires
test_rf(300, 3, FALSE, "red")
test_rf(300, 5, TRUE, "blue")
test_rf(500, 3, TRUE, "green")
test_rf(500, 5, TRUE, "orange")

# K plus proches voisins
test_knn(10, 1, FALSE, "red")
test_knn(10, 2, TRUE, "blue")
test_knn(20, 1, TRUE, "green")
test_knn(20, 2, TRUE, "orange")

# Support vector machines
test_svm("linear", FALSE, "red")
test_svm("polynomial", TRUE, "blue")
test_svm("radial", TRUE, "green")
test_svm("sigmoid", TRUE, "orange")

# Naive Bayes
test_nb(0, FALSE, FALSE, "red")
test_nb(20, FALSE, TRUE, "blue")
test_nb(0, TRUE, TRUE, "green")
test_nb(20, TRUE, TRUE, "orange")

# Réseaux de neurones nnet()
test_nnet(50, 0.01, 100, FALSE, "red")
test_nnet(50, 0.01, 300, TRUE, "tomato")
test_nnet(25, 0.01, 100, TRUE, "blue")
test_nnet(25, 0.01, 300, TRUE, "purple")
test_nnet(50, 0.001, 100, TRUE, "green")
test_nnet(50, 0.001, 300, TRUE, "turquoise")
test_nnet(25, 0.001, 100, TRUE, "grey")
test_nnet(25, 0.001, 300, TRUE, "black")

Apredire <- read.csv("Data Projet New.csv", header = TRUE, sep = ",", dec = ".",
                    stringsAsFactors = TRUE)

Apredire <- subset(Apredire, select = -branch)
Apredire <- subset(Apredire, select = -ncust)

knn <- kknn(default~., emprunt_EA, emprunt_ET)

#Probleme dans le knn_pred avec le type
knn_pred <- predict(knn, Apredire, type = "class")
knn_prob <- predict(knn, Apredire, type ="prob")
knn_auc <- performance(knn_pred, "auc")
cat("AUC = ", as.character(attr(knn_auc, "y.values")))

Resultat <- cbind(Apredire$customer, knn_pred, knn_prob)
Resultat <- data.frame(Resultat)
print(Resultat)

write.csv(x = Resultat, file = "Projet.csv")