
### linear
Public = read.csv("public_merge2.csv")
Public = Public[, -c(50:59)]

Public1 = Public[, -c(1, 3:4)]
Public1 = Public1[, -99]
Public1[, 1:112] = scale(Public1[, 1:112])

name = names(Public1)[-113]

Private = read.csv("private_merged2.csv")
Private = Private[, -c(50:59)]

Private1 = Private[, -c(1, 3:4)]
Private1 = Private1[, -104]
Private1[, 1:112] = scale(Private1[, 1:112])

### NN
private.nn = read.csv("private_cate_merged_nn171119.csv")

private.nn = private.nn[, -c(1, 3:4, 49:59)]
for (ii in 1:112){
  private.nn[, ii] = as.numeric(as.character(private.nn[, ii]))
}
private.nn = na.omit(private.nn)
private.nn[, 1:112] = scale( private.nn[, 1:112] )

public.nn = read.csv("public_cate_merged_nn171119.csv")

public.nn = public.nn[, -c(1, 3:4, 50:59)]
for (ii in 1:113){
  public.nn[, ii] = as.numeric(as.character(public.nn[, ii]))
}
public.nn = na.omit(public.nn)
public.nn[, 1:113] = scale( public.nn[, 1:113] )
public.nn = public.nn[, -99]





### random forest with conditional inference tree
library(party)

set.seed(500)

fit_rf = cforest(rankscore ~ ., data = Public1, control = cforest_unbiased(mtry = 5, ntree = 1000))

imp = varimp(fit_rf)
imp_sorted = sort(imp, decreasing=T)

par(las=1, cex.axis=0.4)
barplot(imp_sorted[30:1], main="Random Forest with Conditional Inference Tree, Public Institute", horiz=TRUE,
        names.arg=names(imp_sorted)[30:1], col="blue", xlab="Mean Decrease in Accuracy Importance Score")

set.seed(500)

fit_rf = cforest(rankscore ~ ., data = Private1, control = cforest_unbiased(mtry = 5, ntree = 1000))

imp = varimp(fit_rf)
imp_sorted = sort(imp, decreasing=T)

par(las=1, cex.axis=0.4)
barplot(imp_sorted[10:1], main="Random Forest with Conditional Inference Tree, Private Institute", horiz=TRUE,
        names.arg=names(imp_sorted)[30:1], col="blue", xlab="Mean Decrease in Accuracy Importance Score")


### gradient boosting
library(gbm)

fit_gbm = gbm(rankscore ~., data = Public1, distribution = "gaussian", n.tree=10000, 
              shrinkage = 0.01, cv.folds = 5, interaction.depth = 4)
gbm_perf = gbm.perf(fit_gbm, method = "cv")

par(las=1, cex.axis=0.4)
summary(fit_gbm, n.tree=10000, cBars=30, main="Gradient Boosting, Public Institute") 

fit_gbm = gbm(rankscore ~., data = Private1, distribution = "gaussian", n.tree=10000, 
              shrinkage = 0.01, cv.folds = 5, interaction.depth = 4)
gbm_perf = gbm.perf(fit_gbm, method = "cv")

par(las=1, cex.axis=0.4)
summary(fit_gbm, n.tree=10000, cBars=30, main="Gradient Boosting, Private Institute")
