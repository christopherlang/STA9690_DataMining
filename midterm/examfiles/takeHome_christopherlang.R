# Install the glmnet package if you haven't done so
# install.packages("glmnet")

# You many want to comment this out as this is environment specific
setwd("C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9690 - Advanced Data Mining/midterm/examfiles")

library(glmnet)
source("plotfuns.R")
load("bstar.Rdata")

plot.image(bstar)

p = length(bstar)
set.seed(0)
n = 1300
X = matrix(rnorm(n*p),nrow=n)
y = X%*%bstar + rnorm(n,sd=5)

K = 10
d = ceiling(n/K)
set.seed(0)
i.mix = sample(1:n)

# Tuning parameter values for lasso, and ridge regression
lam.las = c(seq(1e-3,0.1,length=100),seq(0.12,2.5,length=100))
lam.rid = lam.las*1000

nlam = length(lam.las)
# These two matrices store the prediction errors for each
# observation (along the rows), when we fit the model using
# each value of the tuning parameter (along the columns)
e.rid = matrix(0,n,nlam)
e.las = matrix(0,n,nlam)

for (k in 1:K) {
  cat("Fold",k,"\n")

  folds=(1+(k-1)*d):(k*d);
  i.tr=i.mix[-folds]
  i.val=i.mix[folds]

  X.tr = X[i.tr,]   # training predictors
  y.tr = y[i.tr]    # training responses
  X.val = X[i.val,] # validation predictors
  y.val = y[i.val]  # validation responses

  # TODO
  # Now use the function glmnet on the training data to get the
  # ridge regression solutions at all tuning parameter values in
  # lam.rid, and the lasso solutions at all tuning parameter
  # values in lam.las
  #
  # ||-ANSWER-|| below fills in the necessary parameters for fitting the
  # regressions

  # for the ridge regression solutions, use alpha=0
  a.rid = glmnet(X.tr, y.tr, alpha = 0, nlambda = nlam, lambda = lam.rid)

  # for the lasso solutions, use alpha=1
  a.las = glmnet(X.tr, y.tr, alpha = 1, nlambda = nlam, lambda = lam.las)

  # Here we're actually going to reverse the column order of the
  # a.rid$beta and a.las$beta matrices, because we want their columns
  # to correspond to increasing lambda values (glmnet's default makes
  # it so that these are actually in decreasing lambda order), i.e.,
  # in the same order as our lam.rid and lam.las vectors
  rid.beta = as.matrix(a.rid$beta[,nlam:1])
  las.beta = as.matrix(a.las$beta[,nlam:1])

  yhat.rid = X.val%*%rid.beta
  yhat.las = X.val%*%las.beta

  e.rid[i.val,] = (yhat.rid-y.val)^2
  e.las[i.val,] = (yhat.las-y.val)^2
}

# TODO
# Here you need to compute:
# -cv.rid, cv.las: vectors of length nlam, giving the cross-validation
#  errors for ridge regresssion and the lasso, across all values of the
#  tuning parameter
# -se.rid, se.las: vectors of length nlam, giving the standard errors
#  of the cross-validation estimates for ridge regression and the lasso,
#  across all values of the tunining parameter
#
# ||-ANSWER-||
cv.rid <- apply(e.rid, 2, mean)
cv.las <- apply(e.las, 2, mean)

se.rid <- apply(e.rid, 2, sd) / sqrt(n)
se.las <- apply(e.las, 2, sd) / sqrt(n)

# Usual rule for choosing lambda
i1.rid = which.min(cv.rid)
i1.las = which.min(cv.las)

# TODO
# One standard error rule for choosing lambda
# Here you need to compute:
# -i2.rid: the index of the lambda value in lam.rid chosen
#  by the one standard error rule
# -i2.las: the index of the lambda value in lam.las chosen
#  by the one standard error rule
#
#  ||-ANSWER-||
i2.rid <- max(which(cv.rid <= cv.rid[i1.rid] + se.rid[i1.rid]))
i2.las <- max(which(cv.las <= cv.las[i1.las] + se.las[i1.las]))

plot.cv(cv.rid,se.rid,lam.rid,i1.rid,i2.rid)
plot.cv(cv.las,se.las,lam.las,i1.las,i2.las)

# ||-ANSWER-|| for Part B ======================================================

# For finding ridge regression lambda selection, both rules
lam.rid[i1.rid]  # usual rule
lam.rid[i2.rid]  # standard error rule

# For finding lasso regression lambda selection, both rules
lam.las[i1.las]  # usual rule
lam.las[i2.las]  # standard error rule

# For finding minimum error for the ridge regression CV
cv.rid[i1.rid]
cv.rid[i2.rid]

# For finding minimum error for the lasso regression CV
cv.las[i1.las]
cv.las[i2.las]

# ||-ANSWER-|| for Part C ======================================================
a.rid <- glmnet(X, y, alpha = 0, nlambda = nlam, lambda = lam.rid)
a.las <- glmnet(X, y, alpha = 1, nlambda = nlam, lambda = lam.las)

rid.beta <- as.matrix(a.rid$beta[,nlam:1])
las.beta <- as.matrix(a.las$beta[,nlam:1])

plot.image(rid.beta[,i1.rid])
mtext('Image Ridge Usual Rule')

plot.image(rid.beta[,i2.rid])
mtext('Image Ridge Standard Error Rule')

plot.image(las.beta[,i1.las])
mtext('Image Lasso Usual Rule')

plot.image(las.beta[,i2.las])
mtext('Image Lasso Standard Error Rule')

# ||-ANSWER-|| for Part D ======================================================
# Ridge regression sum of squared error
sum((bstar - rid.beta[,i1.rid])^2)  # usual rule, ridge
sum((bstar - rid.beta[,i2.rid])^2)  # standard error rule, ridge

# Lasso regression sum of squared error
sum((bstar - las.beta[,i1.las])^2)  # usual rule, lasso
sum((bstar - las.beta[,i2.las])^2)  # standard error rule, lasso

# You many want to comment this out as this is environment specific
setwd("C:/Users/Christopher Lang/Dropbox/Education/Baruch College/Fall 2017/STA 9690 - Advanced Data Mining/")
