% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SL.cforest.R
\name{SL.cforest}
\alias{SL.cforest}
\title{cforest (party)}
\usage{
SL.cforest(Y, X, newX, family, obsWeights, id, ntree = 1000,
  mtry = max(floor(ncol(X)/3), 1), mincriterion = 0, teststat = "quad",
  testtype = "Univ", replace = F, fraction = 0.632, ...)
}
\arguments{
\item{Y}{Outcome variable}

\item{X}{Covariate dataframe}

\item{newX}{Optional dataframe to predict the outcome}

\item{family}{"gaussian" for regression, "binomial" for binary
classification}

\item{obsWeights}{Optional observation-level weights (supported but not tested)}

\item{id}{Optional id to group observations from the same unit (not used
currently).}

\item{ntree}{Number of trees}

\item{mtry}{Number of randomly selected features per node}

\item{mincriterion}{See ?cforest_control}

\item{teststat}{See ?cforest_control}

\item{testtype}{See ?cforest_control}

\item{replace}{See ?cforest_control}

\item{fraction}{See ?cforest_control}

\item{...}{Remaining arguments (unused)}
}
\description{
These defaults emulate cforest_unbiased() but allow customization.
}
