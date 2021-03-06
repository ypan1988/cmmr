% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/jmcm.R
\name{jmcm.mcbd}
\alias{jmcm.mcbd}
\title{Fit MCBD-based Covariance Matrices Models}
\usage{
jmcm.mcbd(formula, data = NULL, quad = c(3, 3, 3, 3),
  cov.method = c("mcd", "acd", "hpc"), control = jmcmControl(),
  start = NULL)
}
\arguments{
\item{formula}{a two-sided linear formula object describing the covariates
for both the mean and covariance matrix part of the model.}

\item{data}{a data frame containing the variables named in formula.}

\item{cov.method}{covariance structure modelling method for matrix Dt,
choose 'mcd' (Pourahmadi, 1999), 'acd' (Chen and Dunson, 2013) or 'hpc'
(Zhang et al. 2015).}

\item{control}{a list (of correct class, resulting from jmcmControl())
containing control parameters, see the *jmcmControl documentation for
details.}

\item{start}{starting values for the parameters in the model.}
}
\description{
Fit a modified Cholesky block decomposition (MCBD) based
 covariance matrices Models to longitudinal data with multiple responses.
}
\examples{
fit <- jmcm.mcbd(height + diameter | id | time ~ 1 | 1 | 1, data = poplar,
quad = c(10,1,1,1), cov.method = "mcd", control = jmcmControl(trace = TRUE))
}
