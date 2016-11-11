#' @title Fit Kronecker Product based Covariance Structure Models
#'
#' @description Fit a kronecker product based covariance structure model to
#' multivariate longitudinal data with multiple responses.
#'
#' @param formula
#' @param data a data frame containing the variables named in formula.
#' @param bcov.method between-subject covariance structure modelling method,
#' choose 'mcd' (Pourahmadi 1999), 'acd' (Chen and Dunson 2013) or 'hpc'
#' (Zhang et al. 2015).
#' @param wcov.method within-subject covariance structure modelling method,
#' choose 'mcd' (Pourahmadi 1999), 'acd' (Chen and Dunson 2013) or 'hpc'
#' (Zhang et al. 2015).
#' @param start starting values for the parameters in the model.
#' @export
kron_cmmr <- function(formula, data = NULL,
                  bcov.method = c('mcd', 'acd', 'hpc'),
                  wcov.method = c('mcd', 'acd', 'hpc'),
                  control = kcmmrControl(), start = NULL)
{
  mc <- mcout <- match.call()

  if (missing(bcov.method))
    stop("bcov.method must be specified")
  if (missing(wcov.method))
    stop("wcov.method must be specified")

}

mcbd_cmmr <- function(formula, data = NULL)
{

}
