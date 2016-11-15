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
kcmmr <- function(formula, data = NULL,
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

ccmmr <- function(formula, data = NULL, cov.method = c('mcd', 'acd', 'hpc'), 
                  control = ccmmrControl(), start = NULL)
{
  mc <- mcout <- match.call()
  
  if (missing(cov.method))
    stop("cov.method must be specified")
  
  missCtrl <- missing(control)
  if (!missCtrl && !inherits(control, "ccmmrControl"))
  {
    if(!is.list(control))
      stop("'control' is not a list; use ccmmrControl()")
    
    warning("please use ccmmrControl() instead", immediate. = TRUE)
    control <- do.call(ccmmrControl, control)
  }
  
  mc[[1]] <- quote(cmmr::ldFormula)
  args <- eval(mc, parent.frame(1L))
}
