namedList <- function(...) {
  L <- list(...)
  snm <- sapply(substitute(list(...)), deparse)[-1]
  if (is.null(nm <- names(L))) nm <- snm
  if (any(nonames <- nm == "")) nm[nonames] <- snm[nonames]
  setNames(L,nm)
}

#' @title Control of MCBD-based Covariance Matrices Model Fitting
jmcmControl <- function(trace = FALSE)
{
  structure(namedList(trace), class = 'jmcmControl')
}

#' #' @title Control of Kronecker Product based Covariance Structure Model Fitting
#' #'
#' #' @description Construct control structures for kronecker product based
#' #' covariance structure model fitting
#' #'
#' #' @param trace whether or not the value of the objective function and the
#' #' parameters should be print on every trace'th iteration.
#' #'
#' #' @export kcmmrControl
#' kcmmrControl <- function(trace = FALSE)
#' {
#'   structure(namedList(trace), class = 'kcmmrControl')
#' }