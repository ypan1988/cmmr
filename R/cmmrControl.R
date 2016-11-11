#' @title Control of Kronecker Product based Covariance Structure Model Fitting
#'
#' @description Construct control structures for kronecker product based
#' covariance structure model fitting
#'
#' @param trace whether or not the value of the objective function and the
#' parameters should be print on every trace'th iteration.
#'
#' @export kcmmrControl
kcmmrControl <- function(trace = FALSE)
{
  structure(jmcm::namedList(trace), class = 'kcmmrControl')
}
