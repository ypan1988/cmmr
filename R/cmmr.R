#' @title Fit MCBD-based Covariance Matrices Models
#' 
#' @description Fit a modified Cholesky block decomposition (MCBD) based 
#'  covariance matrices Models to longitudinal data with multiple responses.
#' 
#' @param formula
#' @param data a data frame containing the variables named in formula.
#' @param cov.method covariance structure modelling method for matrix Dt,
#'  choose 'mcd' (Pourahmadi, 1999), 'acd' (Chen and Dunson, 2013) or 'hpc'
#'  (Zhang et al. 2015).
#' @param control a list (of correct class, resulting from mcmmrControl())
#'  containing control parameters, see the *mcmmrControl documentation for 
#'  details.
#' @param start starting values for the parameters in the model.
#' 
#' @examples
mcmmr <- function(formula, data = NULL, quad = c(3, 3, 3, 3),
                  cov.method = c('mcd', 'acd', 'hpc'), 
                  control = mcmmrControl(), start = NULL)
{
  mc <- mcout <- match.call()
  
  if (missing(cov.method))
    stop("cov.method must be specified")
  
  missCtrl <- missing(control)
  if (!missCtrl && !inherits(control, "mcmmrControl"))
  {
    if(!is.list(control))
      stop("'control' is not a list; use mcmmrControl()")
    
    warning("please use mcmmrControl() instead", immediate. = TRUE)
    control <- do.call(mcmmrControl, control)
  }
  
  mc[[1]] <- quote(cmmr::mldFormula)
  args <- eval(mc, parent.frame(1L))
  
  opt <- do.call(optimizeMcmmr,
                 c(args, cov.method, list(control=control, start=start)))
  
  opt
}

#' @title Modular Functions for Covariance Matrices Model Fits
#'
#' @description Modular function for covariance matrices model fits
#' 
#' @param formula
#'
#' @name modular
NULL
#> NULL

#' @rdname modular
#' @export
mldFormula <- function(formula, data = NULL, quad = c(3, 3, 3, 3), 
                       cov.method = c('mcd', 'acd', 'hpc'), 
                       control = jmcmControl(), start=NULL)
{
  debug <- 1
  
  if (debug) cat("mldFormula():\n")
  
  mf <- mc <- match.call()
  m <- match(c("formula", "data"), names(mf), 0L)
  mf <- mf[c(1, m)]
  
  f <- Formula::Formula(formula)
  mf[[1]] <- as.name("model.frame")
  mf$formula <- f
  mf <- eval(mf, parent.frame())
  
  Y    <- Formula::model.part(f, data = mf, lhs = 1)
  id   <- Formula::model.part(f, data = mf, lhs = 2)
  time <- Formula::model.part(f, data = mf, lhs = 3)

  X <- model.matrix(f, data = mf, rhs = 1)
  V <- model.matrix(f, data = mf, rhs = 2)
  W <- model.matrix(f, data = mf, rhs = 3)
  
  index <- order(id, time)
  
  Y    <- as.matrix(Y[index, ]) 
  id   <- id[index, ]
  time <- time[index, ]
  
  m <- table(id)
  attr(m, "dimnames") <- NULL
  
  U <- NULL
  for (i in 1:length(m))
  {
    if (i == 1) {
      ti <- time[1:m[1]]
    } else {
      first_index <- 1+sum(m[1:(i-1)])
      last_index <- sum(m[1:i])
      ti <- time[first_index:last_index]
    }
    
    if(m[i] != 1) {
      for (j in 2:m[i])
      {
        for (k in 1:(j - 1))
        {
          uijk = (ti[j]-ti[k])^(0:quad[2])
          U = rbind(U,uijk)
        }
      }
    }
  }
  
  # covariates from rhs of the formula
  Xtmp <- X[index, -1]
  Vtmp <- V[index, -1]
  Wtmp <- W[index, -1]
  
  # covariates based on polynomials of time
  X <- rep(1, length(time))
  V <- rep(1, length(time))
  W <- rep(1, length(time))
  for (i in 1:quad[1]) X = cbind(X, time^i)
  for (i in 1:quad[3]) V = cbind(V, time^i)
  for (i in 1:quad[4]) W = cbind(W, time^i)
  
  # combine two parts of the covariates
  X <- cbind(X, Xtmp)
  V <- cbind(V, Vtmp)
  W <- cbind(W, Wtmp)
  
  if (debug) cat("Is Y a matrix: ", is.matrix(Y), "\n")
  
  list(m = m, Y = Y, X = X, U = U, V = V, W = W, time = time)
}

#' @rdname modular
#' @export 
optimizeMcmmr <- function(m, Y, X, U, V, W, time, cov.method, control, start)
{
  debug <- 1
  if (debug) cat("optimizeMcmmr():\n")
  
  missStart <- is.null(start)
  
  J <- dim(Y)[2]
  
  lbta <- (J * dim(X)[2]) * 1
  lgma <- (J * dim(U)[2]) * J
  lpsi <- dim(V)[2]       * (J * (J - 1) / 2)
  llmd <- dim(W)[2]       * J

  if (!missStart && (lbta+lgma+lpsi+llmd) != length(start)) 
    Stop("Incorrect start input")

  if (missStart) {
    bta0 <- NULL
    lmd0 <- NULL
    for (j in 1:J) {
      cat("dim(Y)", dim(Y), "\n")
      lm.obj <- lm(Y[,j] ~ X - 1)
      bta0 <- c(bta0, coef(lm.obj))
      resid(lm.obj) -> res
      lmd0 <- c(lmd0, coef(lm(log(res^2) ~ W - 1)))
    }
    gma0 <- rep(0, lgma)
    psi0 <- rep(0, lpsi)

    if (debug) cat("bta0[", length(bta0),"]: ", bta0, "\n")
    if (debug) cat("gma0[", length(gma0),"]: ", gma0, "\n")
    if (debug) cat("psi0[", length(psi0),"]: ", psi0, "\n")
    if (debug) cat("lmd0[", length(lmd0),"]: ", lmd0, "\n")
    start <- c(bta0, gma0, psi0, lmd0)
    if(anyNA(start)) stop("failed to find an initial value with lm(). NA detected.")
  }
  est <- mcbd_estimation(m, Y, X, U, V, W, cov.method, start, control$trace)
  est
}


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



