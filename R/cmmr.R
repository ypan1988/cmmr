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
  debug <- 0

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
  debug <- 0
  if (debug) cat("optimizeMcmmr():\n")
  if (debug) cat("dim(Y)", dim(Y), "\n")
  if (debug) cat("dim(X)", dim(X), "\n")
  if (debug) cat("dim(U)", dim(U), "\n")
  if (debug) cat("dim(V)", dim(V), "\n")
  if (debug) cat("dim(W)", dim(W), "\n")
  
  missStart <- is.null(start)

  J <- dim(Y)[2]

  lbta <- (J * dim(X)[2]) * 1
  lgma <- (J * dim(U)[2]) * J
  lpsi <- dim(V)[2]       * (J * (J - 1) / 2)
  llmd <- dim(W)[2]       * J

  if (!missStart && (lbta+lgma+lpsi+llmd) != length(start))
    Stop("Incorrect start input")

  isBalancedData <- all(m == m[1])
  if (missStart && isBalancedData) {
    Y.new <- c(t(Y))
    X.new <- NULL
    U.new <- NULL
    
    for (idx in 1:dim(X)[1])
      X.new <- rbind(X.new, kronecker(diag(J), matrix(X[idx, ], 1)))
    for (idx in 1:dim(U)[1])
      U.new <- rbind(U.new, kronecker(diag(J), matrix(U[idx, ], 1)))
    
    lm.obj <- lm(Y.new ~ X.new - 1)
    bta0 <- coef(lm.obj)

    scm <- 0
    for(i in 1:length(m))
    {
      row.idx = (i-1) * (m[1] * J)
      Yi <- Y.new[(row.idx+1):(row.idx+m[1]*J)]
      Xi <- X.new[(row.idx+1):(row.idx+m[1]*J),]
      scm <- scm + (Yi - Xi %*% bta0) %*% t(Yi - Xi %*% bta0)
    }
    scm <- scm/length(m)
    
    chol.C      <- t(chol(scm))
    chol.D      <- diag(J * m[1])
    chol.D.sqrt <- diag(J * m[1])
    chol.T.bar  <- diag(J * m[1])
    chol.D.bar  <- diag(J * m[1]) 

    Tau   <- NULL
    Delta <- NULL
    for(t in 1:m[1])
    {
      row.idx = (t-1) * J
      index <- (row.idx+1):(row.idx+J)
      Dt.sqrt <- chol.C[index,index]
      Dt <- Dt.sqrt %*% t(Dt.sqrt)
      chol.D.sqrt[index, index] <- Dt.sqrt
      chol.D[index, index]      <- Dt
      
      chol2.C <- t(chol(Dt))
      chol2.D.sqrt <- diag(diag(chol2.C))
      chol2.D <- chol2.D.sqrt %*% chol2.D.sqrt
      chol2.T <- chol2.D.sqrt %*% forwardsolve(chol2.C, diag(J))

      Ttmp  <- t(chol2.T)
      Tau   <- rbind(Tau, Ttmp[upper.tri(Ttmp)])
      Delta <- rbind(Delta, log(diag(chol2.D)))
    }

    lm.obj3 <- lm(c(Tau) ~ (kronecker(diag(J*(J-1)/2), V[1:m[1],])) - 1)
    psi0 <- coef(lm.obj3)
    lm.obj4 <- lm(c(Delta) ~ (kronecker(diag(J), W[1:m[1],])) - 1)
    lmd0 <- coef(lm.obj4)
        
    chol.T <- chol.D.sqrt %*% forwardsolve(chol.C, diag(J * m[1]))
    Phi <- NULL
    for(t in 2:m[1]) {
      for(k in 1:(t-1)) {
        index1 <- ((t-1) * J + 1) : ((t-1) * J + J)
        index2 <- ((k-1) * J + 1) : ((k-1) * J + J)
        Ttk <- chol.T[index1, index2]
        Phi <- rbind(Phi, Ttk) 
      }
    }
    lm.obj2 <- lm(c(Phi) ~ (kronecker(diag(J), U.new[1:dim(Phi)[1],])) - 1)
    gma0 <- coef(lm.obj2)
    
    start <- c(bta0, gma0, psi0, lmd0)
    if(anyNA(start)) stop("failed to find an initial value with lm(). NA detected.")
    
  } else if (missStart && !isBalancedData) {
    bta0 <- NULL
    lmd0 <- NULL
    gma0 <- rep(0, lgma)
    for (j in 1:J) {
      lm.obj <- lm(Y[,j] ~ X - 1)
      mcd.bta0 <- coef(lm.obj)
      
      resid(lm.obj) -> res
      mcd.lmd0 <- coef(lm(log(res^2) ~ W - 1))
      mcd.gma0 <- rep(0, dim(U)[2])
      
      mcd.start <- c(mcd.bta0, mcd.lmd0, mcd.gma0)
      
      est <- jmcm::mcd_estimation(m, Y[,j], X, W, U, mcd.start, Y[,j])
      
      if(debug) cat("est:", str(est))
      bta0 <- c(bta0, est$beta)
      lmd0 <- c(lmd0, est$lambda)

      idx = J*J*(j-1) + J*(j-1)
      gma0[(idx+1):(idx+dim(U)[2])] = est$gamma
    }

    psi0 <- NULL
    for (j in 2:J) {
      for (k in 1:(j-1)) {
        tmp <- rep(0, dim(V)[2])
        if (cov.method == 'hpc') tmp[1] <- pi/2
        psi0 <- c(psi0, tmp)
      }
    }
#        psi0 <- rep(0, lpsi)

    if (debug) cat("bta0: ", bta0, bta0, "\n")
    if (debug) cat("gma0: ", gma0, gma0, "\n")
    if (debug) cat("psi0: ", psi0, psi0, "\n")
    if (debug) cat("lmd0: ", lmd0, lmd0, "\n")
    
    if (debug) cat("bta0[", length(bta0),"]: ", bta0, "\n")
    if (debug) cat("gma0[", length(gma0),"]: ", gma0, "\n")
    if (debug) cat("psi0[", length(psi0),"]: ", psi0, "\n")
    if (debug) cat("lmd0[", length(lmd0),"]: ", lmd0, "\n")
    start <- c(bta0, gma0, psi0, lmd0)
    if(anyNA(start)) stop("failed to find an initial value with lm(). NA detected.")
  }

  #est <- mcbd_test(m, Y, X, U, V, W, cov.method, start, control$trace)
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
