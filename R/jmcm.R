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
#' @param control a list (of correct class, resulting from jmcmControl())
#'  containing control parameters, see the *jmcmControl documentation for
#'  details.
#' @param start starting values for the parameters in the model.
#'
#' @examples
jmcm.mcbd <- function(formula, data = NULL, quad = c(3, 3, 3, 3),
                      cov.method = c('mcd', 'acd', 'hpc'),
                      control = jmcmControl(), start = NULL)
{
  mc <- mcout <- match.call()

  if (missing(cov.method))
    stop("cov.method must be specified")

  missCtrl <- missing(control)
  if (!missCtrl && !inherits(control, "jmcmControl"))
  {
    if(!is.list(control))
      stop("'control' is not a list; use jmcmControl()")

    warning("please use jmcmControl() instead", immediate. = TRUE)
    control <- do.call(jmcmControl, control)
  }

  mc[[1]] <- quote(cmmr::mldFormula)
  args <- eval(mc, parent.frame(1L))

  opt <- do.call(optimizeMcbd,
                 c(args, cov.method, list(control=control, start=start)))

  mkMcbdMod(opt=opt, args=args, quad=quad, cov.method=cov.method,mc=mcout)
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
optimizeMcbd <- function(m, Y, X, U, V, W, time, cov.method, control, start)
{
  debug <- 0
  if (debug) cat("optimizeMcbd():\n")
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
    stop("Incorrect start input")

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
      Xi <- X.new[(row.idx+1):(row.idx+m[1]*J), ]
      scm <- scm + (Yi - Xi %*% bta0) %*% t(Yi - Xi %*% bta0)
    }
    scm <- scm/length(m)

    chol.C      <- t(chol(scm))
    chol.D.sqrt <- diag(J * m[1])

    Tau   <- NULL
    Delta <- NULL
    for(t in 1:m[1])
    {
      row.idx = (t-1) * J
      index <- (row.idx+1):(row.idx+J)

      Dt.sqrt <- chol.C[index,index]
      Dt <- Dt.sqrt %*% t(Dt.sqrt)

      chol.D.sqrt[index, index] <- Dt.sqrt

      if (cov.method == 'mcd') {
        chol2.C <- t(chol(Dt))
        chol2.D.sqrt <- diag(diag(chol2.C))
        chol2.D.elem <- diag(chol2.D.sqrt)^2
        chol2.T <- chol2.D.sqrt %*% forwardsolve(chol2.C, diag(J))

        tmp  <- t(chol2.T)
        Tau   <- rbind(Tau, tmp[upper.tri(tmp)])
        Delta <- rbind(Delta, log(chol2.D.elem))
      } else if (cov.method == 'acd') {
        chol2.C <- t(chol(Dt))
        chol2.D.elem <- diag(chol2.C)
        chol2.L <- diag(chol2.D.elem^(-1)) %*% chol2.C

        tmp  <- t(chol2.L)
        Tau   <- rbind(Tau, tmp[upper.tri(tmp)])
        Delta <- rbind(Delta, log(chol2.D.elem^2))
      } else if (cov.method == 'hpc' ) {
        drd.H.elem <- sqrt(diag(Dt))
        Delta <- rbind(Delta, log(drd.H.elem^2))

        drd.R <- diag(drd.H.elem^(-1)) %*% Dt %*% diag(drd.H.elem^(-1))
        B <- t(chol(drd.R))
        PhiMat <- matrix(0, dim(B)[1], dim(B)[2])
        for(j in 2:dim(B)[1]) {
          for(k in 1:(j-1)) {
            tmp <- 1
            if (k != 1) {
              tmp <- prod(sin(PhiMat[j, 1:(k-1)]))
            } # if
            PhiMat[j,k] <- acos(B[j, k]/tmp)
          } # for k
        } # for j
        PhiMat
        tmp  <- t(PhiMat)
        Tau   <- rbind(Tau, tmp[upper.tri(tmp)])
      }
    }

    mm3 <- kronecker(diag(J*(J-1)/2), V[1:m[1],])
    lm.obj3 <- lm(c(Tau) ~ mm3 - 1)
    psi0 <- coef(lm.obj3)
    mm4 <- kronecker(diag(J), W[1:m[1],])
    lm.obj4 <- lm(c(Delta) ~ mm4 - 1)
    lmd0 <- coef(lm.obj4)

    chol.T <- chol.D.sqrt %*% forwardsolve(chol.C, diag(J * m[1]))
    Phi <- NULL
    for(t in 2:m[1]) {
      for(k in 1:(t-1)) {
        index1 <- ((t-1) * J + 1) : ((t-1) * J + J)
        index2 <- ((k-1) * J + 1) : ((k-1) * J + J)
        Ttk <- -chol.T[index1, index2]
        Phi <- rbind(Phi, Ttk)
      }
    }
    mm2 <- kronecker(diag(J), U.new[1:dim(Phi)[1],])
    cat("dim(Phi) = ", dim(Phi), "\n")
    lm.obj2 <- lm(c(Phi) ~ mm2 - 1)
    gma0 <- coef(lm.obj2)

    start <- c(bta0, gma0, psi0, lmd0)
    start[is.na(start)] <- 0
    #cat("bta = ", bta0, "\n")
    cat("gma = ", gma0, "\n")
    cat("psi = ", psi0, "\n")
    cat("lmd = ", lmd0, "\n")
    if(anyNA(start)) stop("failed to find an initial value with lm(). NA detected.")

  } else if (missStart && !isBalancedData) {
    bta0 <- NULL
    lmd0 <- NULL
    Gamma <- matrix(0, (J * dim(U)[2]), J)
    #gma0 <- rep(0, lgma)
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


      index1 <- ((j-1) * dim(U)[2] + 1) : ((j-1) * dim(U)[2] + dim(U)[2])
      index2 <- j:j
      Gamma[index1, index2] <- est$gamma
      # idx = J*J*(j-1) + J*(j-1)
      # gma0[(idx+1):(idx+dim(U)[2])] = est$gamma
    }
    gma0 <- c(Gamma)

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

#' @rdname modular
#' @export
mkMcbdMod <- function(opt, args, quad, cov.method, mc)
{
  if(missing(mc)) mc <- match.call()
  
  isMCD <- (cov.method == "mcd")
  isACD <- (cov.method == "acd")
  isHPC <- (cov.method == "hpc")
  
  dims <- c(
    nsub = length(args$m),
    max.nobs = max(args$m),
    p = quad[1],
    q = quad[2],
    s = quad[3],
    r = quad[4],
    MCD = isMCD,
    ACD = isACD,
    HPC = isHPC)
  new("mcbdMod", call = mc, opt = opt, args = args, quad = quad, devcomp = list(dims = dims))
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
jmcm.kron <- function(formula, data = NULL,
                      bcov.method = c('mcd', 'acd', 'hpc'),
                      wcov.method = c('mcd', 'acd', 'hpc'),
                      control = jmcmControl(), start = NULL)
{
  mc <- mcout <- match.call()

  if (missing(bcov.method))
    stop("bcov.method must be specified")
  if (missing(wcov.method))
    stop("wcov.method must be specified")

}

###----- Printing etc ----------------------------
methodTitle <- function(object, dims = object@devcomp$dims)
{
  MCD <- dims[["MCD"]]
  ACD <- dims[["ACD"]]
  HPC <- dims[["HPC"]]
  kind <- switch(MCD * 1L + ACD * 2L + HPC * 3L, "MCD", "ACD", "HPC")
  paste("Joint mean-covariance model based on MCBD -", kind)
}

cat.f <- function(...) cat(..., fill = TRUE)

.prt.methTit <- function(mtit, class) {
  cat(sprintf("%s ['%s']\n", mtit, class))
}

.prt.call <- function(call, long = TRUE) {
  if (!is.null(cc <- call$formula))
    cat.f("Formula:", deparse(cc))
  if (!is.null(cc <- call$quad))
    cat.f("   quad:", deparse(cc))
  if (!is.null(cc <- call$data))
    cat.f("   Data:", deparse(cc))
}

.prt.loglik <- function(n2ll, digits=4)
{
  t.4 <- round(n2ll, digits)
  cat.f("logLik:", t.4)
}

.prt.bic <- function(bic, digits=4)
{
  t.4 <- round(bic, digits)
  cat.f("   BIC:", t.4)
}

print.mcbdMod <- function(x, digits=4, ...)
{
  dims <- x@devcomp$dims
  .prt.methTit(methodTitle(x, dims = dims), class(x))
  .prt.call(x@call); cat("\n")
  .prt.loglik(x@opt$loglik)
  .prt.bic(x@opt$BIC); cat("\n")
  
  cat("Mean Parameters:\n")
  print.default(format(drop(x@opt$beta), digits = digits),
                print.gap = 2L, quote = FALSE)

  cat("\n-----T_itk-----\n")
  cat("Autoregressive Parameters:\n")
  print.default(format(drop(x@opt$gamma), digits = digits),
                print.gap = 2L, quote = FALSE)
  
  cat("\n-----D_it -----\n")
  
  if(dims["MCD"] == 1)
    cat("Autoregressive Parameters:\n")
  else if(dims["ACD"] == 1)
    cat("Moving Average Parameters:\n")
  else if(dims["HPC"] == 1)
    cat("Angle Parameters:\n")
  print.default(format(drop(x@opt$psi), digits = digits),
                print.gap = 2L, quote = FALSE)
  
  if(dims["MCD"] == 1 || dims["ACD"] == 1)
    cat("Innovation Variance Parameters:\n")
  else if(dims["HPC"] == 1)
    cat("Variance Parameters:\n")
  print.default(format(drop(x@opt$lambda), digits = digits),
                print.gap = 2L, quote = FALSE)
  
  invisible(x)
}

#' Print information for mcbdMod-class
#'
#' @param object a fitted joint mean covariance model of class "mcbdMod", i.e.,
#' typically the result of jmcm.mcbd().
#'
#' @exportMethod show
setMethod("show", "mcbdMod", function(object) print.mcbdMod(object))