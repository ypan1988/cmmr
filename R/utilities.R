#' @export
getJMCM <- function(object, name, sub.num) UseMethod("getJMCM")

#' @export
getJMCM.mcbdMod <- function(object, 
                            name = c("m", "Y", "X", "U", "V", "W"),
                            sub.num = 0)
{
  if(missing(name)) stop("'name' must not be missing")
  stopifnot(is(object,"mcbdMod"))
  
  opt     <- object@opt
  args    <- object@args
  devcomp <- object@devcomp
  
  if(sub.num < 0 || sub.num > length(args$m))
    stop("incorrect value for 'sub.num'")
  
  m = args$m
  Y = args$Y
  X = args$X
  U = args$U
  V = args$V
  W = args$W
  theta  = drop(opt$par)
  
  if (devcomp$dims['MCD']) CovMethod = "mcd"
  if (devcomp$dims['ACD']) CovMethod = "acd"
  if (devcomp$dims['HPC']) CovMethod = "hpc"
  
  obj <- .Call("mcbd__new", m, Y, X, U, V, W, CovMethod)
  
  if(sub.num == 0) {
    switch(name,
           "m" = args$m,
           "Y" = args$Y,
           "X" = args$X,
           "U" = args$U,
           "V" = args$V,
           "W" = args$W,
           "theta"  = drop(opt$par),
           "beta"   = drop(opt$beta),
           "gamma"  = drop(opt$gamma),
           "psi"    = drop(opt$psi),
           "lambda" = drop(opt$lambda),
           "loglik" = opt$loglik,
           "BIC"    = opt$BIC,
           "iter"   = opt$iter)
  } else {
    if (sub.num == 1) vindex = 1
    else vindex = sum(m[1:(sub.num-1)]) + 1
    switch(name,
           "m"     = .Call("mcbd__get_m",     obj, sub.num),
           "Y"     = .Call("mcbd__get_Y",     obj, sub.num),
           "X"     = .Call("mcbd__get_X",     obj, sub.num),
           "U"     = .Call("mcbd__get_U",     obj, sub.num),
           "V"     = .Call("mcbd__get_V",     obj, sub.num),
           "W"     = .Call("mcbd__get_W",     obj, sub.num),
           "T"     = .Call("mcbd__get_T",     obj, theta, sub.num),
           "D"     = .Call("mcbd__get_D",     obj, theta, sub.num),
           "mu"    = .Call("mcbd__get_mu",    obj, theta, sub.num),
           "Sigma" = .Call("mcbd__get_Sigma", obj, theta, sub.num))
  }
  
}