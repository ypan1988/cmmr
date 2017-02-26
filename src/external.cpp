#include <RcppArmadillo.h>
#include "mcbd.h"

//'@export
// [[Rcpp::export]]
void
mcbd_estimation(arma::uvec m, arma::mat Y, arma::mat X, arma::mat U, arma::mat V, arma::mat W,
		std::string cov_method, arma::vec start, bool trace = false)
{
  int debug = 1;

  mcbd_mode cov_obj(0);
  if(cov_method == "mcd") cov_obj.setid(1);

  cmmr::mcbd mcbd_obj(m, Y, X, U, V, W, cov_obj);

  double result = mcbd_obj(start);
  if (debug) std::cout << "result: " << result << std::endl;

  mcbd_obj.CalcDeriv(start).t().print("grad = ");
  
  arma::vec grad;
  mcbd_obj.Gradient(start, grad);
  
  grad.t().print("grad = ");
  
}

