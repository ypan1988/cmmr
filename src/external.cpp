#include <cmath>

#include <iostream>
using std::cout;

#include <RcppArmadillo.h>

#include "bfgs.h"
#include "linesearch.h"
#include "mcbd.h"
#include "roptim.h"

//'@export
// [[Rcpp::export]]
Rcpp::List mcbd_estimation(arma::uvec m, arma::mat Y, arma::mat X, arma::mat U,
                           arma::mat V, arma::mat W, std::string cov_method,
                           arma::vec start, arma::vec mean, bool trace = false,
                           bool profile = true, bool covonly = false,
                           std::string optim_method = "default") {
  int debug = 0;

  arma::uword n_subs = m.n_elem;
  arma::uword n_atts = Y.n_cols;

  arma::uvec poly = arma::zeros<arma::uvec>(4);
  poly(0) = X.n_cols;
  poly(1) = U.n_cols;
  poly(2) = V.n_cols;
  poly(3) = W.n_cols;

  arma::uword ltht, lbta, lgma, lpsi, llmd, ltht2;
  lbta = (n_atts * poly(0)) * 1;
  lgma = (n_atts * poly(1)) * n_atts;
  lpsi = poly(2) * (n_atts * (n_atts - 1) / 2);
  llmd = poly(3) * n_atts;
  ltht = lbta + lgma + lpsi + llmd;
  ltht2 = lgma + lpsi + llmd;

  mcbd_mode cov_obj(0);
  if (cov_method == "mcd") cov_obj.setid(1);
  if (cov_method == "acd") cov_obj.setid(2);
  if (cov_method == "hpc") cov_obj.setid(3);

  cmmr::mcbd mcbd_obj(m, Y, X, U, V, W, cov_obj);
  if (covonly) {
    mcbd_obj.set_mean(mean);
  }

  pan::BFGS<cmmr::mcbd> bfgs;
  bfgs.set_trace(trace);
  roptim::Roptim<cmmr::mcbd> optim("BFGS");
  optim.control.trace = trace;

  pan::LineSearch<cmmr::mcbd> linesearch;

  arma::vec x = start;

  double f_min = 0.0;
  int n_iters = 0;

  if (profile) {
    const int kIterMax = 200;  // Maximum number of iterations
    const double kEpsilon =
        std::numeric_limits<double>::epsilon();  // Machine precision
    const double kTolX = 4 * kEpsilon;  // Convergence criterion on x values
    const double kScaStepMax =
        100;  // Scaled maximum step length allowed in line searches
    const double grad_tol = 1e-6;
    const int n_pars = x.n_rows;  // number of parameters

    double f = mcbd_obj(x);
    arma::vec grad;
    mcbd_obj.Gradient(x, grad);
    arma::mat hess_inv = arma::eye<arma::mat>(
        n_pars, n_pars);  // Initialized the inverse Hessian to a unit matrix
    arma::vec p = -hess_inv * grad;  // Initialize Newton Step

    // Calculate the maximum step length
    double sum = sqrt(arma::dot(x, x));
    const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));

    // Main loop over the iterations
    for (int iter = 0; iter != kIterMax; ++iter) {
      n_iters = iter;

      arma::vec x2 = x;  // Save the old point

      // original linesearch:
      // linesearch.GetStep(mcbd_obj, x, p, kStepMax);

      arma::vec x_mean = x.subvec(0, lbta - 1);
      arma::vec x_cov = x.subvec(lbta, ltht - 1);
      arma::vec p_mean = p.subvec(0, lbta - 1);
      arma::vec p_cov = p.subvec(lbta, ltht - 1);

      mcbd_obj.set_free_param(1);
      linesearch.GetStep(mcbd_obj, x_mean, p_mean, kStepMax);
      mcbd_obj.set_free_param(0);

      mcbd_obj.set_free_param(2);
      linesearch.GetStep(mcbd_obj, x_cov, p_cov, kStepMax);
      mcbd_obj.set_free_param(0);

      x.subvec(0, lbta - 1) = x_mean;
      x.subvec(lbta, ltht - 1) = x_cov;

      f = mcbd_obj(x);  // Update function value
      p = x - x2;       // Update line direction
      x2 = x;
      f_min = f;

      if (trace) {
        Rcpp::Rcout << std::setw(5) << iter << ": " << std::setw(10)
                    << mcbd_obj(x) << ": ";
        x.t().print();
      }

      if (debug) Rcpp::Rcout << "Checking convergence..." << std::endl;
      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }
      if (test < kTolX) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on Delta x: converged."
                      << std::endl;
        break;
      }

      arma::vec grad2 = grad;      // Save the old gradient
      mcbd_obj.Gradient(x, grad);  // Get the new gradient

      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }
      if (test < grad_tol) {
        if (debug)
          Rcpp::Rcout << "Test for convergence on zero gradient: converged."
                      << std::endl;
        break;
      }

      if (debug) Rcpp::Rcout << "Update beta..." << std::endl;
      if (!covonly) mcbd_obj.UpdateBeta();

      if (debug) Rcpp::Rcout << "Update gamma, psi, lambda..." << std::endl;
      arma::vec tht2 = x.rows(lbta, lbta + ltht2 - 1);

      if (trace) {
        Rcpp::Rcout
            << "--------------------------------------------------"
            << "\n Updating the Three Parameters in Covariance Matrix ..."
            << std::endl;
      }

      mcbd_obj.set_free_param(2);
      if (optim_method == "default")
        bfgs.Optimize(mcbd_obj, tht2);
      else
        optim.minimize(mcbd_obj, tht2);
      mcbd_obj.set_free_param(0);

      if (trace) {
        Rcpp::Rcout << "--------------------------------------------------"
                    << std::endl;
      }

      mcbd_obj.UpdateTheta2(tht2);

      if (debug) Rcpp::Rcout << "Update theta..." << std::endl;
      arma::vec xnew = mcbd_obj.get_theta();

      p = xnew - x;
    }
  } else {
    if (optim_method == "default") {
      bfgs.Optimize(mcbd_obj, x);
      f_min = bfgs.f_min();
      n_iters = bfgs.n_iters();
    } else {
      optim.minimize(mcbd_obj, x);
      f_min = optim.value();
    }
  }

  arma::vec beta = x.rows(0, lbta - 1);
  arma::vec gamma = x.rows(lbta, lbta + lgma - 1);
  arma::vec psi = x.rows(lbta + lgma, lbta + lgma + lpsi - 1);
  arma::vec lambda = x.rows(lbta + lgma + lpsi, lbta + lgma + lpsi + llmd - 1);

  arma::uvec npars = {ltht, lbta, lgma, lpsi, llmd};

  double loglik = f_min / (-2.0) +
                  n_atts * arma::sum(m) / (-2.0) * log(2 * arma::datum::pi);
  double bic =
      -2 * loglik / n_subs + ltht * log(static_cast<double>(n_subs)) / n_subs;

  return Rcpp::List::create(
      Rcpp::Named("par") = x, Rcpp::Named("beta") = beta,
      Rcpp::Named("gamma") = gamma, Rcpp::Named("psi") = psi,
      Rcpp::Named("lambda") = lambda, Rcpp::Named("npars") = npars,
      Rcpp::Named("loglik") = loglik, Rcpp::Named("BIC") = bic,
      Rcpp::Named("iter") = n_iters);
}

RcppExport SEXP mcbd__new(SEXP m_, SEXP Y_, SEXP X_, SEXP U_, SEXP V_, SEXP W_,
                          SEXP cov_method_) {
  int debug = 0;

  if (debug) Rcpp::Rcout << "initialization..." << std::endl;
  arma::uvec m = Rcpp::as<arma::uvec>(m_);
  arma::mat Y = Rcpp::as<arma::mat>(Y_);
  arma::mat X = Rcpp::as<arma::mat>(X_);
  arma::mat U = Rcpp::as<arma::mat>(U_);
  arma::mat V = Rcpp::as<arma::mat>(V_);
  arma::mat W = Rcpp::as<arma::mat>(W_);
  std::string cov_method = Rcpp::as<std::string>(cov_method_);
  if (debug) Rcpp::Rcout << "initialization...finished" << std::endl;

  if (debug) Rcpp::Rcout << "method setting..." << std::endl;
  mcbd_mode cov_obj(0);
  if (cov_method == "mcd") cov_obj.setid(1);
  if (cov_method == "acd") cov_obj.setid(2);
  if (cov_method == "hpc") cov_obj.setid(3);
  if (debug) Rcpp::Rcout << "method setting...finished" << std::endl;

  Rcpp::XPtr<cmmr::mcbd> ptr(new cmmr::mcbd(m, Y, X, U, V, W, cov_obj), true);

  return ptr;
}

RcppExport SEXP mcbd__get_m(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_m(i));
}

RcppExport SEXP mcbd__get_Y(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_Y(i));
}

RcppExport SEXP mcbd__get_X(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_X(i));
}

RcppExport SEXP mcbd__get_U(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_U(i));
}

RcppExport SEXP mcbd__get_V(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_V(i));
}

RcppExport SEXP mcbd__get_W(SEXP xp, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);
  int i = Rcpp::as<int>(i_) - 1;

  return Rcpp::wrap(ptr->get_W(i));
}

RcppExport SEXP mcbd__get_D(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateMcbd(x);

  return Rcpp::wrap(ptr->get_D(i));
}

RcppExport SEXP mcbd__get_T(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateMcbd(x);

  return Rcpp::wrap(ptr->get_T(i));
}

RcppExport SEXP mcbd__get_mu(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  ptr->UpdateMcbd(x);

  return Rcpp::wrap(ptr->get_mu(i));
}

RcppExport SEXP mcbd__get_Sigma(SEXP xp, SEXP x_, SEXP i_) {
  Rcpp::XPtr<cmmr::mcbd> ptr(xp);

  arma::vec x = Rcpp::as<arma::vec>(x_);
  int i = Rcpp::as<int>(i_) - 1;

  arma::mat Sigmai;

  ptr->UpdateMcbd(x);

  return Rcpp::wrap(ptr->get_Sigma(i));
}