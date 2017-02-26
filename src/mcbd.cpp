#include "mcbd.h"
#include "utils.h"

#include <RcppArmadillo.h>

namespace cmmr {
  mcbd::mcbd (const arma::uvec &m, const arma::mat &Y, const arma::mat &X,
              const arma::mat &U, const arma::mat &V, const arma::mat &W,
              const mcbd_mode &mcbd_mode_obj) :
    n_atts_(Y.n_cols), n_subs_(m.n_elem), m_(m), V_(V), W_(W),
    mcbd_mode_obj_(mcbd_mode_obj)
  {
    int debug = 0;

    if (debug) {
      std::cout << "n_atts_ = " << n_atts_ << std::endl
                << "n_subs_ = " << n_subs_ << std::endl;
    }

    poly_ = arma::zeros<arma::uvec>(4);
    poly_(0) = X.n_cols;
    poly_(1) = U.n_cols;
    poly_(2) = V.n_cols;
    poly_(3) = W.n_cols;

    arma::mat eye_J = arma::eye<arma::mat>(n_atts_, n_atts_);

    // initialize Y_
    Y_ = arma::vectorise(Y.t());

    // initialize X_
    for(arma::uword idx = 0; idx != X.n_rows; ++idx)
      X_ = arma::join_cols(X_, arma::kron(eye_J, X.row(idx)));

    // initialize U_
    for(arma::uword idx = 0; idx != U.n_rows; ++idx)
      U_ = arma::join_cols(U_, arma::kron(eye_J, U.row(idx)));

    free_param_ = 0;

    arma::uword ltht, lbta, lgma, lpsi, llmd;
    lbta = (n_atts_ * poly_(0)) * 1;
    lgma = (n_atts_ * poly_(1)) * n_atts_;
    lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    llmd = poly_(3)             * n_atts_;
    ltht = lbta + lgma + lpsi + llmd;

    arma::vec x = arma::zeros<arma::vec>(ltht);
    UpdateMcbd(x);

    if (debug) std::cout << "mcbd obj created..." << std::endl;
    if (debug) X_.rows(0, 9).print("X = ");
    if (debug) U_.rows(0, 9).print("U = ");
  }

  arma::uword mcbd::get_m(const arma::uword i) const {
    return m_(i);
  }

  arma::vec mcbd::get_Y(const arma::uword i) const {
    arma::mat Yi;
    if (i == 0) Yi = Y_.rows(0, n_atts_ * m_(0)-1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      Yi = Y_.rows(index, index + n_atts_ * m_(i) -1);
    }

    return Yi;
  }

  arma::mat mcbd::get_X(const arma::uword i) const {
    arma::mat Xi;
    if (i==0) Xi = X_.rows(0, n_atts_ * m_(0)-1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      Xi = X_.rows(index, index + n_atts_ * m_(i) -1);
    }

    return Xi;
  }

  arma::mat mcbd::get_U(const arma::uword i) const {
    arma::mat Ui;
    if (m_(i) != 1) {
      if (i == 0) {
        arma::uword first_index = 0;
        arma::uword last_index = n_atts_ * m_(0) * (m_(0) - 1) / 2 - 1;
        Ui = U_.rows(first_index, last_index);
      } else {
        arma::uword first_index = 0;
        for (arma::uword idx = 0; idx != i; ++idx) {
          first_index += m_(idx) * (m_(idx) - 1) / 2 - 1;
        }
        first_index *= n_atts_;
        arma::uword last_index = first_index + n_atts_ * m_(i) * (m_(i) - 1) / 2 - 1;
        Ui = U_.rows(first_index, last_index);
      }
    }

    return Ui;
  }

  arma::mat mcbd::get_V(const arma::uword i) const {
    arma::mat Vi;
    if (i==0) Vi = V_.rows(0, m_(0)-1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Vi = V_.rows(index, index + m_(i) -1);
    }

    return Vi;
  }

  arma::mat mcbd::get_W(const arma::uword i) const {
    arma::mat Wi;
    if (i==0) Wi = W_.rows(0, m_(0)-1);
    else {
      int index = arma::sum(m_.subvec(0, i - 1));
      Wi = W_.rows(index, index + m_(i) -1);
    }

    return Wi;
  }

  arma::vec mcbd::get_Resid ( const arma::uword i ) const {
    int debug = 0;

    if (debug) std::cout << "mcbd::get_Resid(): "
                         << "length(Resid_) = " << Resid_.n_rows
                         << std::endl;

    arma::mat Residi;
    if (i == 0) Residi = Resid_.rows(0, n_atts_ * m_(0) - 1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));

      if (debug) std::cout << "mcbd::get_Resid(): "
                           << "index = " << index
                           << std::endl;

      Residi = Resid_.rows(index, index + n_atts_ * m_(i) - 1);
    }

    return Residi;
  }

  arma::mat mcbd::get_U(const arma::uword i, const arma::uword t, const arma::uword k) const {
    arma::mat Uitk;
    if (m_(i) != 1) {
      arma::uword rindex = 0;
      for (arma::uword idx_i = 0; idx_i != i; ++idx_i) {
        rindex += m_(idx_i) * (m_(idx_i) - 1) / 2;
      }

      for (arma::uword idx_t = 1; idx_t < t; ++idx_t) {
        for (arma::uword idx_k = 0; idx_k < idx_t; ++idx_k) {
          ++rindex;
        }
      }

      rindex += k;
      rindex *= n_atts_;

      Uitk = U_.rows(rindex, rindex + n_atts_ - 1);
    }

    return Uitk;
  }

  arma::vec mcbd::get_V(const arma::uword i, const arma::uword t) const {
    arma::uword rindex = 0;
    if (i != 0) rindex = arma::sum(m_.rows(0, i-1));
    rindex += t;

    return V_.row(rindex).t();
  }

  arma::vec mcbd::get_W(const arma::uword i, const arma::uword t) const {
    arma::uword rindex = 0;
    if (i != 0) rindex = arma::sum(m_.rows(0, i-1));
    rindex += t;

    return W_.row(rindex).t();
  }

  arma::vec mcbd::get_Resid(const arma::uword i, const arma::uword t) const {
    arma::uword rindex = 0;
    if (i != 0) rindex = n_atts_ * arma::sum(m_.rows(0, i-1));
    rindex += n_atts_ * t;

    return Resid_.rows(rindex, rindex + n_atts_ - 1);
  }
  
  void mcbd::set_theta(const arma::vec &x) {
    int fp2 = free_param_;
    free_param_ = 0;
    UpdateMcbd(x);
    free_param_ = fp2;
  }

  void mcbd::set_beta(const arma::vec &x) {
    int fp2 = free_param_;
    free_param_ = 1;
    UpdateMcbd(x);
    free_param_ = fp2;
  }

  void mcbd::set_gamma(const arma::vec &x) {
    int fp2 = free_param_;
    free_param_ = 2;
    UpdateMcbd(x);
    free_param_ = fp2;
  }

  void mcbd::set_psi(const arma::vec &x) {
    int fp2 = free_param_;
    free_param_ = 3;
    UpdateMcbd(x);
    free_param_ = fp2;
  }

  void mcbd::set_lambda(const arma::vec &x) {
    int fp2 = free_param_;
    free_param_ = 4;
    UpdateMcbd(x);
    free_param_ = fp2;
  }

  arma::mat mcbd::get_T(const arma::uword i, const arma::uword t, const arma::uword k ) const {
    //int debug = 0;

    int mat_cnt = 0;
    if (i == 0) mat_cnt = 0;
    else {
      for(arma::uword idx = 0; idx != i; ++idx) {
        mat_cnt += m_(i) * (m_(i) - 1) / 2 ;
      }
    }

    for (arma::uword idx = 1; idx <= t; ++idx) {
      if ( idx != t ) {
        mat_cnt += idx;
      } else {
        mat_cnt += k;
      }
    }

    int rindex = mat_cnt * n_atts_;
    return UGma_.rows ( rindex, rindex + n_atts_ - 1 );
  }

  arma::mat mcbd::get_T(const arma::uword i) const {
    arma::mat Ti = arma::eye(n_atts_ * m_(i), n_atts_ * m_(i));
    for(arma::uword t = 0; t != m_(i); ++t) {
      for(arma::uword k = 0; k != t; ++k) {
        arma::mat Phi_itk = get_T(i, t, k);

        arma::uword rindex = t * n_atts_;
        arma::uword cindex = k * n_atts_;

        Ti(rindex, cindex, arma::size(Phi_itk)) = Phi_itk;
      }
    }

    return Ti;
  }

  arma::mat mcbd::get_T_bar(const arma::uword i, const arma::uword t) const {

    int debug = 0;

    if (debug) std::cout << "mcbd::get_T_bar(): "
                         << "size(VPsi_): " << arma::size(VPsi_)
                         << std::endl;

    arma::uword index = 0;
    if (i != 0) index = arma::sum(m_.rows(0, i-1));

    if (debug) std::cout << "mcbd::get_T_bar(): index = " << index << std::endl;
    arma::vec Ti_bar_elem = -arma::trans(VPsi_.row(index + t));

    arma::mat Ti_bar = arma::eye(n_atts_, n_atts_);
    Ti_bar = dragonwell::ltrimat(n_atts_, Ti_bar_elem);

    return Ti_bar;
  }

  arma::mat mcbd::get_T_bar(const arma::uword i) const {
    arma::mat Ti_bar = arma::eye(n_atts_ * m_(i), n_atts_ * m_(i));
    for(arma::uword t = 0; t != m_(i); ++t) {
      arma::mat Tit = get_T_bar(i, t);
      arma::uword rindex = t * n_atts_;
      Ti_bar(rindex, rindex, arma::size(Tit)) = Tit;
    }

    return Ti_bar;

  }

  arma::mat mcbd::get_D_bar(const arma::uword i, const arma::uword t) const {

    arma::uword index = 0;
    if (i != 0) index = arma::sum(m_.rows(0, i-1));

    arma::vec Di_bar_elem = -arma::trans(WLmd_.row(index + t));

    arma::mat Di_bar = arma::eye(n_atts_, n_atts_);
    Di_bar.diag() = Di_bar_elem;

    return Di_bar;
  }

  arma::mat mcbd::get_D_bar_inv(const arma::uword i) const {
    int debug = 0;

    arma::mat Di_bar_inv = arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

    for(arma::uword t = 0; t != m_(i); ++t) {
      arma::mat Dit_bar = get_D_bar(i, t);
      arma::mat Dit_bar_inv = arma::diagmat(arma::pow(Dit_bar.diag(), -1));

      arma::uword rindex = t * n_atts_;
      arma::uword cindex = t * n_atts_;

      Di_bar_inv(rindex, cindex, arma::size(Dit_bar_inv)) = Dit_bar_inv;
    }

    return Di_bar_inv;
  }
  
  arma::mat mcbd::get_D_inv(const arma::uword i, const arma::uword t) const {
    int debug = 0;

    if (debug) std::cout << "mcbd::get_D_inv(i,t): getting Tit_bar" << std::endl;
    arma::mat Tit_bar = get_T_bar(i, t);
    if (debug) std::cout << "mcbd::get_D_inv(i,t): getting Dit_bar" << std::endl;
    arma::mat Dit_bar = get_D_bar(i, t);
    arma::mat Dit_bar_inv = arma::diagmat(arma::pow(Dit_bar.diag(), -1));
    arma::mat Dit_inv = Tit_bar.t() * Dit_bar_inv * Tit_bar;

    return Dit_inv;
  }

  arma::mat mcbd::get_D_inv(const arma::uword i) const {
    int debug = 0;

    arma::mat Di_inv = arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

    if (debug) std::cout << "mcbd::get_D_inv(): before for loop" << std::endl;
    for(arma::uword t = 0; t != m_(i); ++t) {
      if (debug) std::cout << "mcbd::get_D_inv(): getting Dit_inv" << std::endl;
      arma::mat Dit_inv = get_D_inv(i, t);

      arma::uword rindex = t * n_atts_;
      arma::uword cindex = t * n_atts_;

      Di_inv(rindex, cindex, arma::size(Dit_inv)) = Dit_inv;
    }
    if (debug) std::cout << "mcbd::get_D_inv(): after for loop" << std::endl;

    return Di_inv;
  }

  arma::mat mcbd::get_Sigma_inv(const arma::uword i) const {
    int debug = 0;

    if (debug)
      std::cout << "mcbd::get_Sigma_inv(): "
                << "getting Ti..." << std::endl;

    arma::mat Ti = get_T(i);

    if (debug)
      std::cout << "mcbd::get_Sigma_inv(): "
                << "getting Di_inv..." << std::endl;
 
    arma::mat Di_inv = get_D_inv(i);

    arma::mat Sigmai_inv = Ti.t() * Di_inv * Ti;

    return Sigmai_inv;
  }

  void mcbd::UpdateMcbd(const arma::vec &x) {
    UpdateParam(x);
    UpdateModel();
  }

  void mcbd::UpdateParam ( const arma::vec &x ) {
    arma::uword lbta, lgma, lpsi, llmd;
    lbta = (n_atts_ * poly_(0)) * 1;
    lgma = (n_atts_ * poly_(1)) * n_atts_;
    lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    llmd = poly_(3)             * n_atts_;

    switch ( free_param_ ) {
    case 0:
      tht_ = x;
      bta_ = x.rows ( 0,                  lbta - 1 );
      gma_ = x.rows ( lbta,               lbta + lgma - 1 );
      psi_ = x.rows ( lbta + lgma,        lbta + lgma + lpsi - 1 );
      lmd_ = x.rows ( lbta + lgma + lpsi, lbta + lgma + lpsi + llmd - 1 );

      Gma_ = arma::reshape(arma::mat(gma_), U_.n_cols, n_atts_);
      Psi_ = arma::reshape(arma::mat(psi_), V_.n_cols, (n_atts_ * (n_atts_-1) / 2));
      Lmd_ = arma::reshape(arma::mat(lmd_), W_.n_cols, n_atts_);

      break;

    case 1:
      tht_.rows(0, lbta - 1) = x;
      bta_ = x;

      break;

    case 2:
      tht_.rows (lbta, lbta + lgma + lpsi + llmd - 1) = x;
      gma_ = x;
      psi_ = x;
      lmd_ = x;

      Gma_ = arma::reshape(arma::mat(gma_), U_.n_cols, n_atts_);
      Psi_ = arma::reshape(arma::mat(psi_), V_.n_cols, (n_atts_ * (n_atts_-1) / 2));
      Lmd_ = arma::reshape(arma::mat(lmd_), W_.n_cols, n_atts_);

      break;

    default: Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
    }
  }

  void mcbd::UpdateModel() {
    switch (free_param_) {
    case 0:
      Xbta_ = X_ * bta_;
      UGma_ = U_ * Gma_;
      VPsi_ = V_ * Psi_;
      WLmd_ = W_ * Lmd_;
      Resid_ = Y_ - Xbta_;
      mcd_UpdateTResid();
      mcd_UpdateTTResid();
      break;

    case 1:
      Xbta_ = X_ * bta_;
      Resid_ = Y_ - Xbta_;
      mcd_UpdateTResid();
      mcd_UpdateTTResid();
      break;

    case 2:
      UGma_ = U_ * Gma_;
      VPsi_ = V_ * Psi_;
      WLmd_ = W_ * Lmd_;
      mcd_UpdateTResid();
      mcd_UpdateTTResid();
      break;

    default: Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
    }
  }

  void mcbd::UpdateBeta() {
    arma::uword lbta = (n_atts_ * poly_(0)) * 1;
    arma::mat XSX = arma::zeros<arma::mat>(lbta, lbta);
    arma::vec XSY = arma::zeros<arma::vec>(lbta);

    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::mat Xi = get_X(i);
      arma::vec Yi = get_Y(i);
      arma::mat Sigmai_inv = get_Sigma_inv(i);

      XSX += Xi.t() * Sigmai_inv * Xi;
      XSY += Xi.t() * Sigmai_inv * Yi;
    }

    arma::vec beta = XSX.i() * XSY;
    set_beta(beta);
  }

  double mcbd::operator()(const arma::vec &x) {
    int debug = 0;

    UpdateMcbd(x);

    if (debug) std::cout << "mcbd::operator(): before for loop" << std::endl;
    double result = 0.0;
    for (arma::uword i = 0; i != n_subs_; ++i) {
      if (debug) std::cout << "iter " << i << ": getting ri..." << std::endl;
      arma::vec ri = get_Resid(i);
      if (debug) std::cout << "iter " << i << ": getting Sigmai_inv..." << std::endl;
      arma::mat Sigmai_inv = get_Sigma_inv(i);

      result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
    }
    if (debug) std::cout << "mcbd::operator(): after for loop" << std::endl;

    arma::vec one_M = arma::ones<arma::vec>(arma::sum(m_));
    arma::vec one_J = arma::ones<arma::vec>(n_atts_);
    double log_det_Sigma = arma::as_scalar(one_M.t() * WLmd_ * one_J);

    result += log_det_Sigma;

    if (debug) std::cout << "mcbd::operator(): loglik = " << result << std::endl;

    return result;
  }

  arma::vec mcbd::CalcDeriv(const arma::vec &x) {
    const double kEps = 1.0e-8;
    arma::uword n = x.n_rows;
    arma::vec xh = x;
    double fold = operator()(x);
    arma::vec grad = arma::zeros<arma::vec>(n);
    for (arma::uword idx = 0; idx != n; ++idx) {
      double tmp = x(idx);
      double h = kEps * std::abs(tmp);
      if (h == 0.0) h = kEps;
      xh(idx) = tmp + h;
      h = xh(idx) - tmp;
      double fh = operator()(xh);
      xh(idx) = tmp;
      grad(idx) = (fh - fold) / h;
    }

    return grad;
  } 
  
  void mcbd::Gradient(const arma::vec &x, arma::vec &grad) {
    int debug = 0;

    UpdateMcbd(x);

    arma::uword ltht, lbta, lgma, lpsi, llmd;
    lbta = (n_atts_ * poly_(0)) * 1;
    lgma = (n_atts_ * poly_(1)) * n_atts_;
    lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    llmd = poly_(3)             * n_atts_;
    ltht = lbta + lgma + lpsi + llmd;

    arma::vec grad1, grad2;
    switch (free_param_) {
    case 0:
      if (debug) std::cout << "mcbd::Gradient(): Calculating grad1..." << std::endl;
      Grad1(grad1);
      if (debug) std::cout << "mcbd::Gradient(): Calculating grad2..." << std::endl;
      Grad2(grad2);

      grad = arma::zeros<arma::vec>(ltht);
      grad.subvec(0, lbta - 1) = grad1;
      grad.subvec(lbta, ltht - 1) = grad2;

      break;

    case 1:
      Grad1(grad);
      break;

    case 2:
      Grad2(grad);
      break;

    default: Rcpp::Rcout << "wrong value for free_param_" << std::endl;
    }
  }

  void mcbd::Grad1(arma::vec &grad1) {
    arma::uword lbta = (n_atts_ * poly_(0)) * 1;
    grad1 = arma::zeros<arma::vec>(lbta);
    for (arma::uword i = 0; i != n_subs_; ++i ) {
      arma::vec Yi = get_Y(i);
      arma::mat Xi = get_X(i);
      arma::mat Sigmai_inv = get_Sigma_inv(i);

      grad1 += Xi.t() * Sigmai_inv * ( Yi - Xi * bta_ );
    }
    grad1 *= -2;
  }

  void mcbd::Grad2(arma::vec &grad2) {
    int debug = 0;

    arma::uword lgma, lpsi, llmd;
    lgma = (n_atts_ * poly_(1)) * n_atts_;
    lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    llmd = poly_(3)             * n_atts_;

    arma::vec grad_gma = arma::zeros<arma::vec>(lgma);
    arma::vec grad_psi = arma::zeros<arma::vec>(lpsi);
    arma::vec grad_lmd = arma::zeros<arma::vec>(llmd);

    const arma::vec one_J = arma::ones<arma::vec>(n_atts_);
    const arma::mat eye_Jr = arma::eye<arma::mat>(llmd, llmd);

    if (debug) std::cout << "mcbd::Grad2(): before for loop" << std::endl;
    for (arma::uword i = 0; i != n_subs_; ++i) {

      if (debug) std::cout << "mcbd::Grad2(): Calculate grad_gma" << std::endl;
      arma::mat Ci = get_C(i);
      arma::mat Di_inv = get_D_inv(i);
      arma::mat ei = get_Resid(i);

      grad_gma += Ci.t() * Di_inv * (ei - Ci * gma_);

      if (debug) std::cout << "mcbd::Grad2(): Calculate grad_psi" << std::endl;
      if (debug) std::cout << "mcbd::Grad2(): Getting G" << std::endl;
      arma::mat Gi = mcd_get_G(i);
      if (debug) std::cout << "mcbd::Grad2(): Getting Di_bar_inv" << std::endl;
      arma::mat Di_bar_inv = get_D_bar_inv(i);
      arma::mat epsi = mcd_get_TResid(i);

      if (debug) std::cout << "mcbd::Grad2(): size(Gi) = " << arma::size(Gi) << std::endl;
      if (debug) std::cout << "mcbd::Grad2(): size(Di_bar_inv) = " << arma::size(Di_bar_inv) << std::endl;
      if (debug) std::cout << "mcbd::Grad2(): size(epsi) = " << arma::size(epsi) << std::endl;
      if (debug) std::cout << "mcbd::Grad2(): size(psi_) = " << arma::size(psi_) << std::endl;

      grad_psi -= Gi.t() * Di_bar_inv * (epsi + Gi * psi_);

      if (debug) std::cout << "mcbd::Grad2(): Calculate grad_lmd" << std::endl;
      arma::mat one_T = arma::ones<arma::vec>(m_(i));
      arma::mat Wi = get_W(i);

      grad_lmd += -0.5 * arma::kron(one_J.t(), one_T.t() * Wi).t();
      arma::vec TTr = mcd_get_TTResid(i);
      for (arma::uword t = 0; t != m_(i); ++t) {
        arma::uword index = n_atts_ * t;
        grad_lmd += -0.5 * arma::kron(one_J.t(), eye_Jr) * mcd_CalcDbarDeriv(i,t)
          * TTr.subvec(index, index + n_atts_ - 1);
      }
    }
    if (debug) std::cout << "mcbd::Grad2(): after for loop" << std::endl;

    grad2 = -2 * dragonwell::join_vecs({grad_gma, grad_psi, grad_lmd});
  }

  arma::mat mcbd::get_C(const arma::uword i, const arma::uword t) const {
    int debug = 0;

    arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
    arma::mat Cit = arma::zeros<arma::mat>(n_atts_, lgma);
    if (debug) std::cout << "mcbd::get_C(i, t): size(Cit) = " << arma::size(Cit) << std::endl;

    if (t == 0) return Cit;
    else {
      // arma::mat Tit_bar = get_T_bar(i, t);
      // arma::mat eye_J   = arma::eye(n_atts_, n_atts_);
      for (arma::uword k = 0; k != t; ++k) {
	arma::vec eik = get_Resid(i, k);
        //arma::vec eik  = get_e(i, k);// WRONG!

        if (debug) std::cout << "mcbd::get_C(i, t): size(eik) = " << arma::size(eik) << std::endl;
        if (debug) std::cout << "mcbd::get_C(i, t): size(Uitk) = " << arma::size(get_U(i, t, k)) << std::endl;
        if (debug) get_U(i,t,k).print("Uitk = ");

        // arma::mat Aitk = Tit_bar * get_U(i, t, k);
        // Cit += arma::kron(eik.t(), Aitk);
        Cit += arma::kron(eik.t(), get_U(i, t, k));
      }
    }

    return Cit;
  }

  arma::mat mcbd::get_C(const arma::uword i) const {
    int debug = 0;
    if (debug) std::cout << "mcbd::get_C():" << std::endl;

    arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
    arma::mat Ci = arma::zeros<arma::mat>(n_atts_ * m_(i), lgma);

    for (arma::uword t = 1; t != m_(i); ++t) {
      arma::mat Cit = get_C(i, t);

      arma::uword rindex = n_atts_ * t;
      Ci.rows(rindex, rindex + n_atts_ - 1) = Cit;
      if (debug) std::cout << "mcbd::get_C(): for loop" << std::endl;
      if (debug) Ci.print("Ci = ");
    }

    return Ci;
  }

  arma::mat mcbd::get_e(const arma::uword i, const arma::uword t) const {
    arma::mat Ti = get_T(i);
    arma::mat Yi = get_Y(i);

    arma::mat ei = Ti * Yi;
    arma::uword rindex = n_atts_ * t;

    return ei.rows(rindex, rindex + n_atts_ - 1);
  }

  arma::mat mcbd::get_e(const arma::uword i) const {
    arma::mat Ti = get_T(i);
    arma::mat Yi = get_Y(i);

    return Ti * Yi;
  }

  void mcbd::mcd_UpdateTResid() {
    mcd_TResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::vec ri = get_Resid(i);
      arma::mat Ti = get_T(i);

      arma::vec Tr = Ti * ri;
      if (i == 0) mcd_TResid_.subvec(0, n_atts_ * m_(0) - 1) = Tr;
      else{
        int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
        mcd_TResid_.subvec(index, index + n_atts_ * m_(i) - 1) = Tr;
      }
    }
  }

  void mcbd::mcd_UpdateTTResid() {
    mcd_TTResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::vec ri = get_Resid(i);
      arma::mat Ti = get_T(i);
      arma::mat Ti_bar = get_T_bar(i);

      arma::vec TTr = Ti_bar * Ti * ri;
      if (i == 0) mcd_TTResid_.subvec(0, n_atts_ * m_(0) - 1) = TTr;
      else{
        int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
        mcd_TTResid_.subvec(index, index + n_atts_ * m_(i) - 1) = TTr;
      }
    }
  }

  arma::vec mcbd::mcd_get_TResid(const arma::uword i) const {
    if (i == 0) return mcd_TResid_.subvec(0, n_atts_ * m_(0) - 1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      return mcd_TResid_.subvec(index, index + n_atts_ * m_(i) - 1);
    }
  }

  arma::vec mcbd::mcd_get_TTResid(const arma::uword i) const {
    if (i == 0) return mcd_TTResid_.subvec(0, n_atts_ * m_(0) - 1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      return mcd_TTResid_.subvec(index, index + n_atts_ * m_(i) - 1);
    }
  }

  arma::mat mcbd::mcd_get_G(const arma::uword i) const {
    int debug = 0;

    if (debug) std::cout << "mcbd::mcd_get_G(): before for loop" << std::endl;
    arma::uword  lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    arma::mat result = arma::zeros<arma::mat>(n_atts_ * m_(i), lpsi);

    for (arma::uword t = 0, index = 0; t != m_(i); ++t) {
      arma::vec epsi = mcd_get_TResid(i);
      for (arma::uword j = 0, idx = 0; j != n_atts_; ++j) {
        if (debug) std::cout << "t = " << t << " j = " << j << ": " << std::endl;
        arma::vec gitj = arma::zeros<arma::vec>(lpsi);
        if (j == 0) { ++index; continue; }

        if (debug) std::cout << "Generate Vitj_t" << std::endl;
        arma::vec vit = get_V(i, t);
        arma::mat Vitj_t = arma::zeros<arma::mat>(j, lpsi);
        for (arma::uword k = 0; k <= (j-1); ++k, ++idx) {
          arma::vec av = arma::zeros<arma::vec>(lpsi);
          av.subvec(idx * poly_(2), idx * poly_(2) + poly_(2) - 1) = vit;
          Vitj_t.row(k) = av.t();
        }
        if (debug) std::cout << "Generate Vitj_t...done" << std::endl;

        gitj = Vitj_t.t() * epsi.subvec(0, j-1);

        result.row(index++) = gitj.t();
      }
    }

    if (debug) std::cout << "mcbd::mcd_get_G(): after for loop" << std::endl;

    return result;
  }
  
  arma::mat mcbd::mcd_CalcDbarDeriv(const arma::uword i, const arma::uword t) const {
    const arma::uword llmd = poly_(3) * n_atts_;
    arma::mat result = arma::zeros<arma::mat>(n_atts_ * llmd, n_atts_);

    arma::vec wit = get_W(i, t);
    arma::mat Dit_bar = get_D_bar(i, t);
    for(arma::uword j = 0; j != n_atts_; ++j) {
      result(j*poly_(3), j, arma::size(poly_(3), 1)) = -wit / Dit_bar(j,j);
    }

    return result;
  }
  
  // void CovMcbd::Grad1 ( arma::vec& grad1 ) {
  //   int debug = 1;

  //   int lbta = bta_.n_elem;
  //   grad1 = arma::zeros<arma::vec> ( lbta );
  //   for ( int i = 1; i <= n_subs_; ++i ) {
  //     arma::vec Yi = get_Y ( i );
  //     arma::mat Xi = get_X ( i );
  //     grad1 += Xi.t() * Sigma_inv_ * ( Yi - Xi * bta_ );
  //   }

  //   if ( debug ) grad1.print ( "grad1 = " );
  // }

  // void CovMcbd::Grad2 ( arma::vec& grad2 ) {
  //   int debug = 1;

  //   int llmd = lmd_.n_elem;
  //   grad2 = arma::zeros<arma::vec> ( llmd );

  //   arma::vec oneJ = arma::ones<arma::vec> ( n_atts_ );
  //   arma::vec oneT = arma::ones<arma::vec> ( n_dims_ );
  //   arma::vec oneTJ = arma::ones<arma::vec> ( n_dims_ * n_atts_ );
  //   grad2 = n_dims_ * arma::kron ( oneJ, W_.t() * oneT );

  //   if ( debug ) grad2.print("grad2 = ");
  // }
}
