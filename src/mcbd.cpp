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
    int debug = 1;

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

    if ( debug ) std::cout << "mcbd obj created..." << std::endl;
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
    arma::mat Residi;
    if (i == 0) Residi = Resid_.rows(0, n_atts_ * m_(0) - 1);
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      Residi = Resid_.rows(index, index + n_atts_ * m_(i) - 1);
    }

    return Residi;
  }

  arma::mat mcbd::get_U(const arma::uword i, const arma::uword t, const arma::uword k) const {
    arma::mat Uitk;
    if (m_(i) != 1) {
      arma::uword index = 0;
      for (arma::uword idx_i = 0; idx_i <= i; ++idx_i) {
        for (arma::uword idx_t = 0; idx_t <= t; ++idx_t) {
          for (arma::uword idx_k = 0; idx_k <=; ++idx_k) {
          }
        }
      }


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

    arma::uword index = 0;
    for (arma::uword idx = 0; idx != i; ++idx) {
      index += m_(idx) * (m_(idx) - 1) / 2;
    }
    arma::vec Ti_bar_elem = -arma::trans(VPsi_.row(index + t));

    arma::mat Ti_bar = arma::eye(n_atts_, n_atts_);
    Ti_bar = dragonwell::ltrimat(n_atts_, Ti_bar_elem);

    return Ti_bar;
  }

  arma::mat mcbd::get_D_bar(const arma::uword i, const arma::uword t) const {

    arma::uword index = 0;
    for (arma::uword idx = 0; idx != i; ++idx) {
      index += m_(idx) * (m_(idx) - 1) / 2;
    }
    arma::vec Di_bar_elem = -arma::trans(WLmd_.row(index + t));

    arma::mat Di_bar = arma::eye(n_atts_, n_atts_);
    Di_bar.diag() = Di_bar_elem;

    return Di_bar;
  }

  arma::mat mcbd::get_D_inv(const arma::uword i, const arma::uword t) const {
    arma::mat Tit_bar = get_T_bar(i, t);
    arma::mat Dit_bar = get_D_bar(i, t);
    arma::mat Dit_bar_inv = arma::diagmat(arma::pow(Dit_bar.diag(), -1));
    arma::mat Dit_inv = Tit_bar.t() * Dit_bar_inv * Tit_bar;

    return Dit_inv;
  }

  arma::mat mcbd::get_D_inv(const arma::uword i) const {
    arma::mat Di_inv = arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

    for(arma::uword t = 0; t != m_(i); ++t) {
      arma::mat Dit_inv = get_D_inv(i, t);

      arma::uword rindex = t * n_atts_;
      arma::uword cindex = t * n_atts_;

      Di_inv(rindex, cindex, arma::size(Dit_inv)) = Dit_inv;
    }

    return Di_inv;
  }

  arma::mat mcbd::get_Sigma_inv(const arma::uword i) const {
    arma::mat Ti = get_T(i);
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

    case 1:
      Xbta_ = X_ * bta_;
      Resid_ = Y_ - Xbta_;

      break;

    case 2:
      UGma_ = U_ * Gma_;
      VPsi_ = V_ * Psi_;
      WLmd_ = W_ * Lmd_;

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
    int debug = 1;

    UpdateMcbd(x);

    if (debug) std::cout << "mcbd::operator(): before for loop" << std::endl;
    double result = 0.0;
    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::vec ri = get_Resid(i);
      arma::mat Sigmai_inv = get_Sigma_inv(i);

      result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
    }

    arma::vec one_M = arma::ones<arma::vec>(arma::sum(m_));
    arma::vec one_J = arma::ones<arma::vec>(n_atts_);
    double log_det_Sigma = arma::as_scalar(one_M.t() * WLmd_ * one_J);

    result += log_det_Sigma;

    if (debug) std::cout << "mcbd::operator(): loglik = " << result << std::endl;

    return result;
  }

  void mcbd::Gradient(const arma::vec &x, arma::vec &grad) {
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
      Grad1(grad1);
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
  }

  void mcbd::Grad2(arma::vec &grad2) {
    arma::uword lgma, lpsi, llmd;
    lgma = (n_atts_ * poly_(1)) * n_atts_;
    lpsi = poly_(2)             * (n_atts_ * (n_atts_-1) / 2);
    llmd = poly_(3)             * n_atts_;

    arma::vec grad2_gma = arma::zeros<arma::vec>(lgma);
    arma::vec grad2_psi = arma::zeros<arma::vec>(lpsi);
    arma::vec grad2_lmd = arma::zeros<arma::vec>(llmd);

    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::mat Di_inv = get_D_inv(i);

    }

  }

  arma::mat mcbd::get_C(const arma::uword i, const arma::uword t) const {
    arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
    arma::mat Cit = arma::zeros<arma::mat>(n_atts_, lgma);

    if (t == 0) return Cit;
    else {
      arma::mat Tit_bar = get_T_bar(i, t);
      arma::mat eye_J   = arma::eye(n_atts_, n_atts_);
      for (arma::uword k = 0; k != t; ++k) {
        
      }
    }

    return Cit;
  }

  arma::mat mcbd::get_C(const arma::uword i) const {
    arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
    arma::mat Ci = arma::zeros<arma::mat>(n_atts_ * m_(i), lgma);

    for (arma::uword t = 1; t != m_(i); ++t) {
      amra::mat Cit = get_C(i, t);

      arma::uword rindex = n_atts_ * t;
      Ci.rows(rindex, rindex + n_atts_ - 1) = Cit;
    }

    return Ci;
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
