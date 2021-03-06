#include "mcbd.h"
#include "utils.h"

#include <RcppArmadillo.h>

namespace cmmr {
mcbd::mcbd(const arma::uvec &m, const arma::mat &Y, const arma::mat &X,
           const arma::mat &U, const arma::mat &V, const arma::mat &W,
           const mcbd_mode &mcbd_mode_obj)
    : n_atts_(Y.n_cols),
      n_subs_(m.n_elem),
      m_(m),
      V_(V),
      W_(W),
      mcbd_mode_obj_(mcbd_mode_obj) {
  poly_ = arma::zeros<arma::uvec>(4);
  poly_(0) = X.n_cols;
  poly_(1) = U.n_cols;
  poly_(2) = V.n_cols;
  poly_(3) = W.n_cols;

  arma::mat eye_J = arma::eye<arma::mat>(n_atts_, n_atts_);

  // initialize Y_
  Y_ = arma::vectorise(Y.t());
  cov_only_ = false;
  mean_ = Y_;

  // initialize X_
  for (arma::uword idx = 0; idx != X.n_rows; ++idx)
    X_ = arma::join_cols(X_, arma::kron(eye_J, X.row(idx)));

  // initialize U_
  for (arma::uword idx = 0; idx != U.n_rows; ++idx)
    U_ = arma::join_cols(U_, arma::kron(eye_J, U.row(idx)));

  free_param_ = 0;

  arma::uword ltht, lbta, lgma, lpsi, llmd;
  lbta = (n_atts_ * poly_(0)) * 1;
  lgma = (n_atts_ * poly_(1)) * n_atts_;
  lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  llmd = poly_(3) * n_atts_;
  ltht = lbta + lgma + lpsi + llmd;

  arma::vec x = arma::zeros<arma::vec>(ltht);

  if (mcbd_mode_obj_ == mcbd_hpc) {
    arma::vec psi0 = arma::zeros<arma::vec>(lpsi);
    arma::uword idx = 0;
    for (arma::uword j = 2; j <= n_atts_; ++j) {
      for (arma::uword k = 1; k <= j - 1; ++k) {
        psi0(idx * poly_(2)) = 0.5 * arma::datum::pi;
      }
    }
    x.subvec(lbta + lgma, lbta + lgma + lpsi - 1) = psi0;
  }

  UpdateMcbd(x);
}

arma::uword mcbd::get_m(const arma::uword i) const { return m_(i); }

arma::vec mcbd::get_Y(const arma::uword i) const {
  arma::mat Yi;
  if (i == 0)
    Yi = Y_.rows(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    Yi = Y_.rows(index, index + n_atts_ * m_(i) - 1);
  }

  return Yi;
}

arma::mat mcbd::get_X(const arma::uword i) const {
  arma::mat Xi;
  if (i == 0)
    Xi = X_.rows(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    Xi = X_.rows(index, index + n_atts_ * m_(i) - 1);
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
      arma::uword last_index =
          first_index + n_atts_ * m_(i) * (m_(i) - 1) / 2 - 1;
      Ui = U_.rows(first_index, last_index);
    }
  }

  return Ui;
}

arma::mat mcbd::get_V(const arma::uword i) const {
  arma::mat Vi;
  if (i == 0)
    Vi = V_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Vi = V_.rows(index, index + m_(i) - 1);
  }

  return Vi;
}

arma::mat mcbd::get_W(const arma::uword i) const {
  arma::mat Wi;
  if (i == 0)
    Wi = W_.rows(0, m_(0) - 1);
  else {
    int index = arma::sum(m_.subvec(0, i - 1));
    Wi = W_.rows(index, index + m_(i) - 1);
  }

  return Wi;
}

arma::vec mcbd::get_Resid(const arma::uword i) const {
  arma::mat Residi;
  if (i == 0)
    Residi = Resid_.rows(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    Residi = Resid_.rows(index, index + n_atts_ * m_(i) - 1);
  }

  return Residi;
}

// t = 1, ... , m_(i); k = 0, ... , t-1
arma::mat mcbd::get_U(const arma::uword i, const arma::uword t,
                      const arma::uword k) const {
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
  if (i != 0) rindex = arma::sum(m_.rows(0, i - 1));
  rindex += t;

  return V_.row(rindex).t();
}

arma::vec mcbd::get_W(const arma::uword i, const arma::uword t) const {
  arma::uword rindex = 0;
  if (i != 0) rindex = arma::sum(m_.rows(0, i - 1));
  rindex += t;

  return W_.row(rindex).t();
}

arma::vec mcbd::get_Resid(const arma::uword i, const arma::uword t) const {
  arma::uword rindex = 0;
  if (i != 0) rindex = n_atts_ * arma::sum(m_.rows(0, i - 1));
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

void mcbd::set_theta2(const arma::vec &x) {
  int fp2 = free_param_;
  free_param_ = 2;
  UpdateMcbd(x);
  free_param_ = fp2;
}

// void mcbd::set_gamma(const arma::vec &x) {
//   int fp2 = free_param_;
//   free_param_ = 2;
//   UpdateMcbd(x);
//   free_param_ = fp2;
// }

// void mcbd::set_psi(const arma::vec &x) {
//   int fp2 = free_param_;
//   free_param_ = 2;
//   UpdateMcbd(x);
//   free_param_ = fp2;
// }

// void mcbd::set_lambda(const arma::vec &x) {
//   int fp2 = free_param_;
//   free_param_ = 2;
//   UpdateMcbd(x);
//   free_param_ = fp2;
// }

arma::mat mcbd::get_T(const arma::uword i, const arma::uword t,
                      const arma::uword k) const {
  int mat_cnt = 0;
  if (i == 0)
    mat_cnt = 0;
  else {
    for (arma::uword idx = 0; idx != i; ++idx) {
      mat_cnt += m_(idx) * (m_(idx) - 1) / 2;
    }
  }

  for (arma::uword idx = 1; idx <= t; ++idx) {
    if (idx != t) {
      mat_cnt += idx;
    } else {
      mat_cnt += k;
    }
  }

  int rindex = mat_cnt * n_atts_;
  return UGma_.rows(rindex, rindex + n_atts_ - 1);
}

arma::mat mcbd::get_T(const arma::uword i) const {
  arma::mat Ti = arma::eye(n_atts_ * m_(i), n_atts_ * m_(i));
  for (arma::uword t = 1; t != m_(i); ++t) {
    for (arma::uword k = 0; k != t; ++k) {
      arma::mat Phi_itk = get_T(i, t, k);

      arma::uword rindex = t * n_atts_;
      arma::uword cindex = k * n_atts_;

      Ti(rindex, cindex, arma::size(Phi_itk)) = -Phi_itk;
    }
  }

  return Ti;
}

arma::mat mcbd::get_T_bar(const arma::uword i, const arma::uword t) const {
  arma::uword index = 0;
  if (i != 0) index = arma::sum(m_.rows(0, i - 1));

  arma::vec Tit_bar_elem = arma::trans(VPsi_.row(index + t));

  arma::mat Tit_bar = arma::eye(n_atts_, n_atts_);
  Tit_bar = dragonwell::ltrimat(n_atts_, Tit_bar_elem);

  if (mcbd_mode_obj_ == mcbd_hpc) {
    arma::mat mat_angles = Tit_bar;
    arma::mat result = arma::eye(n_atts_, n_atts_);

    result(0, 0) = 1;
    for (arma::uword j = 1; j != n_atts_; ++j) {
      result(j, 0) = std::cos(mat_angles(j, 0));
      result(j, j) =
          arma::prod(arma::prod(arma::sin(mat_angles.submat(j, 0, j, j - 1))));
      for (arma::uword l = 1; l != j; ++l) {
        result(j, l) = std::cos(mat_angles(j, l)) *
                       arma::prod(arma::prod(
                           arma::sin(mat_angles.submat(j, 0, j, l - 1))));
      }
    }
    return result;
  }

  return Tit_bar;
}

arma::mat mcbd::get_T_bar(const arma::uword i) const {
  arma::mat Ti_bar = arma::eye(n_atts_ * m_(i), n_atts_ * m_(i));
  for (arma::uword t = 0; t != m_(i); ++t) {
    arma::mat Tit = get_T_bar(i, t);
    arma::uword rindex = t * n_atts_;
    Ti_bar(rindex, rindex, arma::size(Tit)) = Tit;
  }

  return Ti_bar;
}

arma::mat mcbd::get_T_bar_inv(const arma::uword i, const arma::uword t) const {
  arma::mat Tit_bar = get_T_bar(i, t);
  arma::mat Tit_bar_inv;
  if (!arma::inv(Tit_bar_inv, Tit_bar)) Tit_bar_inv = arma::pinv(Tit_bar);
  return Tit_bar_inv;
}

arma::mat mcbd::get_T_bar_inv(const arma::uword i) const {
  arma::mat Ti_bar_inv =
      arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

  for (arma::uword t = 0; t != m_(i); ++t) {
    arma::mat Tit_bar_inv = get_T_bar_inv(i, t);

    arma::uword rindex = t * n_atts_;
    arma::uword cindex = t * n_atts_;

    Ti_bar_inv(rindex, cindex, arma::size(Tit_bar_inv)) = Tit_bar_inv;
  }

  return Ti_bar_inv;
}

arma::mat mcbd::get_D_bar(const arma::uword i, const arma::uword t) const {
  arma::uword index = 0;
  if (i != 0) index = arma::sum(m_.rows(0, i - 1));

  arma::vec Dit_bar_elem;

  if (mcbd_mode_obj_ == mcbd_mcd)
    Dit_bar_elem = arma::trans(arma::exp(WLmd_.row(index + t)));
  else if (mcbd_mode_obj_ == mcbd_acd || mcbd_mode_obj_ == mcbd_hpc)
    Dit_bar_elem = arma::trans(arma::exp(WLmd_.row(index + t) / 2));

  arma::mat Dit_bar = arma::eye(n_atts_, n_atts_);
  Dit_bar.diag() = Dit_bar_elem;

  return Dit_bar;
}

arma::mat mcbd::get_D_bar_inv(const arma::uword i, const arma::uword t) const {
  arma::mat Dit_bar = get_D_bar(i, t);
  return arma::diagmat(arma::pow(Dit_bar.diag(), -1));
}

arma::mat mcbd::get_D_bar_inv(const arma::uword i) const {
  arma::mat Di_bar_inv =
      arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

  for (arma::uword t = 0; t != m_(i); ++t) {
    arma::mat Dit_bar_inv = get_D_bar_inv(i, t);

    arma::uword rindex = t * n_atts_;
    arma::uword cindex = t * n_atts_;

    Di_bar_inv(rindex, cindex, arma::size(Dit_bar_inv)) = Dit_bar_inv;
  }

  return Di_bar_inv;
}

arma::mat mcbd::get_D(const arma::uword i, const arma::uword t) const {
  arma::mat Tit_bar = get_T_bar(i, t);
  arma::mat Tit_bar_inv = get_T_bar_inv(i, t);
  arma::mat Dit_bar = get_D_bar(i, t);
  arma::mat Dit;

  if (mcbd_mode_obj_ == mcbd_mcd) {
    Dit = Tit_bar_inv * Dit_bar * Tit_bar_inv.t();
  } else if (mcbd_mode_obj_ == mcbd_acd || mcbd_mode_obj_ == mcbd_hpc) {
    Dit = Dit_bar * Tit_bar * Tit_bar.t() * Dit_bar;
  }

  return Dit;
}

arma::mat mcbd::get_D_inv(const arma::uword i, const arma::uword t) const {
  arma::mat Tit_bar = get_T_bar(i, t);
  arma::mat Dit_bar_inv = get_D_bar_inv(i, t);
  arma::mat Dit_inv;

  if (mcbd_mode_obj_ == mcbd_mcd) {
    Dit_inv = Tit_bar.t() * Dit_bar_inv * Tit_bar;
  } else if (mcbd_mode_obj_ == mcbd_acd || mcbd_mode_obj_ == mcbd_hpc) {
    arma::mat Tit_bar_inv = get_T_bar_inv(i, t);
    Dit_inv = Dit_bar_inv * Tit_bar_inv.t() * Tit_bar_inv * Dit_bar_inv;
  }

  return Dit_inv;
}

arma::mat mcbd::get_D(const arma::uword i) const {
  arma::mat Di = arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

  for (arma::uword t = 0; t != m_(i); ++t) {
    arma::mat Dit = get_D(i, t);

    arma::uword rindex = t * n_atts_;
    arma::uword cindex = t * n_atts_;

    Di(rindex, cindex, arma::size(Dit)) = Dit;
  }

  return Di;
}

arma::mat mcbd::get_D_inv(const arma::uword i) const {
  arma::mat Di_inv = arma::zeros<arma::mat>(n_atts_ * m_(i), n_atts_ * m_(i));

  for (arma::uword t = 0; t != m_(i); ++t) {
    arma::mat Dit_inv = get_D_inv(i, t);

    arma::uword rindex = t * n_atts_;
    arma::uword cindex = t * n_atts_;

    Di_inv(rindex, cindex, arma::size(Dit_inv)) = Dit_inv;
  }

  return Di_inv;
}

arma::vec mcbd::get_mu(const arma::uword i) const {
  arma::vec mui;
  if (i == 0)
    mui = Xbta_.rows(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    mui = Xbta_.rows(index, index + n_atts_ * m_(i) - 1);
  }

  return mui;
}

arma::mat mcbd::get_Sigma(const arma::uword i) const {
  arma::mat Ti = get_T(i);
  arma::mat Ti_inv;
  if (!arma::inv(Ti_inv, Ti)) Ti_inv = arma::pinv(Ti);

  arma::mat Di = get_D(i);
  arma::mat Sigmai = Ti_inv * Di * Ti_inv.t();

  return Sigmai;
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

void mcbd::UpdateParam(const arma::vec &x) {
  arma::uword lbta, lgma, lpsi, llmd;
  lbta = (n_atts_ * poly_(0)) * 1;
  lgma = (n_atts_ * poly_(1)) * n_atts_;
  lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  llmd = poly_(3) * n_atts_;

  switch (free_param_) {
    case 0:
      tht_ = x;
      bta_ = x.rows(0, lbta - 1);
      gma_ = x.rows(lbta, lbta + lgma - 1);
      psi_ = x.rows(lbta + lgma, lbta + lgma + lpsi - 1);
      lmd_ = x.rows(lbta + lgma + lpsi, lbta + lgma + lpsi + llmd - 1);

      Gma_ = arma::reshape(arma::mat(gma_), U_.n_cols, n_atts_);
      Psi_ = arma::reshape(arma::mat(psi_), V_.n_cols,
                           (n_atts_ * (n_atts_ - 1) / 2));
      Lmd_ = arma::reshape(arma::mat(lmd_), W_.n_cols, n_atts_);

      break;

    case 1:
      tht_.rows(0, lbta - 1) = x;
      bta_ = x;

      break;

    case 2:
      tht_.rows(lbta, lbta + lgma + lpsi + llmd - 1) = x;
      gma_ = x.rows(0, lgma - 1);
      psi_ = x.rows(lgma, lgma + lpsi - 1);
      lmd_ = x.rows(lgma + lpsi, lgma + lpsi + llmd - 1);

      Gma_ = arma::reshape(arma::mat(gma_), U_.n_cols, n_atts_);
      Psi_ = arma::reshape(arma::mat(psi_), V_.n_cols,
                           (n_atts_ * (n_atts_ - 1) / 2));
      Lmd_ = arma::reshape(arma::mat(lmd_), W_.n_cols, n_atts_);

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
  }
}

void mcbd::UpdateModel() {
  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * bta_;

      UGma_ = U_ * Gma_;
      VPsi_ = V_ * Psi_;
      WLmd_ = W_ * Lmd_;
      Resid_ = Y_ - Xbta_;

      if (mcbd_mode_obj_ == mcbd_mcd) {
        mcd_UpdateTResid();
        mcd_UpdateTTResid();
      } else if (mcbd_mode_obj_ == mcbd_acd) {
        acd_UpdateTResid();
        acd_UpdateTDTResid();
      } else if (mcbd_mode_obj_ == mcbd_hpc) {
        hpc_UpdateTResid();
        hpc_UpdateTDTResid();
      }

      break;

    case 1:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = X_ * bta_;

      Resid_ = Y_ - Xbta_;

      if (mcbd_mode_obj_ == mcbd_mcd) {
        mcd_UpdateTResid();
        mcd_UpdateTTResid();
      } else if (mcbd_mode_obj_ == mcbd_acd) {
        acd_UpdateTResid();
        acd_UpdateTDTResid();
      } else if (mcbd_mode_obj_ == mcbd_hpc) {
        hpc_UpdateTResid();
        hpc_UpdateTDTResid();
      }

      break;

    case 2:
      UGma_ = U_ * Gma_;
      VPsi_ = V_ * Psi_;
      WLmd_ = W_ * Lmd_;

      if (mcbd_mode_obj_ == mcbd_mcd) {
        mcd_UpdateTResid();
        mcd_UpdateTTResid();
      } else if (mcbd_mode_obj_ == mcbd_acd) {
        acd_UpdateTResid();
        acd_UpdateTDTResid();
      } else if (mcbd_mode_obj_ == mcbd_hpc) {
        hpc_UpdateTResid();
        hpc_UpdateTDTResid();
      }

      break;

    default:
      Rcpp::Rcout << "Wrong value for free_param_" << std::endl;
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

void mcbd::UpdateTheta2(const arma::vec &x) {}

double mcbd::operator()(const arma::vec &x) {
  UpdateMcbd(x);

  double result = 0.0;
  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);

    result += arma::as_scalar(ri.t() * Sigmai_inv * ri);
  }

  arma::vec one_M = arma::ones<arma::vec>(arma::sum(m_));
  arma::vec one_J = arma::ones<arma::vec>(n_atts_);
  double log_det_Sigma = 0.0;

  if (mcbd_mode_obj_ == mcbd_mcd || mcbd_mode_obj_ == mcbd_acd)
    log_det_Sigma = arma::as_scalar(one_M.t() * WLmd_ * one_J);

  if (mcbd_mode_obj_ == mcbd_hpc) {
    for (arma::uword i = 0; i != n_subs_; ++i) {
      arma::mat Ti_bar = get_T_bar(i);
      log_det_Sigma += 2 * arma::sum(arma::log(Ti_bar.diag()));
    }
    log_det_Sigma += arma::as_scalar(one_M.t() * WLmd_ * one_J);
  }

  result += log_det_Sigma;

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
  UpdateMcbd(x);

  arma::uword ltht, lbta, lgma, lpsi, llmd;
  lbta = (n_atts_ * poly_(0)) * 1;
  lgma = (n_atts_ * poly_(1)) * n_atts_;
  lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  llmd = poly_(3) * n_atts_;
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

    default:
      Rcpp::Rcout << "wrong value for free_param_" << std::endl;
  }
}

void mcbd::Grad1(arma::vec &grad1) {
  arma::uword lbta = (n_atts_ * poly_(0)) * 1;
  grad1 = arma::zeros<arma::vec>(lbta);
  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec Yi = get_Y(i);
    arma::mat Xi = get_X(i);
    arma::mat Sigmai_inv = get_Sigma_inv(i);

    grad1 += Xi.t() * Sigmai_inv * (Yi - Xi * bta_);
  }
  grad1 *= -2;
}

void mcbd::Grad2(arma::vec &grad2) {
  arma::uword lgma, lpsi, llmd;
  lgma = (n_atts_ * poly_(1)) * n_atts_;
  lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  llmd = poly_(3) * n_atts_;

  arma::vec grad_gma = arma::zeros<arma::vec>(lgma);
  arma::vec grad_psi = arma::zeros<arma::vec>(lpsi);
  arma::vec grad_lmd = arma::zeros<arma::vec>(llmd);

  const arma::vec one_J = arma::ones<arma::vec>(n_atts_);
  const arma::mat eye_Jr = arma::eye<arma::mat>(llmd, llmd);

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::mat Ci = get_C(i);
    arma::mat Di_inv = get_D_inv(i);
    arma::mat ei = get_Resid(i);

    grad_gma += Ci.t() * Di_inv * (ei - Ci * gma_);

    // Calculate grad_psi and grad_lmd in MCBD-MCD
    if (mcbd_mode_obj_ == mcbd_mcd) {
      arma::mat Gi = mcd_get_G(i);
      arma::mat Di_bar_inv = get_D_bar_inv(i);
      arma::mat epsi = mcd_get_TResid(i);

      grad_psi += -Gi.t() * Di_bar_inv * (Gi * psi_ + epsi);

      arma::mat one_T = arma::ones<arma::vec>(m_(i));
      arma::mat Wi = get_W(i);

      grad_lmd += -0.5 * arma::kron(one_J.t(), one_T.t() * Wi).t();
      arma::vec TTr = mcd_get_TTResid(i);
      for (arma::uword t = 0; t != m_(i); ++t) {
        arma::uword index = n_atts_ * t;
        grad_lmd += -0.5 * arma::kron(one_J.t(), eye_Jr) *
                    mcd_CalcDbarDeriv(i, t) *
                    arma::pow(TTr.subvec(index, index + n_atts_ - 1), 2);
      }
    } else if (mcbd_mode_obj_ == mcbd_acd) {
      arma::mat one_T = arma::ones<arma::vec>(m_(i));
      arma::mat Wi = get_W(i);

      grad_lmd += -0.5 * arma::kron(one_J.t(), one_T.t() * Wi).t();

      arma::vec epsi = acd_get_TResid(i);
      arma::vec TDTr = acd_get_TDTResid(i);
      for (arma::uword t = 0; t != m_(i); ++t) {
        arma::uword index = n_atts_ * t;

        arma::vec epsit = epsi.subvec(index, index + n_atts_ - 1);
        arma::vec xi_it = TDTr.subvec(index, index + n_atts_ - 1);

        // arma::mat Tit_bar_inv = get_T_bar(i, t).i();
        arma::mat Tit_bar_inv = get_T_bar_inv(i, t);
        arma::mat Tit_bar_trans_deriv = acd_CalcTransTbarDeriv(i, t);

        grad_psi += arma::kron(xi_it.t(), arma::eye(lpsi, lpsi)) *
                    Tit_bar_trans_deriv * Tit_bar_inv.t() * xi_it;

        grad_lmd -= arma::kron(epsit.t(), eye_Jr) * acd_CalcDbarDeriv(i, t) *
                    Tit_bar_inv.t() * xi_it;
      }
    } else if (mcbd_mode_obj_ == mcbd_hpc) {
      arma::mat one_T = arma::ones<arma::vec>(m_(i));
      arma::mat Wi = get_W(i);

      grad_lmd += -0.5 * arma::kron(one_J.t(), one_T.t() * Wi).t();
      arma::vec epsi = hpc_get_TResid(i);
      arma::vec TDTr = hpc_get_TDTResid(i);
      for (arma::uword t = 0; t != m_(i); ++t) {
        arma::uword index = n_atts_ * t;

        arma::vec epsit = epsi.subvec(index, index + n_atts_ - 1);
        arma::vec xi_it = TDTr.subvec(index, index + n_atts_ - 1);

        arma::mat Tit_bar = get_T_bar(i, t);
        arma::mat Tit_bar_inv = get_T_bar_inv(i, t);
        arma::mat Tit_bar_trans_deriv = hpc_CalcTransTbarDeriv(i, t);

        for (arma::uword j = 0; j != n_atts_; ++j) {
          grad_psi -= 1 / Tit_bar(j, j) * hpc_CalcTitjkDeriv(i, t, j, j);
        }

        grad_psi += arma::kron(xi_it.t(), arma::eye(lpsi, lpsi)) *
                    Tit_bar_trans_deriv * Tit_bar_inv.t() * xi_it;

        grad_lmd -= arma::kron(epsit.t(), eye_Jr) * hpc_CalcDbarDeriv(i, t) *
                    Tit_bar_inv.t() * xi_it;
      }
    }
  }

  grad2 = -2 * dragonwell::join_vecs({grad_gma, grad_psi, grad_lmd});
}

arma::mat mcbd::get_C(const arma::uword i, const arma::uword t) const {
  arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
  arma::mat Cit = arma::zeros<arma::mat>(n_atts_, lgma);

  if (t == 0)
    return Cit;
  else {
    for (arma::uword k = 0; k != t; ++k) {
      arma::vec eik = get_Resid(i, k);
      Cit += arma::kron(eik.t(), get_U(i, t, k));
    }
  }

  return Cit;
}

arma::mat mcbd::get_C(const arma::uword i) const {
  arma::uword lgma = (n_atts_ * poly_(1)) * n_atts_;
  arma::mat Ci = arma::zeros<arma::mat>(n_atts_ * m_(i), lgma);

  for (arma::uword t = 1; t != m_(i); ++t) {
    arma::mat Cit = get_C(i, t);

    arma::uword rindex = n_atts_ * t;
    Ci.rows(rindex, rindex + n_atts_ - 1) = Cit;
  }

  return Ci;
}

void mcbd::mcd_UpdateTResid() {
  mcd_TResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);

    arma::vec Tr = Ti * ri;
    if (i == 0)
      mcd_TResid_.subvec(0, n_atts_ * m_(0) - 1) = Tr;
    else {
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
    if (i == 0)
      mcd_TTResid_.subvec(0, n_atts_ * m_(0) - 1) = TTr;
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      mcd_TTResid_.subvec(index, index + n_atts_ * m_(i) - 1) = TTr;
    }
  }
}

arma::vec mcbd::mcd_get_TResid(const arma::uword i) const {
  if (i == 0)
    return mcd_TResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return mcd_TResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

arma::vec mcbd::mcd_get_TTResid(const arma::uword i) const {
  if (i == 0)
    return mcd_TTResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return mcd_TTResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

void mcbd::acd_UpdateTResid() {
  acd_TResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);

    arma::vec Tr = Ti * ri;
    if (i == 0)
      acd_TResid_.subvec(0, n_atts_ * m_(0) - 1) = Tr;
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      acd_TResid_.subvec(index, index + n_atts_ * m_(i) - 1) = Tr;
    }
  }
}

void mcbd::acd_UpdateTDTResid() {
  acd_TDTResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);

    arma::mat Di_bar_inv = get_D_bar_inv(i);
    arma::mat Ti_bar_inv = get_T_bar_inv(i);

    arma::vec TDTr = Ti_bar_inv * Di_bar_inv * Ti * ri;
    if (i == 0)
      acd_TDTResid_.subvec(0, n_atts_ * m_(0) - 1) = TDTr;
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      acd_TDTResid_.subvec(index, index + n_atts_ * m_(i) - 1) = TDTr;
    }
  }
}

arma::vec mcbd::acd_get_TResid(const arma::uword i) const {
  if (i == 0)
    return acd_TResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return acd_TResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

arma::vec mcbd::acd_get_TDTResid(const arma::uword i) const {
  if (i == 0)
    return acd_TDTResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return acd_TDTResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

void mcbd::hpc_UpdateTResid() {
  hpc_TResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);

    arma::vec Tr = Ti * ri;
    if (i == 0)
      hpc_TResid_.subvec(0, n_atts_ * m_(0) - 1) = Tr;
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      hpc_TResid_.subvec(index, index + n_atts_ * m_(i) - 1) = Tr;
    }
  }
}

void mcbd::hpc_UpdateTDTResid() {
  hpc_TDTResid_ = arma::zeros<arma::vec>(n_atts_ * arma::sum(m_));

  for (arma::uword i = 0; i != n_subs_; ++i) {
    arma::vec ri = get_Resid(i);
    arma::mat Ti = get_T(i);

    arma::mat Di_bar_inv = get_D_bar_inv(i);
    arma::mat Ti_bar_inv = get_T_bar_inv(i);

    arma::vec TDTr = Ti_bar_inv * Di_bar_inv * Ti * ri;
    if (i == 0)
      hpc_TDTResid_.subvec(0, n_atts_ * m_(0) - 1) = TDTr;
    else {
      int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
      hpc_TDTResid_.subvec(index, index + n_atts_ * m_(i) - 1) = TDTr;
    }
  }
}

arma::vec mcbd::hpc_get_TResid(const arma::uword i) const {
  if (i == 0)
    return hpc_TResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return hpc_TResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

arma::vec mcbd::hpc_get_TDTResid(const arma::uword i) const {
  if (i == 0)
    return hpc_TDTResid_.subvec(0, n_atts_ * m_(0) - 1);
  else {
    int index = n_atts_ * arma::sum(m_.subvec(0, i - 1));
    return hpc_TDTResid_.subvec(index, index + n_atts_ * m_(i) - 1);
  }
}

arma::mat mcbd::mcd_get_V(const arma::uword i, const arma::uword t,
                          const arma::uword j) const {
  arma::uword lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  arma::vec vit = get_V(i, t);

  arma::uword index = 0;
  if (j != 1) {
    for (arma::uword cnt = 1; cnt != j; ++cnt) index += cnt;
  }

  arma::mat result = arma::zeros<arma::mat>(j, lpsi);
  for (arma::uword k = 0; k != j; ++k) {
    result(k, poly_(2) * (index + k), arma::size(vit.t())) = vit.t();
  }

  return result.t();
}

arma::mat mcbd::mcd_get_G(const arma::uword i) const {
  arma::uword lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  arma::mat result = arma::zeros<arma::mat>(n_atts_ * m_(i), lpsi);
  for (arma::uword t = 0, index = 0; t != m_(i); ++t) {
    arma::vec epsi = mcd_get_TResid(i);
    arma::vec epsit = epsi.subvec(n_atts_ * t, n_atts_ * t + n_atts_ - 1);
    for (arma::uword j = 0; j != n_atts_; ++j) {
      arma::vec gitj = arma::zeros<arma::vec>(lpsi);
      if (j == 0) {
        ++index;
        continue;
      }
      arma::mat Vitj = mcd_get_V(i, t, j);
      gitj = Vitj * epsit.subvec(0, j - 1);
      result.row(index++) = gitj.t();
    }
  }

  return result;
}

arma::mat mcbd::mcd_CalcDbarDeriv(const arma::uword i,
                                  const arma::uword t) const {
  const arma::uword llmd = poly_(3) * n_atts_;
  arma::mat result = arma::zeros<arma::mat>(n_atts_ * llmd, n_atts_);

  arma::vec wit = get_W(i, t);
  arma::mat Dit_bar = get_D_bar(i, t);
  for (arma::uword j = 0; j != n_atts_; ++j) {
    result(j * llmd + j * poly_(3), j, arma::size(poly_(3), 1)) =
        -wit / Dit_bar(j, j);
  }

  return result;
}

arma::mat mcbd::acd_CalcTransTbarDeriv(const arma::uword i,
                                       const arma::uword t) const {
  arma::uword lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  arma::mat result = arma::zeros<arma::mat>(lpsi * n_atts_, n_atts_);

  arma::vec vit = get_V(i, t);

  arma::uword index = 0;
  for (arma::uword k = 1; k != n_atts_; ++k) {
    for (arma::uword j = 0; j < k; ++j) {
      arma::vec tmp = arma::zeros<arma::vec>(lpsi);
      tmp.subvec(index * vit.n_elem, index * vit.n_elem + vit.n_elem - 1) = vit;
      ++index;

      result.submat(j * lpsi, k, j * lpsi + lpsi - 1, k) = tmp;
    }
  }

  return result;
}

arma::mat mcbd::acd_CalcDbarDeriv(const arma::uword i,
                                  const arma::uword t) const {
  const arma::uword llmd = poly_(3) * n_atts_;
  arma::mat result = arma::zeros<arma::mat>(n_atts_ * llmd, n_atts_);

  arma::vec wit = get_W(i, t);
  arma::mat Dit_bar = get_D_bar(i, t);
  for (arma::uword j = 0; j != n_atts_; ++j) {
    result(j * llmd + j * poly_(3), j, arma::size(poly_(3), 1)) =
        -0.5 * wit / Dit_bar(j, j);
  }

  return result;
}

arma::mat mcbd::hpc_get_angles(const arma::uword i, const arma::uword t) const {
  arma::uword index = 0;
  if (i != 0) index = arma::sum(m_.rows(0, i - 1));

  arma::vec angles_elem = arma::trans(VPsi_.row(index + t));

  arma::mat result = arma::zeros<arma::mat>(n_atts_, n_atts_);
  result = dragonwell::ltrimat(n_atts_, angles_elem);

  return result;
}

arma::vec mcbd::hpc_CalcTitjkDeriv(const arma::uword i, const arma::uword t,
                                   const arma::uword j,
                                   const arma::uword k) const {
  arma::uword lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);

  arma::mat mat_angles = hpc_get_angles(i, t);
  arma::mat Tit_bar = get_T_bar(i, t);

  arma::vec result = arma::zeros<arma::vec>(lpsi);

  arma::vec vit = get_V(i, t);

  if (j == 0 || j < k)
    return result;
  else if (j == k) {
    arma::uword idx = 0;
    for (arma::uword rindex = 0; rindex != j; ++rindex)
      idx += poly_(2) * rindex;

    for (arma::uword cindex = 0; cindex != k; ++cindex) {
      result.subvec(idx, idx + poly_(2) - 1) = Tit_bar(j, j) /
                                               sin(mat_angles(j, cindex)) *
                                               cos(mat_angles(j, cindex)) * vit;
      idx += poly_(2);
    }
  } else if (k == 0) {
    arma::uword idx = 0;
    for (arma::uword rindex = 0; rindex != j; ++rindex)
      idx += poly_(2) * rindex;

    result.subvec(idx, idx + poly_(2) - 1) = -sin(mat_angles(j, 0)) * vit;

  } else {
    arma::uword idx = 0;
    for (arma::uword rindex = 0; rindex != j; ++rindex)
      idx += poly_(2) * rindex;

    for (arma::uword cindex = 0; cindex <= k - 1; ++cindex) {
      result.subvec(idx, idx + poly_(2) - 1) = Tit_bar(j, k) /
                                               sin(mat_angles(j, cindex)) *
                                               cos(mat_angles(j, cindex)) * vit;
      idx += poly_(2);
    }
    result.subvec(idx, idx + poly_(2) - 1) =
        Tit_bar(j, k) / cos(mat_angles(j, k)) * (-sin(mat_angles(j, k))) * vit;
  }

  return result;
}

arma::mat mcbd::hpc_CalcTransTbarDeriv(const arma::uword i,
                                       const arma::uword t) const {
  arma::uword lpsi = poly_(2) * (n_atts_ * (n_atts_ - 1) / 2);
  arma::mat result = arma::zeros<arma::mat>(lpsi * n_atts_, n_atts_);

  for (arma::uword k = 1; k != n_atts_; ++k) {
    for (arma::uword j = 0; j <= k; ++j) {
      result.submat(j * lpsi, k, j * lpsi + lpsi - 1, k) =
          hpc_CalcTitjkDeriv(i, t, k, j);
    }
  }

  return result;
}

arma::mat mcbd::hpc_CalcDbarDeriv(const arma::uword i,
                                  const arma::uword t) const {
  const arma::uword llmd = poly_(3) * n_atts_;
  arma::mat result = arma::zeros<arma::mat>(n_atts_ * llmd, n_atts_);

  arma::vec wit = get_W(i, t);
  arma::mat Dit_bar = get_D_bar(i, t);
  for (arma::uword j = 0; j != n_atts_; ++j) {
    result(j * llmd + j * poly_(3), j, arma::size(poly_(3), 1)) =
        -0.5 * wit / Dit_bar(j, j);
  }

  return result;
}

}  // namespace cmmr
