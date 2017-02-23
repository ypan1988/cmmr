//
// cov_mcbd.h:
//   MCBD-based covariance matrix structure modelling
//
// Centre for Musculoskeletel Research
// The University of Manchester
//
// Written by Yi Pan - ypan1988@gmail.com
//

#ifndef CMMR_MCBD_H_
#define CMMR_MCBD_H_

#include <RcppArmadillo.h>

struct mcbd_mode {
  arma::uword id_;
  void setid(const arma::uword id) { id_ = id; }
  inline explicit mcbd_mode(const arma::uword id) : id_(id) {}
};

inline bool operator==(const mcbd_mode &a, const mcbd_mode &b) {
  return (a.id_ == b.id_);
}

inline bool operator!=(const mcbd_mode &a, const mcbd_mode &b) {
  return (a.id_ != b.id_);
}

struct mcbd_mode_mcd : public mcbd_mode {
  inline mcbd_mode_mcd() : mcbd_mode(1) {}
};

struct mcbd_mode_acd : public mcbd_mode {
  inline mcbd_mode_acd() : mcbd_mode(2) {}
};

struct mcbd_mode_hpc : public mcbd_mode {
  inline mcbd_mode_hpc() : mcbd_mode(3) {}
};

static const mcbd_mode_mcd mcbd_mcd;
static const mcbd_mode_acd mcbd_acd;
static const mcbd_mode_hpc mcbd_hpc;

namespace cmmr
{
  /**
   * Covariance matrix structure modelling based on modified Cholesky block
   * decomposition (MCBD), an extention of Pourahmadi's modified Cholesky
   * decomposition (MCD) $ T \Sigma T^\top = D $ for the multivariate
   * longitudinal data setting by replacing the scalar entries of $T$ and $D$ by
   * $J x J$ block matrices.
   *
   * Each on-diagonal block of $D$ will be modelled by Cholesky-type approaches,
   * namely modified Cholesky decomposition (MCD), alternative Cholesky
   * decomposition (ACD) and hyperspherical parametrization of its Cholesky
   * factor (HPC).
   *
   * MCBD-MCD: Kim, Chulmin, and Dale L. Zimmerman. "Unconstrained models for
   * the covariance structure of multivariate longitudinal data." Journal of
   * Multivariate Amalysis, 107, 104-118.
   */
  class mcbd
  {
  private:
    const arma::uword n_atts_; // number of attributes J
    const arma::uword n_subs_; // number of subjects N
    arma::uvec  poly_;

    arma::uvec m_;
    arma::vec Y_;
    arma::mat X_, U_, V_, W_;

    arma::uword free_param_;
    arma::vec tht_, bta_, gma_, psi_, lmd_;
    arma::mat Gma_, Psi_, Lmd_;

    arma::vec Xbta_;
    arma::mat UGma_, VPsi_, WLmd_;
    arma::mat Resid_;

    const mcbd_mode mcbd_mode_obj_;

  public:
    mcbd(const arma::uvec &m, const arma::mat &Y, const arma::mat &X,
         const arma::mat &U, const arma::mat &V, const arma::mat &W,
         const mcbd_mode &mcbd_mode_obj);
    ~mcbd() {}

    arma::uvec get_m() const { return m_; }
    arma::vec get_Y() const { return Y_; }
    arma::mat get_X() const { return X_; }
    arma::mat get_U() const { return U_; }
    arma::mat get_V() const { return V_; }
    arma::mat get_W() const { return W_; }
    arma::vec get_Resid() const { return Resid_; }

    arma::uword get_m(const arma::uword i) const;
    arma::vec get_Y(const arma::uword i) const;
    arma::mat get_X(const arma::uword i) const;
    arma::mat get_U(const arma::uword i) const;
    arma::mat get_V(const arma::uword i) const;
    arma::mat get_W(const arma::uword i) const;
    arma::vec get_Resid(const arma::uword i) const;

    arma::mat get_U(const arma::uword i, const arma::uword t, const arma::uword k) const;
    arma::vec get_W(const arma::uword i, const arma::uword t) const;

    arma::vec get_theta() const { return tht_; }
    arma::vec get_beta() const { return bta_; }
    arma::vec get_gamma() const { return gma_; }
    arma::vec get_psi() const { return psi_; }
    arma::vec get_lambda() const { return lmd_; }

    void set_free_param(const arma::uword n) { free_param_ = n; }
    void set_theta(const arma::vec &x);
    void set_beta(const arma::vec &x);
    void set_gamma(const arma::vec &x);
    void set_psi(const arma::vec &x);
    void set_lambda(const arma::vec &x);

    arma::mat get_T(const arma::uword i, const arma::uword t, const arma::uword k) const;
    arma::mat get_T(const arma::uword i) const;

    arma::mat get_T_bar(const arma::uword i, const arma::uword t) const;
    arma::mat get_T_bar(const arma::uword i) const;

    arma::mat get_D_bar(const arma::uword i, const arma::uword t) const;

    arma::mat get_D_inv(const arma::uword i, const arma::uword t) const;
    arma::mat get_D_inv(const arma::uword i) const;
    arma::mat get_Sigma_inv(const arma::uword i) const;

    double operator() (const arma::vec &x);
    void Gradient(const arma::vec &x, arma::vec &grad);
    void Grad1(arma::vec &grad1);
    void Grad2(arma::vec &grad2);

    void UpdateMcbd (const arma::vec &x);
    void UpdateParam (const arma::vec &x);
    void UpdateModel();

    void UpdateBeta();
    //void UpdateGamma();

  private:
    arma::mat get_C(const arma::uword i, const arma::uword t) const;
    arma::mat get_C(const arma::uword i) const;

    arma::mat get_e(const arma::uword i, const arma::uword t) const;
    arma::mat get_e(const arma::uword i) const;

    arma::vec mcd_TResid_;
    arma::vec mcd_TTResid_;
    void      mcd_UpdateTResid();
    void      mcd_UpdateTTResid();
    arma::vec mcd_get_TResid(const arma::uword i);
    arma::vec mcd_get_TTResid(const arma::uword i);
    arma::mat mcd_CalcDbarDeriv(const arma::uword i, const arma::uword t) const;

    /* void gma_vec2mat() { */
    /*   int q = poly_ ( 3 ); */
    /*   for ( int i = 1; i <= n_atts_; ++i ) { */
    /*     int vindex = ( i - 1 ) * n_atts_; */
    /*     int mindex = ( i - 1 ) * q; */
    /*     arma::vec gma_i = gma_.rows ( vindex, vindex + n_atts_ * q - 1 ); */
    /*     arma::mat MatGma_i = arma::reshape ( arma::mat ( gma_i ), q, n_atts_ ); */
    /*     MatGma_.rows ( mindex, mindex + q - 1 ) = MatGma_i; */
    /*     // MatGma_.print ( "GAMMA = " ); */

    /*   } */
    /* } */

    /**
     * Update matrix D (also calculate log_det_Sigma)
     */
    /* void Update_D() { */
    /*   int debug = 0; */

    /*   log_det_Sigma_ = 0.0; */
    /*   arma::mat result = arma::eye ( n_atts_ * n_dims_, n_atts_ * n_dims_ ); */
    /*   for ( int i = 1; i <= n_dims_; ++i ) { */
    /*     arma::mat Dt = get_D ( i ); */

    /*     double val; */
    /*     double sign; */
    /*     arma::log_det(val, sign, Dt); */
    /*     log_det_Sigma_ += val; */

    /*     int rindex = ( i - 1 ) * n_atts_; */
    /*     int cindex = ( i - 1 ) * n_atts_; */
    /*     result ( rindex, cindex, arma::size ( Dt ) ) = Dt; */

    /*     if ( debug ) { */
    /*       result.print ( "result = " ); */
    /*     } */
    /*   } */

    /*   D_ = result; */
    /* } */

  };
}

#endif
