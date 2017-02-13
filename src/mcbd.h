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
  const arma::uword id_;
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
    arma::mat UGma_, VPsi_, WLmd;
    arma::mat Resid_;

  public:
    mcbd(arma::uvec &m, arma::mat &Y, arma::mat &X,
         arma::mat &U, arma::mat &V, arma::mat &W );
    ~mcbd();

    arma::uvec get_m() const { return m_; }
    arma::vec get_Y() const { return Y_; }
    arma::mat get_X() const { return X_; }
    arma::mat get_U() const { return U_; }
    arma::mat get_V() const { return V_; }
    arma::mat get_W() const { return W_; }

    arma::uword get_m(const arma::uword i) const { return m_(i); }
    arma::vec get_Y(const arma::uword i) const {
        arma::mat Yi;
        if (i == 0) Yi = Y_.rows(0, m_(0)-1);
        else {
            int index = arma::sum(m_subvec(0, i - 1));
            Yi = Y_.rows(index, index + m_(i) -1);
        }

        return arma::vectorise(Yi.t());
    }

    arma::mat get_X(const arma::uword i) const {
        arma::mat Xi;
        if (i==0) Xi = X_.rows(0, m_(0)-1);
        else {
            int index = arma::sum(m_subvec(0, i - 1));
            Xi = X_.rows(index, index + m_(i) -1);
        }

        arma::mat eye_J = arma::eye<arma::mat>(n_atts_, n_atts_);
        arma::mat result;
        for(arma::uword t = 0; t != m_(i); ++t) {
            result = arma::join_cols(result, arma::kron(eye_J, Xi.row(t)));
        }

        return result;
    }
    arma::mat get_X() const {
        arma::mat Xi, result;
        for(arma::uword i = 0; i != n_sub_; ++i) {
            Xi = get_X(i);
            result = arma::join_cols(result, Xi);
        }
        return result;
    }

    arma::mat get_U(const arma::uword i) const {
        arma::mat Ui;
        if (m_(i) != 1) {
            if (i == 1) {
                arma::uword first_index = 0, last_index = m_(0) * (m_(0) - 1) / 2 - 1;
                Ui = U_.rows(first_index, last_index);
            } else {
                arma::uword first_index = 0;
                for (arma::uword idx = 0; idx != i; ++idx) {
                    first_index += m_(idx) * (m_(idx) - 1) / 2 - 1;
                }
                arma::uword last_index = first_index + m_(idx) * (m_(idx) - 1) / 2 - 1;
                Ui = U_.rows(first_index, last_index);
            }
        }

        arma::mat eye_J = arma::eye<arma::mat>(n_atts_, n_atts_);
        arma::mat result;
        for(arma::uword t = 0; t != Ui.n_rows; ++t) {
            result = arma::join_cols(result, arma::kron(eye_J, Ui.row(t)));
        }

        return result;
    }
    arma::mat get_X() const {
        arma::mat Xi, result;
        for(arma::uword i = 0; i != n_sub_; ++i) {
            Xi = get_X(i);
            result = arma::join_cols(result, Xi);
        }
        return result;
    }

    arma::mat get_V() const { return V_; }
    arma::mat get_V(const arma::uword i) const {
        if (i == 0) Vi = V_.rows(0, m_(0) - 1);
        else {
            int index = arma::sum(m_subvec(0, i - 1));
            Vi = V_.rows(index, index + m_(i) -1);
        }
    }

    arma::mat get_W() const { return W_; }
    arma::mat get_W(const arma::uword i) const {
        if (i == 0) Wi = W_.rows(0, m_(0) - 1);
        else {
            int index = arma::sum(m_subvec(0, i - 1));
            Wi = W_.rows(index, index + m_(i) -1);
        }
    }

    arma::vec get_Resid ( const int i ) const {
      int debug = 0;

      int vindex = n_dims_ * n_atts_ * ( i - 1 );

      if(debug){
        std::cout << "dim of Resid_: " << Resid_.n_elem << std::endl
                  << "n_dims_: " << n_dims_ << std::endl
                  << "n_atts_: " << n_atts_ << std::endl
                  << "vindex : " << vindex << std::endl;
      }

      arma::vec result
        = Resid_.rows ( vindex, vindex + n_dims_ * n_atts_ - 1 );

      return result;
    }

    arma::mat get_D ( const int t ) const {
      int debug = 0;

      arma::mat Ht = get_H ( t );
      if ( debug ) {
        Ht.print ( "Ht = " );
      }
      arma::mat Bt = get_B ( t );
      if ( debug ) {
        Bt.print ( "Bt = " );
      }
      arma::mat Bt_inv = Bt.i();
      if ( debug ) {
        Bt_inv.print ( "Bt_inv = " );
      }

      return Bt_inv * Ht * Bt_inv.t();
    }

    arma::mat get_T ( const int t, const int k ) const {

      int debug = 0;

      int index = 0;
      for ( int i = 2; i <= t; ++i ) {
        if ( i != t ) {
          index += i - 1;
        } else {
          index += k;
        }
      }

      int mindex = ( index - 1 ) * n_atts_;
      arma::mat result  = Wgma_.rows ( mindex, mindex + n_atts_ - 1 );

      if ( debug ) {
        Wgma_.print ( "Wgma = " );
        result.print ( "result = " );
      }

      return result;
    }

    /**
     * Calculate -2 * loglik as a object function.
     */
    double operator() ( const arma::vec &x );
    void Gradient(const arma::vec &x, arma::vec &grad);
    void Grad1(arma::vec &grad1);
    void Grad2(arma::vec &grad2);

    void UpdateMcbd ( const arma::vec &x );
    void UpdateParam ( const arma::vec &x );
    void UpdateModel();

    void UpdateBeta();

  private:


    arma::mat D_;               /**< matrix $D$ in MCBD */
    arma::mat T_;               /**< matrix $T$ in MCBD */
    arma::mat Sigma_;           /**< matrix $\Sigma$ in MCBD */
    arma::mat Sigma_inv_;       /**< inverse of $\Sigma$ */

    double log_det_Sigma_;      /**< $\log|\Sigma_i|$ in loglik  */
    

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

    arma::mat get_H ( const int i ) const {

      int debug = 0;
      arma::mat Hi = arma::eye ( n_atts_, n_atts_ );
      Hi.diag() = Ulmd_.row ( i - 1 );
      if ( debug ) {
        Hi.print ( "Hi = " );
      }

      return arma::exp ( Hi );
    }

    arma::mat get_B ( const int i ) const {
      arma::mat Bi = arma::eye ( n_atts_, n_atts_ );
      arma::vec Bi_elem = -arma::trans ( Vpsi_.row ( i-1 ) );

      Bi = ltrimat ( n_atts_, Bi_elem );

      return Bi;
    }

    void Update_T() {
      int debug = 0;

      arma::mat result = arma::eye ( n_atts_ * n_dims_, n_atts_ * n_dims_ );
      for ( int i = 1; i <= n_dims_; ++i ) {
        for ( int j = 1; j < i; ++j ) {
          arma::mat Ttk = get_T ( i, j );

          int rindex = ( i - 1 ) * n_atts_;
          int cindex = ( j - 1 ) * n_atts_;
          result ( rindex, cindex, arma::size ( Ttk ) ) = Ttk;

          if ( debug ) {
            result.print ( "T = " );
          }
        }
      }

      T_ = result;
    }

    /**
     * Update matrix D (also calculate log_det_Sigma) 
     */
    void Update_D() {
      int debug = 0;
        
      log_det_Sigma_ = 0.0;
      arma::mat result = arma::eye ( n_atts_ * n_dims_, n_atts_ * n_dims_ );
      for ( int i = 1; i <= n_dims_; ++i ) {
        arma::mat Dt = get_D ( i );
            
        double val;
        double sign;
        arma::log_det(val, sign, Dt);
        log_det_Sigma_ += val;
            
        int rindex = ( i - 1 ) * n_atts_;
        int cindex = ( i - 1 ) * n_atts_;
        result ( rindex, cindex, arma::size ( Dt ) ) = Dt;

        if ( debug ) {
          result.print ( "result = " );
        }
      }

      D_ = result;
    }

    void Update_Sigma() {
      Update_D();
      Update_T();

      arma::mat D_inv = D_.i();
      arma::mat T_inv = T_.i();

      Sigma_ = T_inv * D_ * T_inv.t();
      Sigma_inv_ = T_.t() * D_inv * T_;

    }
  };
}

#endif
