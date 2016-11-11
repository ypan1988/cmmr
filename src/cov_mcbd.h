/** 
 * @file cov_mcbd.h
 * @brief MCBD-based covariance matrix structure modelling
 *        
 * @author Yi Pan
 * @date 26/04/2016 
 */

#ifndef MCBD_H_
#define MCBD_H_

#include <armadillo>
#include "stats.h"

namespace jmcm
{
  /**
   * Covariance matrix structure modelling with Modified Cholesky Block 
   * Decomposition(MCBD), defined by 
   * $$
   * T \Sigma T^\top = D 
   * $$
   *
   * Kim, Chulmin, and Dale L. Zimmerman. "Unconstrained models for the 
   * covariance structure of multivariate longitudinal data." Journal of 
   * Multivariate Amalysis, 107, 104-118.
   */
  class CovMcbd
  {
  public:
    /**
     * Constructor of the CovMcbd object.
     *
     * @param n_atts The number of attibutions.
     * @param Y      The vector of responses for all subjects. 
     * @param X      The matrix of covariates for modelling mean structure.
     * @param U      The matrix of covariates for modelling matrix $H$.
     * @param V      The matrix of covariates for modelling matrix $B$.
     * @param W      The matrix of covariates for modelling matrix $T$.
     */      
    CovMcbd ( int n_atts,
              arma::vec &Y,
              arma::mat &X,
              arma::mat &U,
              arma::mat &V,
              arma::mat &W );

    /**
     * Destructor of the CovMcbd object.
     */
    ~CovMcbd();

    /**
     * Return the vector $Y$
     */
    arma::vec get_Y() const {
      return Y_;
    }

    /**
     * Return the matrix $X$
     */
    arma::vec get_X() const {
      return X_;
    }

    /**
     * Return the matrix $U$
     */
    arma::vec get_U() const {
      return U_;
    }

    /**
     * Return the matrix $V$
     */
    arma::vec get_V() const {
      return V_;
    }

    /**
     * Return the matrix $W$
     */
    arma::vec get_W() const {
      return W_;
    }

    /**
     * Return the sub-vector $Y_i$
     */
    inline arma::vec get_Y(const int i) const;

    /**
     * Return the sub-vector $X_i$
     */
    inline arma::mat get_X(const int i) const;

    /**
     * Return the sub-matrix $U_i$
     */
    inline arma::mat get_U(const int i) const;

    /**
     * Return the sub-matrix $V_i$
     */
    inline arma::mat get_V(const int i) const;

    /**
     * Return the sub-matrix $W_i$
     */
    inline arma::mat get_W(const int i) const;

    
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
    
    void UpdateCovMcbd ( const arma::vec &x );
    void UpdateParam ( const arma::vec &x );
    void UpdateModel();

  private:
    arma::vec Y_;      /**< responses for all subjects */
    arma::mat X_;      /**< covariates for modelling mean structure */
    arma::mat U_;      /**< covariates for modelling matrix $H$ */
    arma::mat V_;      /**< covariates for modelling matrix $B$ */
    arma::mat W_;      /**< covariates for modelling matrix $T$ */

    arma::vec bta_;    /**< parameters for modelling mean structure */
    arma::vec lmd_;    /**< parameters for modelling matrix $H$ */
    arma::vec psi_;    /**< parameters for modelling matrix $B$ */
    arma::vec gma_;    /**< parameters for modelling matrix $T$ */
    arma::vec tht_;    /**< all parameters as a vector */
    
    arma::mat MatLmd_;          /**< matrix form for $\lambda$ 0*/
    arma::mat MatPsi_;          /**< matrix form for $\psi$ */
    arma::mat MatGma_;          /**< matrix form for $\gamma$ */

    arma::mat Xbta_;            /**< $X \beta $ */
    arma::mat Ulmd_;            /**< $U \lambda$ */
    arma::mat Vpsi_;            /**< $V \psi$ */
    arma::mat Wgma_;            /**< $W \gamma$ */
    arma::mat Resid_;           /**< $Y - X \beta$ */

    arma::mat D_;               /**< matrix $D$ in MCBD */
    arma::mat T_;               /**< matrix $T$ in MCBD */
    arma::mat Sigma_;           /**< matrix $\Sigma$ in MCBD */
    arma::mat Sigma_inv_;       /**< inverse of $\Sigma$ */

    int n_atts_;                /**< number of attibutions */
    int n_dims_;                /**< number of dimensions(observations) */
    int n_subs_;                /**< number of subjects */
    
    double log_det_Sigma_;      /**< $\log|\Sigma_i|$ in loglik  */

    int free_param_;
    arma::vec poly_;

    void gma_vec2mat() {
      int q = poly_ ( 3 );
      for ( int i = 1; i <= n_atts_; ++i ) {
        int vindex = ( i - 1 ) * n_atts_;
        int mindex = ( i - 1 ) * q;
        arma::vec gma_i = gma_.rows ( vindex, vindex + n_atts_ * q - 1 );
        arma::mat MatGma_i = arma::reshape ( arma::mat ( gma_i ), q, n_atts_ );
        MatGma_.rows ( mindex, mindex + q - 1 ) = MatGma_i;
        // MatGma_.print ( "GAMMA = " );

      }
    }

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
