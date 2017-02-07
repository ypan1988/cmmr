#include "mcbd.h"

#include <iostream>
#include <armadillo>

namespace cmmr {
  mcbd::mcbd (arma::uvec m, arma::vec &Y, arma::mat &X,
              arma::mat &U, arma::mat &V, arma::mat &W ) :
    n_atts_ ( Y_.n_cols ), n_subs_ ( m.n_elem ),
    m_ ( m ), Y_ ( Y ), X_ ( X ), U_ ( U ), V_ ( V ), W_ ( W ),
  {
    int debug = 1;

    poly_ = arma::zeros<arma::uvec> ( 4 );
    poly_ ( 0 ) = X_.n_cols;
    poly_ ( 1 ) = U_.n_cols;
    poly_ ( 2 ) = V_.n_cols;
    poly_ ( 3 ) = W_.n_cols / n_atts_;

    if ( debug ) {
        std::cout << "n_atts_ = " << n_atts_ << std::endl
                  << "n_subs_ = " << n_subs_ << std::endl;
    }

    MatLmd_ = arma::zeros<arma::mat> ( poly_ ( 1 ), n_atts_ );
    MatPsi_ = arma::zeros<arma::mat> ( poly_ ( 2 ), n_atts_ * ( n_atts_ - 1 ) / 2 );
    MatGma_ = arma::zeros<arma::mat> ( poly_ ( 3 ) * n_atts_, n_atts_ );

    int lbta = X_.n_cols;
    int llmd = poly_ ( 1 ) * n_atts_;
    int lpsi = poly_ ( 2 ) * n_atts_ * ( n_atts_ - 1 ) / 2;
    int lgma = poly_ ( 3 ) * n_atts_ * n_atts;

    tht_ = arma::zeros<arma::vec> ( lbta + llmd + lpsi + lgma );
    bta_ = arma::zeros<arma::vec> ( lbta );
    lmd_ = arma::zeros<arma::vec> ( llmd );
    psi_ = arma::zeros<arma::vec> ( lpsi );
    gma_ = arma::zeros<arma::vec> ( lgma );

    free_param_ = 0;

    if ( debug ) std::cout << "CovMcbd obj created..." << std::endl;
}

CovMcbd::~CovMcbd() {}

arma::vec CovMcbd::get_Y ( const int i ) const {
    int vindex = n_dims_ * n_atts_ * ( i - 1 );

    arma::vec result
        = Y_.rows ( vindex, vindex + n_dims_ * n_atts_ - 1 );

    return result;
}

arma::mat CovMcbd::get_X ( const int i ) const {
    int rindex = n_dims_ * n_atts_ * ( i - 1 );
    arma::mat result = X_.rows ( rindex, rindex + n_dims_ * n_atts_ - 1 );

    return result;
}


double CovMcbd::operator() ( const arma::vec &x ) {
    int debug = 1;

    UpdateCovMcbd ( x );

    if ( debug ) std::cout << "before for loop" << std::endl;

    double result = 0.0;
    for ( int i = 1; i <= n_subs_; ++i ) {
        if ( debug ) std::cout << "i = " << i << std::endl;
        arma::vec ri = get_Resid ( i );
        //if ( debug ) ri.print("ri = ");
        result += arma::as_scalar ( ri.t() * Sigma_inv_ * ri );
    }

    result += log_det_Sigma_;

    if ( debug ) std::cout << "loglik = " << result << std::endl;

    return result;
}


void CovMcbd::Gradient ( const arma::vec& x, arma::vec& grad ) {
    int debug = 0;
    UpdateCovMcbd ( x );
    arma::vec grad1, grad2, grad3, grad4;

    switch ( free_param_ ) {
    case 0:
        Grad1 ( grad1 );
    }

}

void CovMcbd::Grad1 ( arma::vec& grad1 ) {
    int debug = 1;

    int lbta = bta_.n_elem;
    grad1 = arma::zeros<arma::vec> ( lbta );
    for ( int i = 1; i <= n_subs_; ++i ) {
        arma::vec Yi = get_Y ( i );
        arma::mat Xi = get_X ( i );
        grad1 += Xi.t() * Sigma_inv_ * ( Yi - Xi * bta_ );
    }

    if ( debug ) grad1.print ( "grad1 = " );
}

void CovMcbd::Grad2 ( arma::vec& grad2 ) {
    int debug = 1;

    int llmd = lmd_.n_elem;
    grad2 = arma::zeros<arma::vec> ( llmd );

    arma::vec oneJ = arma::ones<arma::vec> ( n_atts_ );
    arma::vec oneT = arma::ones<arma::vec> ( n_dims_ );
    arma::vec oneTJ = arma::ones<arma::vec> ( n_dims_ * n_atts_ );
    grad2 = n_dims_ * arma::kron ( oneJ, W_.t() * oneT );

    if ( debug ) grad2.print("grad2 = ");
    }


void CovMcbd::UpdateCovMcbd ( const arma::vec &x ) {
    int debug = 1;
    UpdateParam ( x );
    std::cout << "params updated..." << std::endl;
    UpdateModel();
    std::cout << "model updated..." << std::endl;
}

void CovMcbd::UpdateParam ( const arma::vec &x ) {
    int debug = 0;

    int lbta = X_.n_cols;
    int llmd = U_.n_cols * n_atts_;
    int lpsi = V_.n_cols * n_atts_ * ( n_atts_ - 1 ) / 2;
    int lgma = W_.n_cols * n_atts_;

    switch ( free_param_ ) {
    case 0:
        tht_ = x;
        bta_ = x.rows ( 0, lbta - 1 );
        lmd_ = x.rows ( lbta, lbta + llmd - 1 );
        psi_ = x.rows ( lbta + llmd, lbta + llmd + lpsi - 1 );
        gma_ = x.rows ( lbta + llmd + lpsi, lbta + llmd + lpsi + lgma -1 );

        if ( debug ) {
            bta_.t().print ( "beta = " );
            lmd_.t().print ( "lambda = " );
            psi_.t().print ( "psi = " );
            gma_.t().print ( "gamma = " );
        }

        MatLmd_ = arma::reshape ( arma::mat ( lmd_ ), U_.n_cols, n_atts_ );
        MatPsi_ = arma::reshape ( arma::mat ( psi_ ), V_.n_cols, lpsi / V_.n_cols );
        gma_vec2mat();

        if ( debug ) {
            MatLmd_.print ( "MatLmd = " );
            MatPsi_.print ( "MatPsi = " );
            MatGma_.print ( "MatGma = " );
        }

    }
}

void CovMcbd::UpdateModel() {
    int debug = 0;

    Xbta_ = X_ * bta_;
    Ulmd_ = U_ * MatLmd_;
    Vpsi_ = V_ * MatPsi_;
    Wgma_ = W_ * MatGma_;
    Resid_ = Y_ - Xbta_;

    if ( debug ) {
        Ulmd_.print ( "Ulmd = " );
        Vpsi_.print ( "Vpsi = " );
        Wgma_.print ( "Wgma = " );
    }

    Update_Sigma();
}

}
