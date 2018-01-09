#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .Call calls */
extern SEXP _cmmr_mcbd_estimation(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mcbd__new(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mcbd__get_m(SEXP, SEXP);
extern SEXP mcbd__get_Y(SEXP, SEXP);
extern SEXP mcbd__get_X(SEXP, SEXP);
extern SEXP mcbd__get_U(SEXP, SEXP);
extern SEXP mcbd__get_V(SEXP, SEXP);
extern SEXP mcbd__get_W(SEXP, SEXP);
extern SEXP mcbd__get_D(SEXP, SEXP, SEXP);
extern SEXP mcbd__get_T(SEXP, SEXP, SEXP);
extern SEXP mcbd__get_mu(SEXP, SEXP, SEXP);
extern SEXP mcbd__get_Sigma(SEXP, SEXP, SEXP);
//extern SEXP mcbd__get_fim(SEXP, SEXP);
//extern SEXP mcbd__get_sd(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
	{"_cmmr_mcbd_estimation", (DL_FUNC) &_cmmr_mcbd_estimation,  9},
    {"mcbd__new",        (DL_FUNC) &mcbd__new,         7},
    {"mcbd__get_m",      (DL_FUNC) &mcbd__get_m,       2},
    {"mcbd__get_Y",      (DL_FUNC) &mcbd__get_Y,       2},
    {"mcbd__get_X",      (DL_FUNC) &mcbd__get_X,       2},
    {"mcbd__get_U",      (DL_FUNC) &mcbd__get_U,       2},
    {"mcbd__get_V",      (DL_FUNC) &mcbd__get_V,       2},
    {"mcbd__get_W",      (DL_FUNC) &mcbd__get_W,       2},
    {"mcbd__get_D",      (DL_FUNC) &mcbd__get_D,       3},
    {"mcbd__get_T",      (DL_FUNC) &mcbd__get_T,       3},
    {"mcbd__get_mu",     (DL_FUNC) &mcbd__get_mu,      3},
    {"mcbd__get_Sigma",  (DL_FUNC) &mcbd__get_Sigma,   3},
    //{"mcbd__get_fim",    (DL_FUNC) &mcbd__get_fim,     2},
    //{"mcbd__get_sd",     (DL_FUNC) &mcbd__get_sd,      2},
    {NULL, NULL, 0}
};

void R_init_cmmr(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}