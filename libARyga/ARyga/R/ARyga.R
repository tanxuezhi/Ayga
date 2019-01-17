
#' ARyga: Optimal transport and Bias correction
#'
#' This library makes it possible to study dynamic systems and to statistically
#' correct uni / multivariate data by applying optimal transport to sparse histograms.                                                         
#'
#' @details  This library is centered on multivariate bias correction and optimal transport.
#' An important point is the following: almost all methods have the syntax of
#' a R list, but are a c++ class, wrapped by Rcpp.
#' The key function is the ARyga::SparseHist class, estimating an histogram
#' of any dataset of size N and dimension d with a maximal 
#' complexity of O(dN log(N) ).
#' Next, the solver of optimal transport, coming from the library POT, gives
#' the optimal plan and the cost, also called Wasserstein distance. Other distances
#' between ARyga::SparseHist are also implemented.
#' Finally, OTC (resp. dOTC) methods are given, performing a multivariate and 
#' stationary (resp. non-stationary) bias correction.
#' Note: Some functions are prefixed by "DNU", which means DO NOT USE. These functions are internal
#' functions of ARyga, and not does not be used by users.
#' @docType package
#' @author Yoann Robin Maintainer: Yoann Robin <yoann.robin.k@gmail.com>, Soulivanh Thao <soulivanh.thao@lsce.ipsl.fr
#' @references POT library (Python Optimal Transport) https://github.com/rflamary/POT
#' @import Rcpp
#' @importFrom Rcpp evalCpp sourceCpp
#' @useDynLib ARyga
#'
#' @export
NULL
