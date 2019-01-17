
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program that is part of the ARyga library. This  ##
## library makes it possible to study dynamic systems and to statistically      ##
## correct uni / multivariate data by applying optimal transport to             ##
## sparse histograms.                                                           ##
##                                                                              ##
## This software is governed by the CeCILL-C license under French law and       ##
## abiding by the rules of distribution of free software.  You can  use,        ##
## modify and/ or redistribute the software under the terms of the CeCILL-C     ##
## license as circulated by CEA, CNRS and INRIA at the following URL            ##
## "http://www.cecill.info".                                                    ##
##                                                                              ##
## As a counterpart to the access to the source code and  rights to copy,       ##
## modify and redistribute granted by the license, users are provided only      ##
## with a limited warranty  and the software's author,  the holder of the       ##
## economic rights,  and the successive licensors  have only  limited           ##
## liability.                                                                   ##
##                                                                              ##
## In this respect, the user's attention is drawn to the risks associated       ##
## with loading,  using,  modifying and/or developing or reproducing the        ##
## software by the user in light of its specific status of free software,       ##
## that may mean  that it is complicated to manipulate,  and  that  also        ##
## therefore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are therefore        ##
## encouraged to load and test the software's suitability as regards their      ##
## requirements in conditions enabling the security of their systems and/or     ##
## data to be ensured and,  more generally, to use and operate it in the        ##
## same conditions as regards security.                                         ##
##                                                                              ##
## The fact that you are presently reading this means that you have had         ##
## knowledge of the CeCILL-C license and that you accept its terms.             ##
##                                                                              ##
##################################################################################
##################################################################################

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique faisant partie de la librairie     ##
## ARyga. Cette librairie permet d'étudier les systèmes dynamique et de         ##
## corriger statistiquement des données en uni/multivarié en appliquant le      ##
## transport optimal à des histogrammes creux.                                  ##
##                                                                              ##
## Ce logiciel est régi par la licence CeCILL-C soumise au droit français et    ##
## respectant les principes de diffusion des logiciels libres. Vous pouvez      ##
## utiliser, modifier et/ou redistribuer ce programme sous les conditions       ##
## de la licence CeCILL-C telle que diffusée par le CEA, le CNRS et l'INRIA     ##
## sur le site "http://www.cecill.info".                                        ##
##                                                                              ##
## En contrepartie de l'accessibilité au code source et des droits de copie,    ##
## de modification et de redistribution accordés par cette licence, il n'est    ##
## offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,    ##
## seule une responsabilité restreinte pèse sur l'auteur du programme, le       ##
## titulaire des droits patrimoniaux et les concédants successifs.              ##
##                                                                              ##
## A cet égard  l'attention de l'utilisateur est attirée sur les risques        ##
## associés au chargement,  à l'utilisation,  à la modification et/ou au        ##
## développement et à la reproduction du logiciel par l'utilisateur étant       ##
## donné sa spécificité de logiciel libre, qui peut le rendre complexe à        ##
## manipuler et qui le réserve donc à des développeurs et des professionnels    ##
## avertis possédant  des  connaissances  informatiques approfondies.  Les      ##
## utilisateurs sont donc invités à charger  et  tester  l'adéquation  du       ##
## logiciel à leurs besoins dans des conditions permettant d'assurer la         ##
## sécurité de leurs systèmes et ou de leurs données et, plus généralement,     ##
## à l'utiliser et l'exploiter dans les mêmes conditions de sécurité.           ##
##                                                                              ##
## Le fait que vous puissiez accéder à cet en-tête signifie que vous avez       ##
## pris connaissance de la licence CeCILL-C, et que vous en avez accepté les    ##
## termes.                                                                      ##
##                                                                              ##
##################################################################################
##################################################################################


###############
## Libraries ##
###############

###############
## Functions ##
###############

## OTC Optimal Transport Correction method {{{

#' OTC (Optimal Transport Correction) method
#'
#' Perform a multivariate bias correction of X with respect to Y (joint distribution, i.e. all dependence are corrected).
#'
#' @docType class
#' @importFrom R6 R6Class
#'
#' @param bin_width [vector of NULL]
#'        A vector of lengths of the cells discretizing R^{numbers of variables}.
#'        If NULL, it is estimating during the fit
#' @param bin_origin [vector of NULL]
#'        Coordinate of lower corner of one cell. If NULL, c(0,...,0) is used
#' @param Y  [matrix]
#'        A matrix containing references (time in column, variables in row)
#' @param X [matrix]
#'        A matrix containing biased data (time in column, variables in row)
#'
#' @return Object of \code{\link{R6Class}} with methods for bias correction
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(bin_width,bin_origin)}}{This method is used to create object of this class with \code{OTC}}
#'   \item{\code{fit(Y,X)}}{Fit the bias correction model from Y and X}.
#'   \item{\code{predict(X)}}{Perform the bias correction of X with respect to Y.}.
#' }
#' @examples
#' ## Three bivariate random variables (rnorm and rexp are inverted between ref and bias)
#' Y =  matrix( ncol = 10000 , nrow = 2 )
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y[1,] = rnorm(10000)
#' Y[2,] = rexp(10000)
#' X[1,] = rexp(10000)
#' X[2,] = rnorm(10000)
#'
#' ## Bin length
#' bin_width = c(0.2,0.2)
#'
#' ## Bias correction
#' ## Step 1 : construction of the class OTC 
#' otc = ARyga::OTC$new( bin_width ) 
#' ## Step 2 : Fit the bias correction model
#' otc$fit( Y , X )
#' ## Step 3 : perform the bias correction, uX is the correction of
#' ## X with respect to the estimation of Y
#' uX = otc$predict(X) 
#'
#' @importFrom methods new
#' @importFrom R6 R6Class
#' @export
OTC = R6::R6Class( "OTC" ,
	
	inherit = AbstractBiasCorrectionMethod,
	
	public = list(
	
	
	###############
	## Arguments ##
	###############
	
	bin_width = NULL,
	bin_origin = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function( bin_width = NULL , bin_origin = NULL )
	{
		super$initialize()
		self$bin_width = bin_width
		self$bin_origin = bin_origin
	},
	
	fit = function( Y , X )
	{
		if( is.null(self$bin_width) )
		{
			bwX = ARyga::binwidth_estimator(X)
			bwY = ARyga::binwidth_estimator(Y)
			self$bin_width = base::pmin( bwX , bwY )
		}
		if( is.null(self$bin_origin) )
		{
			self$bin_origin = base::rep( 0. , length(self$bin_width) )
		}
		
		private$otc = new( OTCClass , self$bin_width , self$bin_origin )
		private$otc$fit( Y , X )
	},
	
	predict = function( X )
	{
		return(private$otc$predict(X))
	}
	
	),
	
	
	######################
	## Private elements ##
	######################
	
	private = list(
	
	###############
	## Arguments ##
	###############
	
	otc = NULL
	)
)
##}}}


