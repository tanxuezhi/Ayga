
##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
## Contributor : Mathieu Vrac                                                   ##
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
## Contributor : Mathieu Vrac                                                   ##
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

##################################################################################
##################################################################################
##                                                                              ##
## Original author  : Mathieu Vrac                                              ##
## Contact          : mathieu.vrac@lsce.ipsl.fr                                 ##
##                                                                              ##
## Notes   : CDFt is the re-implementation of the function CDFt of R package    ##
##           "CDFt" developped by Mathieu Vrac, available at                    ##
##           https://cran.r-project.org/web/packages/CDFt/index.html            ##
##           This code is governed by the CeCILL-C license with the             ##
##           authorization of Mathieu Vrac                                      ##
##                                                                              ##
##################################################################################
##################################################################################

###############
## Libraries ##
###############

###############
## Functions ##
###############


## CDFt Correction method {{{

#' CDFt method (Cumulative Distribution Function transfer)
#'
#' Perform an univariate bias correction of X with respect to Y (correction is applied margins by margins).
#'
#' @docType class
#' @importFrom R6 R6Class
#'
#' @param bin_width [list of vector of NULL]
#'        A vector of bin_width (length of a bin)
#'        If NULL, it is estimating during the fit
#' @param Y0  [matrix]
#'        A matrix containing references during calibration period (time in column, variables in row)
#' @param X0 [matrix]
#'        A matrix containing biased data during calibration period (time in column, variables in row)
#' @param X1 [matrix]
#'        A matrix containing biased data during projection period (time in column, variables in row)
#'
#' @return Object of \code{\link{R6Class}} with methods for bias correction
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(bins)}}{This method is used to create object of this class with \code{CDFt}}
#'   \item{\code{fit(Y0,X0,X1)}}{Fit the bias correction model from Y0 and X0 and X1}.
#'   \item{\code{predict(X1)}}{Perform the bias correction of X1 with respect to the estimaton of Y1.}.
#' }
#' @examples
#' ## Three bivariate random variables (rnorm and rexp are inverted between ref and bias)
#' X0 = matrix( NA , nrow = 2 , ncol = 10000 )
#' X1 = matrix( NA , nrow = 2 , ncol = 10000 )
#' Y0 = matrix( NA , nrow = 2 , ncol = 10000 )
#' X0[1,] = rnorm( 10000 )
#' X0[2,] = rexp( 10000 )
#' X1[1,] = rnorm( 10000 , mean = 2 )
#' X1[2,] = rexp( 10000 ) + 2
#' Y0[1,] = rexp( 10000 )
#' Y0[2,] = rnorm( 10000 )
#'
#' ## Bin length
#' bin_width = c(0.1,0.1)
#'
#' ## Bias correction
#' ## Step 1 : construction of the class CDFt 
#' cdft = ARyga::CDFt$new( bin_width ) 
#' ## Step 2 : Fit the bias correction model
#' cdft$fit( Y0 , X0 , X1 )
#' ## Step 3 : perform the bias correction, uX1 is the correction of
#' ## X1 with respect to the estimation of Y1
#' uX1 = cdft$predict(X1) 
#'
#' @export
CDFt = R6::R6Class( "CDFt" ,
	
	inherit = ARyga::AbstractBiasCorrectionMethod,
	
	public = list(
	
	###############
	## Arguments ##
	###############
	
	n_features = 0,
	bins = list(),
	bin_width = NULL,
	dev = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function( bin_width = NULL )
	{
		super$initialize()
		self$bin_width = bin_width
	},
	
	fit = function( Y0 , X0 , X1 )
	{
		## Dimension and data formating
		if( class(Y0) == "numeric" )
		{
			Y0 = matrix( Y0 , nrow = 1 , ncol = length(Y0) )
		}
		if( class(X0) == "numeric" )
		{
			X0 = matrix( X0 , nrow = 1 , ncol = length(X0) )
		}
		if( class(X1) == "numeric" )
		{
			X1 = matrix( X1 , nrow = 1 , ncol = length(X1) )
		}
		self$n_features = dim(X0)[1]
		
		## Bins
		if( is.null(self$bin_width) )
		{
			bwY0 = ARyga::binwidth_estimator(Y0)
			bwX0 = ARyga::binwidth_estimator(X0)
			bwX1 = ARyga::binwidth_estimator(X1)
			self$bin_width = base::pmin( bwY0 , bwX0 , bwX1 )
		}
		
		## Random variable
		private$diff = base::rep( NA , self$n_features )
		self$dev = base::rep( 1 , self$n_features )
		for( i in 1:self$n_features )
		{
			private$cdft_fit( Y0[i,] , X0[i,] , X1[i,] , i )
		}
	},
	
	predict = function( X1 )
	{
		if( class(X1) == "numeric" )
		{
			X1 = matrix( X1 , nrow = 1 , ncol = length(X1) )
		}
		
		Z1 = matrix( NA , nrow = self$n_features , ncol = dim(X1)[2] )
		
		for( i in 1:self$n_features )
		{
			Z1[i,] = private$qmX1Y1[[i]]$predict( X1[i,] + private$diff[i] )
		}
		return(Z1)
	}
	
	),
	
	
	######################
	## Private elements ##
	######################
	
	private = list(
	
	###############
	## Arguments ##
	###############
	
	qmX1Y1 = list(),
	diff = NULL,
	
	
	#############
	## Methods ##
	#############
	
	cdft_fit = function( Y0 , X0 , X1 , idx )
	{
		## Center data on mean(Y0)
		private$diff[idx] = base::mean(Y0) - base::mean(X0)
		X0c = X0 + private$diff[idx]
		X1c = X1 + private$diff[idx]
		
		## Some parameters
		m = base::abs( base::mean(X1) - base::mean(X0) )
		s = stats::sd(X1) / stats::sd(X0)
		s = if( s < 1 ) 1 else s
		
		## Global loop
		dev_ok = FALSE
		cdfY1 = NULL
		while( !dev_ok )
		{
			## Bins
			Min = base::min(Y0,X0,X1,X0c,X1c) - m * s * self$dev[idx]
			Max = base::max(Y0,X0,X1,X0c,X1c) + m * s * self$dev[idx]
			self$bins[[idx]] = base::seq( Min - self$bin_width[idx] , Max + self$bin_width[idx] , self$bin_width[idx] )
			
			## Random variable
			rvY0 = ARyga::rv_histogram$new( Y0  , self$bins[[idx]] )
			rvX0 = ARyga::rv_histogram$new( X0c , self$bins[[idx]] )
			rvX1 = ARyga::rv_histogram$new( X1c , self$bins[[idx]] )
			
			## First estimation
			cdfY0 = rvY0$cdf( self$bins[[idx]] )
			cdfY1 = rvY0$cdf( rvX0$icdf( rvX1$cdf( self$bins[[idx]] ) ) )
			
			## Correction of left part
			if( base::min(Y0) < base::min(X1c) )
			{
				Y0q = stats::quantile( Y0 , probs = cdfY1[1] , names = FALSE )
				i = base::max( base::which( self$bins[[idx]] <= Y0q ) )
				j = base::max( base::which( self$bins[[idx]] < base::min(X1c) ) )
				if( i < j )
				{
					cdfY1[(j-i+1):j] = cdfY0[1:i]
					cdfY1[1:(j-i+1)] = 0.
				}
				else
				{
					cdfY1[1:j] = cdfY0[(i-j+1):i]
				}
			}
			
			## Correction of right part
			kk = base::which( cdfY1 == 1. )
			k = if( length(kk) > 0 ) base::min(kk) else length(cdfY1)
			if( (base::abs(cdfY1[k] - cdfY1[k-1]) > 1. / length(self$bins[[idx]]) ) || k == length(cdfY1) )
			{
				Y0q = stats::quantile( Y0 , probs = cdfY1[k-1] , names = FALSE )
				i = base::min( base::which( Y0q <= self$bins[[idx]] ) )
				j = base::min( base::which( cdfY1 == cdfY1[k] ) )
				
				if( j > 0 )
				{
					diff = length(self$bins[[idx]]) - base::max(i,j)
					cdfY1[j:(j+diff)] = cdfY0[i:(i+diff)]
					if( j + diff < length(self$bins[[idx]]) )
					{
						cdfY1[(j+diff):length(cdfY1)] = 1.
					}
					dev_ok = TRUE
				}
				else
				{
					self$dev[idx] = 2 * self$dev[idx]
				}
			}
			else
			{
				dev_ok = TRUE
			}
		}
		## Final estimator of Y1
		pY1 = base::diff(cdfY1)
		size = length(self$bins[[idx]])
		cY1 = (self$bins[[idx]][2:size] + self$bins[[idx]][1:(size-1)]) / 2.
		Y1 = cY1[ base::sample( length(cY1) , length(X1) , replace = TRUE , prob = pY1 ) ]
		
		## Quantile Mapping corrector
		private$qmX1Y1[[idx]] = ARyga::QM$new( list(self$bins[[idx]]) )
		private$qmX1Y1[[idx]]$fit( Y1 , X1c )
	}
	
	)
)
##}}}

