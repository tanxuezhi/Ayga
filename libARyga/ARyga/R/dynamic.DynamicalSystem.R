
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

## DynamicalSystem {{{

#' DynamicalSystem
#'
#' Base class to define Dynamical system, do not use it!!!
#'
#' @docType class
#' @importFrom R6 R6Class
#'
#' @param dim [integer]
#'        Dimension of the phase space of the dynamical system
#' @param size [integer]
#'        Number of initial condition simultaneously solved
#' @param bounds  [matrix]
#'        Bounds of phase space where initial condition can be drawn
#' @param t [vector]
#'        Time to integrate the dynamical system
#' @param X0 [vector or NULL]
#'        Vector of initial condition of size size*dim, if NULL a random IC is drawn in bounds
#'
#' @return Object of \code{\link{R6Class}}
#' @format \code{\link{R6Class}} object.
#'
#' @section Methods:
#' \describe{
#'   \item{\code{new(dim,size,bounds)}}{This method is used to create object of this class with \code{DynamicalSystem}}
#'   \item{\code{orbit(t,X0)}}{Compute the orbit along t starting at X0. If X0 is NULL, it is randomly drawn by randomIC()}
#'   \item{\code{randomIC()}}{Return a random initial condition}
#' }
#' @examples
#' ## No example because you should not use this class!
#'
#' @export
DynamicalSystem = R6::R6Class( "DynamicalSystem" ,
	
	public = list(
	
	###############
	## arguments ##
	###############
	
	dim    = 0,
	size   = 0,
	bounds = NULL,
	
	
	#################
	## Constructor ##
	#################
	
	initialize = function( dim , size = 1 , bounds = NULL )
	{
		self$dim    = dim
		self$size   = size
		self$bounds = bounds
		
		private$i = list()
		for( i in 1:self$dim )
		{
			private$i[[i]] = seq( i , self$dim * self$size , self$dim )
		}
	},
	
	
	#############
	## Methods ##
	#############
	
	randomIC = function()
	{
		return( as.vector( base::t( base::apply( self$bounds , 1 , function(X) { return( stats::runif( n = self$size , min = X[1] , max = X[2] ) ) } ) ) ) )
	},
	
	orbit = function( t , X0 = NULL )
	{
		X0 = if( is.null(X0) ) self$randomIC() else X0
		X = private$solver( t , X0 )
		if( self$size > 1 )
		{
			l = dim(X)[1]
			X = base::array( X , base::c( l , self$dim , self$size ) )
			Z = base::array( dim = base::c( self$size , self$dim , l ) )
			for( i in 1:self$size )
			{
				Z[i,,] = base::t(X[,,i])
			}
			X = Z
		}
		else
		{
			X = base::t(X)
		}
		return( X )
	}
	
	
	),
	
	private = list(
	
	###############
	## arguments ##
	###############
	
	i = NULL,
	
	
	#############
	## Methods ##
	#############
	
	solver = function( t , X0 )
	{},
	
	equation = function( t , X , par = NULL )
	{}
	
	)
)
##}}}

## Local Dimension {{{

#' Local Dimension and persistence
#'
#' Compute the local dimension and the persistence of a dataset
#'
#' @param X [matrix] A first matrix (time in column, variables in row). Point where you want the local dimension and persistance
#'
#' @param Y [matrix] A second matrix (time in column, variables in row). Point to estimate ld and theta. If Y = NULL, X is used
#'
#' @param q [float] Threshold, default = 0.98
#'
#' @param distXY [matrix] -log of pairwise distances between X and Y. If NULL, computed with ARyga::pairwise_distances( X , Y , metric = "logeuclidean" )
#'
#' @param gpd_fit [NULL or function] Function which fit the scale parameter of a gpd distribution, take a vector containing in fist index the threshold, and other values the dataset, and return the scale. If NULL, mean inverse is used.
#'
#' @return ld,theta [list] list containing local dimension and theta
#'
#' @examples
#' l63 = ARyga::Lorenz63$new()
#' t = base::seq( 0 , 100 , 0.005 )
#' Y = l63$orbit(t)
#' X = Y[,sample(1:length(t),1000)]
#' ldt = ARyga::localDimension( X , Y )
#' print(base::mean(ldt$ld))
#'
#' @export
localDimension = function( X , Y = NULL , q = 0.98 , distXY = NULL , gpd_fit = NULL )
{
	## Fit function for theta
	theta_ferro = function( Z )
	{
		iThreshold = which( Z[-1] > Z[1] )
		l = length(iThreshold)
		Ti = base::diff(iThreshold)
		return( 2 * ( base::sum(Ti - 1)^2 ) / ( (l-1) * base::sum( (Ti-1) * (Ti-2) ) ) )
	}
	
	## Fit function for local dim
	gpdfit_mean = function(Z)
	{
		return( 1. / base::mean( Z[-1][Z[-1] > Z[1]] - Z[1] ) )
	}
	
	if( is.null(gpd_fit) )
	{
		gpd_fit = gpdfit_mean
	}
	
	## Pairwise distances
	if( is.null(distXY) )
	{
		distXY = - ARyga::pairwise_distances( X , Y , metric = "logeuclidean" )
		distXY[which( is.infinite(distXY) )] = - Inf
	}
	
	## Threshold
	thresholds = base::apply( distXY , 1 , quantile , probs = q )
	
	## Fit localdim
	ld = base::apply( base::cbind( thresholds , distXY ) , 1 , gpd_fit )
#	ld = base::apply( base::cbind( thresholds , distXY ) , 1 , function( Z ) { return(1. / base::mean( Z[-1][Z[-1] > Z[1]] - Z[1] ) ) } )
	
	## Fit theta
	theta = base::apply( base::cbind( thresholds , distXY ) , 1 , theta_ferro )
	
	return( list( ld = ld , theta = theta ) )
}
##}}}


