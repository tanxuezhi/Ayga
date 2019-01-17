
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

## Minkowski {{{

#' Minkowski distance between SparseHist
#'
#' Compute the p-Minkowski distance between SparseHist
#'
#' @param muX  [ARyga::SparseHist]
#'
#' @param muY  [ARyga::SparseHist]
#'
#' @param p [double] power of the distances. p = 2 is the Euclidean distance.
#'
#' @return [double] The distance
#'
#' @examples
#' ## Two bivariate random variables
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y = matrix( ncol = 10000 , nrow = 2 )
#' X[1,] = rnorm(10000)
#' X[2,] = rnorm(10000)
#' Y[1,] = rnorm(10000 , mean = 10 )
#' Y[2,] = rnorm(10000 , mean = 10 )
#'
#' ## In measures
#' bin_width = c(0.2,0.2)
#'	muX = ARyga::SparseHist( X , bin_width )
#'	muY = ARyga::SparseHist( Y , bin_width )
#'
#' ## Distance
#' dist = ARyga::dist.minkowski( muX , muY , 1 ) ## Manhattan distance
#' dist = ARyga::dist.minkowski( muX , muY , 2 ) ## Euclidean distance
#'
#' @export
dist.minkowski = function( muX , muY , p )
{
	pX = muX$p()
	pY = muY$p()

	dist = 0
	indx = muY$argwhere(muX$c())
	indy = muX$argwhere(muY$c())
	
	## Common elements of muX in muY
	ii = which( indx > -1 )
	dist = dist + sum( abs( pX[ii] - pY[ii] )^p )
	
	## Elements of muX not in muY
	dist = dist + sum( abs( pX[which(indx == -1)] )^p )

	## Elements of muY not in muX
	dist = dist + sum( abs( pY[which(indy == -1)] )^p )

	invisible( dist^(1./p) )
}
##}}}

## Euclidean {{{

#' Euclidean distance between SparseHist
#'
#' Compute the Euclidean distance between SparseHist
#'
#' @param muX  [ARyga::SparseHist]
#'
#' @param muY  [ARyga::SparseHist]
#'
#' @return [double] The distance
#'
#' @examples
#' ## Two bivariate random variables
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y = matrix( ncol = 10000 , nrow = 2 )
#' X[1,] = rnorm(10000)
#' X[2,] = rnorm(10000)
#' Y[1,] = rnorm(10000 , mean = 10 )
#' Y[2,] = rnorm(10000 , mean = 10 )
#'
#' ## In measures
#' bin_width = c(0.2,0.2)
#'	muX = ARyga::SparseHist( X , bin_width )
#'	muY = ARyga::SparseHist( Y , bin_width )
#'
#' ## Distance
#' dist = ARyga::dist.euclidean( muX , muY ) ## Just call ARyga::dist.minkowski( muX , muY , 2 )
#'
#' @export
dist.euclidean = function( muX , muY )
{
	invisible( ARyga::dist.minkowski( muX , muY , 2 ) )
}
##}}}

## Chebyshev {{{

#' Chebyshev distance between SparseHist
#'
#' Compute the Chebyshev distance between SparseHist
#'
#' @param muX  [ARyga::SparseHist]
#'
#' @param muY  [ARyga::SparseHist]
#'
#' @return [double] The distance
#'
#' @examples
#' ## Two bivariate random variables
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y = matrix( ncol = 10000 , nrow = 2 )
#' X[1,] = rnorm(10000)
#' X[2,] = rnorm(10000)
#' Y[1,] = rnorm(10000 , mean = 10 )
#' Y[2,] = rnorm(10000 , mean = 10 )
#'
#' ## In measures
#' bin_width = c(0.2,0.2)
#'	muX = ARyga::SparseHist( X , bin_width )
#'	muY = ARyga::SparseHist( Y , bin_width )
#'
#' ## Distance
#' dist = ARyga::dist.chebyshev( muX , muY )
#'
#' @export
dist.chebyshev = function( muX , muY )
{
	pX = muX$p()
	pY = muY$p()

	dist = 0
	indx = muY$argwhere(muX$c())
	indy = muX$argwhere(muY$c())
	
	## Common elements of muX in muY
	g = which( indx > -1 )
	for( i in g)
	{
		dist = max( dist , abs( pX[i] - pY[indx[i]] ) )
	}

	## Elements of muX not in muY
	g = which(indx == -1)
	for( i in g)
	{
		dist = max( dist , pX[i] )
	}

	## Elements of muY not in muX
	g = which(indy == -1)
	for( i in g)
	{
		dist = max( dist , pY[i] )
	}

	invisible( dist )
}
##}}}

## Wasserstein {{{

#' Wasserstein distance between SparseHist
#'
#' Compute the Wasserstein distance between SparseHist
#'
#' @param muX  [ARyga::SparseHist]
#'
#' @param muY  [ARyga::SparseHist]
#'
#' @param p [double] power of the distances.
#'
#' @return [double] The distance
#'
#' @examples
#' ## Two bivariate random variables
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y = matrix( ncol = 10000 , nrow = 2 )
#' X[1,] = rnorm(10000)
#' X[2,] = rnorm(10000)
#' Y[1,] = rnorm(10000 , mean = 10 )
#' Y[2,] = rnorm(10000 , mean = 10 )
#'
#' ## In measures
#' bin_width = c(0.2,0.2)
#'	muX = ARyga::SparseHist( X , bin_width )
#'	muY = ARyga::SparseHist( Y , bin_width )
#'
#' ## Distance
#' dist = ARyga::dist.wasserstein( muX , muY )
#'
#' @export
dist.wasserstein = function( muX , muY , p )
{
	otPlan = ARyga::TrPlan( muX , muY , p = p )
	invisible(otPlan$cost())
}
##}}}

## Energy {{{

#' Energy distance between SparseHist
#'
#' Compute the Energy distance between SparseHist
#'
#' @param muX  [ARyga::SparseHist]
#'
#' @param muY  [ARyga::SparseHist]
#'
#' @param p [double] power of the distances.
#'
#' @return [double] The distance
#'
#' @examples
#' ## Two bivariate random variables
#' X = matrix( ncol = 10000 , nrow = 2 )
#' Y = matrix( ncol = 10000 , nrow = 2 )
#' X[1,] = rnorm(10000)
#' X[2,] = rnorm(10000)
#' Y[1,] = rnorm(10000 , mean = 10 )
#' Y[2,] = rnorm(10000 , mean = 10 )
#'
#' ## In measures
#' bin_width = c(0.2,0.2)
#'	muX = ARyga::SparseHist( X , bin_width )
#'	muY = ARyga::SparseHist( Y , bin_width )
#'
#' ## Distance
#' dist = ARyga::dist.energy( muX , muY )
#'
#' @export
dist.energy = function( muX , muY , p = 2 )
{
	distXY = ARyga::pairwise_distances( muX$c() , muY$c() )^p
	distXX = ARyga::pairwise_distances( muX$c() )^p
	distYY = ARyga::pairwise_distances( muY$c() )^p
	
	pX = muX$p()
	pY = muY$p()
	XY = distXY * ( pX %*% t(pY) )
	XX = distXX * ( pX %*% t(pX) )
	YY = distYY * ( pY %*% t(pY) )
	invisible( ( 2 * sum(XY) - sum(XX) - sum(YY) )^(1./p) )
}

## }}}

