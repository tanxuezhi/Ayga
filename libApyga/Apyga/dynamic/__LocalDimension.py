# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program that is part of the Apyga library. This  ##
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
## Apyga. Cette librairie permet d'étudier les systèmes dynamique et de         ##
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

import numpy as np
import sys
import scipy.stats as sc
import scipy.spatial.distance as ssd
import sklearn.metrics.pairwise as skmp
import multiprocessing as mp


###########
## Class ##
###########

def _genpareto_fit( distThX , pareto_fit ):
	if pareto_fit == "mean":
		return np.mean( distThX )
	else:
		return sc.genpareto.fit( distThX , floc = 0 )[2]

## Nicholas Moloney code, original name is extremal_sueveges
def _theta_sueveges_fit( iThreshold , q ):
	Nc = np.count_nonzero( (iThreshold[1:] - iThreshold[:-1] - 1) > 0 )
	N = iThreshold.size - 1
	tmp = ( 1.0 - q ) * ( iThreshold[-1] - iThreshold[0] )
	return ( tmp + N + Nc - np.sqrt( np.power( tmp + N + Nc , 2. ) - 8. * Nc * tmp ) ) / ( 2. * tmp )

def _theta_ferro_fit( iThreshold ):
	Ti = iThreshold[1:] - iThreshold[:-1]
	return 2 * ( np.sum(Ti - 1)**2 ) / ( Ti.size * np.sum( (Ti-1) * (Ti-2) ) )

def _theta_fit( iThreshold , q , theta_fit ):
	if theta_fit == "sueveges":
		return _theta_sueveges_fit( iThreshold , q )
	else:
		return _theta_ferro_fit( iThreshold )

def localDimension_fit( queue , distXY , q , pareto_fit , theta_fit ):
	threshold = np.percentile( distXY , 100 * q , axis = 1 )
	size = threshold.size

	localDim = np.zeros_like( threshold )
	theta = np.zeros_like( threshold )
	
	for i in range(size):
		iThreshold = np.array( np.argwhere( distXY[i,:] > threshold[i] ) ).ravel()
		localDim[i] = 1. / _genpareto_fit( distXY[i,iThreshold] - threshold[i] , pareto_fit )
		theta[i] = _theta_fit( iThreshold , q , theta_fit )
	
	queue[0].put( localDim )
	queue[1].put( theta )


def localDimension( X , Y = None , metric = "euclidean" , q = 0.98 , n_jobs = 1 , pareto_fit = "mean" , theta_fit = "ferro" , distXY = None ):
	"""
		Apyga.dynamic.localDimension
		============================

		Description
		-----------
		Fit a dataset to find its local dimension and persistence index
		
		Parameters
		----------
		X          : np.array[ shape = (n_sample,dimension) ]
			Dataset to fit
		Y          : np.array[ shape = (n_sample_2,dimension) ] or None
			Reference to estimate local dimension. If Y is None, Y = X
		metric     : str or callable = "euclidean"
			Metric used between sample of X and Y, see sklearn.metrics.pairwise.pairwise_distances
		q          : float = 0.98
			Quantile used to find the threshold for generalized pareto distribution
		n_jobs     : int = 1
			Number of CPU available.
		pareto_fit : str = "mean"
			Method to fit the scale of generalized pareto law. If "mean", the mean of the extreme is used. Else, scipy.stats.genpareto.fit is used.
		theta_fit  : str = "ferro"
			Method to fit the theta. "ferro" or "sueveges".
		distXY     : None or np.array[ shape = (n_sample,n_sample) ]
			Pairwise distance between X and Y, if None, sklearn.metrics.pairwise.pairwise_distances is called
		
		Returns
		-------
		LocalDim   : np.array[ shape = (n_sample) ]
			Local dimension of elements of X
		theta      : np.array[ shape = (n_sample) ]
			Persistence (also called extremal index) of elements of X
	"""

	## Distances
	if distXY is None:
		distXY = skmp.pairwise_distances( X , Y = Y , metric = metric , n_jobs = n_jobs )
	distXY[distXY == 0] = sys.float_info.max
	distXY = - np.log( distXY )
	
	## Parameters for parallel fit
	queue = []
	size,_ = distXY.shape
	idx = np.array_split( range(size) , n_jobs )

	## Fit
	if n_jobs > 1:
		lTh = []
		for i in range(n_jobs):
			queue.append( [mp.Queue(),mp.Queue()] )
			lTh.append( mp.Process( target = localDimension_fit , args = ( queue[-1] , distXY[idx[i]] , q , pareto_fit , theta_fit ) ) )
			lTh[-1].start()
		
		for th in lTh:
			th.join()
	else:
		queue.append( [mp.Queue(),mp.Queue()] )
		localDimension_fit( queue[0] , distXY , q , pareto_fit , theta_fit )

	## Extract results from threads
	localDim = np.zeros(size)
	theta = np.zeros(size)
	for i in range(n_jobs):
		localDim[idx[i]] = queue[i][0].get()
		theta[idx[i]] = queue[i][1].get()
	
	return localDim,theta

