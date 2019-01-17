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
import scipy.stats as sc
from Apyga.stats.__bin_width_estimator import bin_width_estimator


###########
## Class ##
###########

class QM:
	"""
		Apyga.stats.bc.QM
		=================

		Description
		-----------
		Quantile Mapping bias corrector
	"""
	def __init__( self , bins = None ):
		"""
			Initialisation of Quantile Mapping bias corrector.
			
			Parameters
			----------
			bins    : None or list(np.array)
				If None bins is estimated during fit, else bins must be a list of length "n_features" of edges of each bin (something like np.linspace(min,max,100))
		"""
		self.bins = bins
		self._n_features = 1
		self._rvY = None
		self._rvX = None
	
	def fit( self , Y , X ):
		"""
			Fit of the quantile mapping model
			
			Parameters
			----------
			Y	: np.array[ shape = (n_samples,n_features) ]
				Reference dataset
			X	: np.array[ shape = (n_samples,n_features) ]
				Biased dataset
		"""
		if len(X.shape) == 1:
			X = X.reshape( (X.size,1) )
		if len(Y.shape) == 1:
			Y = Y.reshape( (Y.size,1) )
		self._n_features = X.shape[1]
		
		if self.bins is None:
			self.bins = self._bin_estimator( Y , X )
		self._rvY = [sc.rv_histogram( np.histogram( Y[:,i] , self.bins[i] ) ) for i in range(self._n_features)]
		self._rvX = [sc.rv_histogram( np.histogram( X[:,i] , self.bins[i] ) ) for i in range(self._n_features)]
	
	def predict( self , X ):
		"""
			Perform the bias correction
			
			Parameters
			----------
			X	: np.array[ shape = (n_samples,n_features) ]
				Dataset to correct

			Returns
			-------
			Z	: np.array or float
				Array of same shape than X, correction of X with respect to Y
		"""
		if len(X.shape) == 1:
			X = X.reshape( (X.size,1) )
		Z = np.zeros_like(X)
		for i in range(self._n_features):
			cdfX = self._rvX[i].cdf(X[:,i])
			cdfX[cdfX > 1] = 1		## sometimes **.cdf return 1 + eps (numerical round)
			cdfX[cdfX < 0] = 0      ## sometimes **.cdf return -eps (numerical round)
			Z[:,i] = self._rvY[i].ppf(cdfX)
		return Z
	
	def _bin_estimator( self , Y , X ):
		Min = np.min( np.vstack( (Y,X) ) , axis = 0 )
		Max = np.max( np.vstack( (Y,X) ) , axis = 0 )
		bwX = bin_width_estimator(X)
		bwY = bin_width_estimator(Y)
		bw = np.min( (bwX,bwY) , axis = 0 )
		return [ np.arange( Min[i] - bw[i] , Max[i] + bw[i] , bw[i] ) for i in range(self._n_features) ]


