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
## correct uni / multivariate X by applying optimal transport to             ##
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
## X to be ensured and,  more generally, to use and operate it in the        ##
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
from Apyga.stats.__SparseHistBase import SparseHistBase
from Apyga.stats.__bin_width_estimator import bin_width_estimator

###########
## Class ##
###########

class SparseHist:
	"""
	Description
	===========
	Estimate a sparse histogram from a dataset
	"""
	
	def __init__( self , X , bin_width = None , bin_origin = None , method = "auto" ):
		"""
			Initialisation of SparseHist
			
			Parameters
			----------
			X           : np.array[ shape = (n_samples,n_features) ]
				Dataset to fit the SparseHistogram
			bin_width   : None or np.array[ shape = (n_features) ]
				Lenght of cells. Each dimension of bin_width is the lenght of regular cells in each dimensions
			bin_origin  : None or np.array[ shape = (n_features) ]
				Coordinate of lower corner of one cell

			Attributes
			----------
			dim        : int
				Dimension
			size       : int
				Number of cells/masses
			p          : np.array[ shape = (size) ]
				Probability vector
			c          : np.array[ shape = (size,dim) ]
				Matrix of centers of bins
			bin_width  : np.array[ shape = (dim) ] 
				Lenght of cells. Each dimension of bin_width is the lenght of regular cells in each dimensions
			bin_origin : np.array[ shape = (dim) ]
				Coordinate of lower corner of one cell
		"""
		if bin_width is None:
			bin_width = bin_width_estimator( X , method )
		if bin_origin is None:
			bin_origin = np.zeros_like(bin_width)
		
		self._sphb = SparseHistBase( X , np.array( bin_width , dtype = np.float ) , np.array( bin_origin , dtype = np.float ) )
		self.p = self._sphb.p
		self.c = self._sphb.c
		self.size = self._sphb.size
		self.dim = self._sphb.dim
		self.bin_width = self._sphb.bin_width
		self.bin_origin = self._sphb.bin_origin

		
	def argwhere( self , X ):
		"""
			Find the cells where data are located

			Parameters
			----------
			X    : np.array[ shape = (number of X,dimension) ]
				Dataset to find index

			returns
			-------
			index   : np.array[ shape = (size) , dtype = np.int ]
				Array of index of X.
				Contains -1 for datum not in cell previously estimated
		"""
		return np.array( self._sphb.argwhere(X) , dtype = np.int )
	
	def sample( self , size = 1 , noise = None , law_noise = "uniform" ):
		"""
			Draw elements according to the law defined

			Parameters
			----------
			size    : int
				Numbers of X to draw
			noise   : None, scalar or np.array[ shape = (dimension) ]
				noise to apply
					=> None      : no noise is applied.
					=> scalar    : converted to np.array[ shape = (dimension) ], fill with noise. 
			law_noise : str
				Law of the noise
					=> "uniform" : Draw according to uniform law in cell defines by [center-noise;center+noise]
					=> "normal"  : Draw according multivariate_normal law, mean = 0, cov = np.diag(noise)
					=> other     : no noise

			Returns
			-------
			X : np.array[ shape = (size,dimension) ]
		"""
		
		ind = np.random.choice( range(self.size) , size = size , p = self.p )
		center = self.c[ind,:]
		
		if noise is not None:
			if np.isscalar(noise):
				noise = np.zeros(self.dim) + np.abs(noise)
			noise = np.abs( np.array( noise , dtype = np.float ) )
			if law_noise == "uniform":
				center += np.random.uniform( low = - noise , high = noise , size = center.shape )
			elif law_noise == "normal":
				center += np.random.multivariate_normal( mean = np.zeros(self.dim) , cov = np.diag(noise) , size = center.shape[0] )
		
		return center
	
	def cdf( self , X ):
		"""
			Computes the Cumulative Distribution Function of the given array

			Parameters
			----------
			X : np.array[ shape = (size,dimension) ]
				Data to estimate the CDF

			Returns
			CDF  : np.array[ shape = (size) ]
				CDF at given points
		"""
		sizeData = X.shape[0]
		cdf = np.zeros( sizeData )
		for i in range(sizeData):
			lexico = np.zeros( self.size , dtype = bool ) + True
			for j in range(self.dim):
				lexico = np.logical_and( lexico , self.c[:,j] <= X[i,j] )
			cdf[i] = np.sum( self.p[lexico] )
		return cdf
	
	def pdf( self , X ):
		"""
			Computes the Probability Distribution Function of the given array

			Parameters
			----------
			X : np.array[ shape = (size,dimension) ]
				Data to estimate the PDF

			Returns
			PDF  : np.array[ shape = (size) ]
				PDF at given points
		"""
		idx = np.array( self.argwhere(X) )
		idxIn = np.argwhere( idx > -1 ).ravel()
		pdf = np.zeros( X.shape[0] , dtype = np.float )
		pdf[idxIn] = self.p[idx[idxIn]]
		return pdf

	def mean( self ):
		"""
			Compute the mean of the histogram

			Returns
			-------
			mean    : np.array[ shape = (dim) ]
				Mean estimated from the histogram
		"""
		return np.average( self.c , weights = self.p , axis = 0 )

	def cov( self ):
		"""
			Compute the covariance matrix of the histogram

			Returns
			-------
			cov     : np.array[ shape = (dim,dim) ]
				Covariance matrix estimated from the histogram
		"""
		return np.cov( self.c , rowvar = False , aweights = self.p )
	
	def pearson_correlations( self ):
		"""
			Compute the Pearson correlations matrix of the histogram

			Returns
			-------
			pears   : np.array[ shape = (dim,dim) ]
				Pearson correlations matrix estimated from the histogram
		"""
		cov = self.cov()
		pears = np.zeros_like(cov) + np.identity(self.dim)
		for i in range(self.dim):
			for j in range(i+1,self.dim):
				pears[i,j] = cov[i,j] / np.sqrt( cov[i,i] * cov[j,j] )
				pears[j,i] = pears[i,j]
		return pears

