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
cimport numpy as np
import cython
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc


#####################
## cpp declaration ##
#####################

cdef extern from "__SparseHistCpp.hpp":
	cdef cppclass SparseHistCpp:
		SparseHistCpp() except +
		SparseHistCpp( vector[vector[double]]& , vector[double]& , vector[double]& ) except +
		int dim()
		int size()
		vector[double] bin_width()
		vector[double] bin_origin()
		vector[double]& p()
		vector[vector[double]]& c()
		vector[int] argwhere( vector[vector[double]] )


####################
## Cython wrapper ##
####################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SparseHistBase:

	cdef SparseHistCpp sparseHist

	
	def __cinit__( self , X , bin_width , bin_origin ):
		self.sparseHist = SparseHistCpp( X , bin_width , bin_origin )
	
	
	@property
	def bin_width(self):
		return self.sparseHist.bin_width()
	
	@property
	def bin_origin(self):
		return self.sparseHist.bin_origin()

	@property
	def dim(self):
		return self.sparseHist.dim()

	@property
	def size(self):
		return self.sparseHist.size()
	
	@property
	def p(self):
		return np.array(self.sparseHist.p())

	@property
	def c(self):
		return np.array(self.sparseHist.c())
	
	
	def argwhere( self , X ):
		return self.sparseHist.argwhere(X)
	
#	def sample( self , size = 1 , noise = None , law_noise = "uniform" ):
#		"""
#			Draw elements according to the law defined
#
#			Parameters
#			----------
#			size    : int
#				Numbers of data to draw
#			noise   : None, scalar or np.array[ shape = (dimension) ]
#				noise to apply
#					=> None      : no noise is applied.
#					=> scalar    : converted to np.array[ shape = (dimension) ], fill with noise. 
#			law_noise : str
#				Law of the noise
#					=> "uniform" : Draw according to uniform law in cell defines by [center-noise;center+noise]
#					=> "normal"  : Draw according multivariate_normal law, mean = 0, cov = np.diag(noise)
#					=> other     : no noise
#
#			Returns
#			-------
#			dataset : np.array[ shape = (size,dimension) ]
#		"""
#		
#		ind = np.random.choice( range(self.size) , size = size , p = self.p )
#		center = self.c[ind,:]
#		
#		if noise is not None:
#			if np.isscalar(noise):
#				noise = np.zeros(self.dim) + np.abs(noise)
#			noise = np.abs( np.array( noise , dtype = np.float ) )
#			if law_noise == "uniform":
#				center += np.random.uniform( low = - noise , high = noise , size = center.shape )
#			elif law_noise == "normal":
#				center += np.random.multivariate_normal( mean = np.zeros(self.dim) , cov = np.diag(noise) , size = center.shape[0] )
#		
#		return center
#	
#	def cdf( self , data ):
#		"""
#			Computes the Cumulative Distribution Function of the given array
#
#			Parameters
#			----------
#			data : np.array[ shape = (size,dimension) ]
#				Data to estimate the CDF
#
#			Returns
#			CDF  : np.array[ shape = (size) ]
#				CDF at given points
#		"""
#		sizeData = data.shape[0]
#		cdf = np.zeros( sizeData )
#		c = self.c
#		p = self.p
#		cdef int i = 0
#		cdef int j = 0
#		for i in range(sizeData):
#			lexico = np.zeros( self.size , dtype = bool ) + True
#			for j in range(self.dim):
#				lexico = np.logical_and( lexico , c[:,j] <= data[i,j] )
#			cdf[i] = np.sum( p[lexico] )
#		return cdf
#	
#	def pdf( self , data ):
#		"""
#			Computes the Probability Distribution Function of the given array
#
#			Parameters
#			----------
#			data : np.array[ shape = (size,dimension) ]
#				Data to estimate the PDF
#
#			Returns
#			PDF  : np.array[ shape = (size) ]
#				PDF at given points
#		"""
#		p = self.p
#		idx = np.array( self.argwhere(data) )
#		idxIn = np.argwhere( idx > -1 ).ravel()
#		pdf = np.zeros( data.shape[0] , dtype = np.float )
#		pdf[idxIn] = p[idx[idxIn]]
#		return pdf
#
#	def mean( self ):
#		"""
#			Compute the mean of the histogram
#
#			Returns
#			-------
#			mean    : np.array[ shape = (dim) ]
#				Mean estimated from the histogram
#		"""
#		c = self.c
#		p = self.p
#		return np.average( c , weights = p , axis = 0 )
#
#	def cov( self ):
#		"""
#			Compute the covariance matrix of the histogram
#
#			Returns
#			-------
#			cov     : np.array[ shape = (dim,dim) ]
#				Covariance matrix estimated from the histogram
#		"""
#		c = self.c
#		p = self.p
#		return np.cov( c , rowvar = False , aweights = p )
#	
#	def pearson_correlations( self ):
#		"""
#			Compute the Pearson correlations matrix of the histogram
#
#			Returns
#			-------
#			pears   : np.array[ shape = (dim,dim) ]
#				Pearson correlations matrix estimated from the histogram
#		"""
#		cov = self.cov()
#		pears = np.zeros_like(cov) + np.identity(self.dim)
#		for i in range(self.dim):
#			for j in range(i+1,self.dim):
#				pears[i,j] = cov[i,j] / np.sqrt( cov[i,i] * cov[j,j] )
#				pears[j,i] = pears[i,j]
#		return pears



