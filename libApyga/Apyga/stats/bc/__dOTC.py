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
## theYore means  that it is reserved for developers  and  experienced        ##
## professionals having in-depth computer knowledge. Users are theYore        ##
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
from Apyga.stats import SparseHist
from Apyga.stats import bin_width_estimator
from Apyga.stats.bc import OTC


###########
## Class ##
###########

class dOTC:
	"""
		Apyga.stats.bc.dOTC
		===================

		Description
		-----------
		Optimal Transport bias Corrector, taking account of an evolution of the distribution
	"""
	
	def _eps_cholesky( self , M , nit = 200 ): #{{{
		MC = None
		try:
			MC = np.linalg.cholesky(M)
		except:
			MC = None
		eps = 0
		if MC is None:
			eps = min( 1e-9 , np.abs(np.diagonal(M)).min() )
			if eps >= 0 and eps <= 0:
				eps = 1e-9
			it = 0
			while MC is None and it < nit:
				perturb = np.identity( M.shape[0] ) * eps
				try:
					MC = np.linalg.cholesky( M + perturb )
				except:
					MC = None
				eps = 2 * eps
				nit += 1
		return MC
	#}}}
	
	
	def __init__( self , bin_width = None , bin_origin = None , cov_factor = "std" , p00 = 2 , p01 = 2 ):
		"""
			Initialisation of Optimal Transport bias Corrector.
			
			Parameters
			----------
			bin_width  : np.array[ shape = (n_features) ] or None
				Lenght of bins, see Apyga.stats.SparseHist. If None, bin_width is estimated during fit.
			bin_origin : np.array[ shape = (n_features) ] or None
				Corner of one bin, see Apyga.stats.SparseHist. If None, np.repeat( 0 , n_features ) is used.
			cov_factor : str or np.array[ shape = (n_features,n_features) ]
				Correction factor during transfer of the evolution between X0 and X1 to Y0
					"cholesky" => compute the cholesky factor
					"std"      => compute the standard deviation factor
					other str  => identity is used
			p00        : float
				Power of the transport plan between biased data and references. Default = 2
			p01        : float
				Power of the transport plan between calibration and projection period. Default = 2
				
			
			Attributes
			----------
			planX1Y1   : Apyga.stats.bc.OTC
				OTC corrector between X1 and the estimation of Y1
		"""
		self._cov_factor_str = cov_factor
		self._cov_factor = None if type(cov_factor) == str else cov_factor
		self._bin_width  = bin_width
		self._bin_origin = bin_origin
		self.p00 = p00
		self.p01 = p01
	
	
	def fit( self , Y0 , X0 , X1 ):
		"""
			Fit the dOTC model to perform non-stationary bias correction during period 1. For period 0, see OTC
			
			Parameters
			----------
			Y0 : np.array[ shape = (n_samples,n_features) ]
				Reference dataset during period 0
			X0 : np.array[ shape = (n_samples,n_features) ]
				Biased dataset during period 0
			X1	: np.array[ shape = (n_samples,n_features) ]
				Biased dataset during period 1
		"""
		## Set the covariance factor correction
		if self._cov_factor is None:
			if self._cov_factor_str in ["std" , "cholesky"]:
				if Y0.shape[1] == 1:
					try:
						self._cov_factor = np.std( Y0 ) / np.std( X0 )
					except:
						self._cov_factor = 1
				elif self._cov_factor_str == "cholesky":
					fact0 = self._eps_cholesky( np.cov( Y0 , rowvar = False ) )
					fact1 = self._eps_cholesky( np.cov( X0 , rowvar = False ) )
					self._cov_factor = np.dot( fact0 , np.linalg.inv( fact1 ) )
				else:
					fact0 = np.std( Y0    , axis = 0 )
					fact1 = np.std( X0 , axis = 0 )
					self._cov_factor = np.diag( fact0 * np.power( fact1 , -1 ) )
			else:
				self._cov_factor = np.identity(Y0.shape[1])
		
		self._bin_width = self._bin_width if self._bin_width is not None else self._bin_width_estimator( Y0 , X0 , X1 )
		
		
		## Optimal plan
		planY0X0 = OTC( self._bin_width , self._bin_origin , p = self.p00 )
		planX0X1 = OTC( self._bin_width , self._bin_origin , p = self.p01 )
		planY0X0.fit( X0 , Y0 )
		planX0X1.fit( X1 , X0 )
		
		## Estimation of Y1
		yX0 = planY0X0.predict(Y0)
		yX1 = planX0X1.predict(yX0)
		motion = yX1 - yX0
		motion = np.apply_along_axis( lambda x : np.dot( self._cov_factor , x ) , 1 , motion )
		Y1 = Y0 + motion
		
		## Optimal plan for correction
		self.planX1Y1 = OTC( self._bin_width , self._bin_origin , p = self.p00 )
		self.planX1Y1.fit( Y1 , X1 )
	
	
	def predict( self , X1 ):
		"""
			Perform the bias correction
			
			Parameters
			----------
			X1  : np.array[ shape = (size,dimension) ]
				Array of value to be corrected
			
			Returns
			-------
			uX1 : np.array[ shape = (size,dimension) ]
				Return an array of correction
		"""
		return self.planX1Y1.predict( X1 )
	
	
	def _bin_width_estimator( self , Y0 , X0 , X1 ):
		bwX0 = bin_width_estimator(X0)
		bwX1 = bin_width_estimator(X1)
		bwY0 = bin_width_estimator(Y0)
		return np.min( (bwY0,bwX0,bwX1) , axis = 0 )

