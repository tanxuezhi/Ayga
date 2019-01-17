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
from Apyga.dynamic.discrete.__DiscDynSyst import DiscDynSyst

###########
## Class ##
###########

class Ikeda(DiscDynSyst):
	"""
		Apyga.dynamic.discrete.Ikeda
		============================

		Description
		-----------
		Ikeda attractor.
	"""
	def __init__( self , R = 1. , C1 = 0.4 , C2 = 0.9 , C3 = 6. , size = 1 ):
		"""
			Description
			-----------
			Initialisation of parameters of the model and the Apyga.Dynamic.DynamicalSystem class

			Parameters
			----------
			R    : float
			   Default = 1.
			C1   : float
			   Default = 0.4
			C2   : float
			   Default = 0.9
			C3   : float
			   Default = 6.
			size : int
			   Numbers of orbits must be computed

			Fix initializations
			-------------------
			dim    : Initialized at 2
			bounds : Initialized at np.array([ [0,-1] [1,0] ])
		"""
		DiscDynSyst.__init__( self , 2 , size , np.array([ [ 0. , -1. ] , [ 1. , 0. ] ]) )
		self.R = R
		self.C1 = C1
		self.C2 = C2
		self.C3 = C3
	
	def _equation( self , X , t = None ):
		Xnext = np.zeros_like(X)
		tau = self.C1 - self.C3 / ( 1. + np.power( X[self._i[0]] , 2 ) + np.power( X[self._i[1]] , 2 ) )
		Xnext[self._i[0]] = self.R + self.C2 * ( X[self._i[0]] * np.cos(tau) - X[self._i[1]] * np.sin(tau) )
		Xnext[self._i[1]] = self.C2 * ( X[self._i[1]] * np.cos(tau) + X[self._i[0]] * np.sin(tau) )
		return Xnext
	
