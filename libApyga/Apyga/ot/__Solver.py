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


import numpy as np
from Apyga.ot.__CySolver import CySolver

def Solver( massX , massY , costM ):
	"""Solves the Earth Movers distance problem and returns the OT matrix
	
	
	.. math::
		\gamma = arg\min_\gamma <\gamma,costM>_F
		
		s.t. \gamma 1 = massX
			\gamma^T 1= massY
			\gamma\geq 0
	where :
	
	- costM is the metric cost matrix
	- massX and massY are the sample weights
	
	Uses the algorithm proposed in [1]_
	
	Parameters
	----------
	massX : (ns,) ndarray, float64
		Source histogram (uniform weigth if empty list)
	massY : (nt,) ndarray, float64
		Target histogram (uniform weigth if empty list)
	costM : (ns,nt) ndarray, float64
		loss matrix
	
	Returns
	-------
	gamma: (ns x nt) ndarray
		Optimal transportation matrix for the given parameters
	
	
	Examples
	--------
	
	Simple example with obvious solution. The function emd accepts lists and
	perform automatic conversion to numpy arrays
	
	>>> import Apyga.ot as apyot
	>>> massX = [.5,.5]
	>>> massY = [.5,.5]
	>>> costM = [[0.,1.],[1.,0.]]
	>>> apyot.solveur( massX , massY , costM )
	array([[ 0.5,  0. ],
	       [ 0. ,  0.5]])
	
	References
	----------
	
	.. [1] Bonneel, N., Van De Panne, M., Paris, S., & Heidrich, W.
		(2011, December).  Displacement interpolation using Lagrangian mass
		transport. In ACM Transactions on Graphics (TOG) (Vol. 30, No. 6, p.
		158). ACM.
	
	"""
	massX = np.asarray( massX , dtype = np.float64 )
	massY = np.asarray( massY , dtype = np.float64 )
	costM = np.asarray( costM , dtype = np.float64 )
	
	# if empty array given then use unifor distributions
	if len(massX) == 0:
		massX = np.ones( ( costM.shape[0],) , dtype = np.float64 ) / costM.shape[0]
	if len(massY) == 0:
		massY = np.ones( ( costM.shape[1],) , dtype = np.float64) / costM.shape[1]
	
	return CySolver( massX , massY , costM )
