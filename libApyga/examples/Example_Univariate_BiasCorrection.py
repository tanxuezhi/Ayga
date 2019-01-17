# -*- coding: utf-8 -*-

##################################################################################
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## This software is a computer program whose purpose is to study the            ##
## statistics of dynamical systems, statisticaly correct in uni/multivariate    ##
## the bias of dataset by applying the optimal transport to sparse histograms.  ##
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
##################################################################################
##                                                                              ##
## Copyright Yoann Robin, 2018                                                  ##
##                                                                              ##
## yoann.robin.k@gmail.com                                                      ##
##                                                                              ##
## Ce logiciel est un programme informatique servant à étudier les statistiques ##
## des systèmes dynamiques, corriger statistiquement en uni/multivarié des      ##
## données en appliquant le transport optimal à des histogrammes creux.         ##
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

##########
## Path ##
##########

import sys,os
pathApyga = os.path.join( os.path.abspath( os.path.dirname( __file__ ) ) , ".." )
sys.path.append(pathApyga)


###############
## Libraries ##
###############

import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import Apyga.stats.bc as apysbc


##########
## main ##
##########

if __name__ == "__main__":
	
	## Construction of biased and reference dataset
	size = 10000
	dsize = 7500

	X0 = np.random.normal( loc = 7 , scale = 1 , size = (size,1) )

	X1 = np.zeros( (size,1) )
	X1[:(size-dsize),:] = np.random.normal( loc = 5 , scale = 1 , size = (size-dsize,1) )
	X1[(size-dsize):,:] = np.random.normal( loc = 9 , scale = 1 , size = (dsize,1) )
	
	Y0 = np.zeros( (size,1) )
	Y0[:dsize,:] = np.random.exponential( scale = 1 , size =  (dsize,1) )
	Y0[dsize:,:] = np.random.normal( loc = 10 , scale = 1 , size = (size-dsize,1) )
	
	## Construction of corrector for period 0
	qm = apysbc.QM()
	qm.fit(Y0,X0)
	
	## Correction of period 0
	uX0 = qm.predict(X0)

	## Construction of corrector for period 1
	cdft = apysbc.CDFt()
	cdft.fit( Y0 , X0 , X1 )

	## Correction of period 1
	uX1 = cdft.predict(X1)

	## Random Variable
	bins = np.arange( -1 , 14 , 0.1 )
	rvY0 = sc.rv_histogram( np.histogram( Y0 , bins ) )
	rvX0 = sc.rv_histogram( np.histogram( X0 , bins ) )
	rvX1 = sc.rv_histogram( np.histogram( X1 , bins ) )
	rvUX0 = sc.rv_histogram( np.histogram( uX0 , bins ) )
	rvUX1 = sc.rv_histogram( np.histogram( uX1 , bins ) )

	## Plot
	bins = cdft._bins[0]
	fig_factor = 0.3
	fig = plt.figure( figsize = ( fig_factor * 30 , fig_factor * 20) )

	ax = fig.add_subplot( 2 , 3 , 1 )
	ax.hist( X0 , bins = bins , color = "red" , density = True )
	ax.set_ylim( (0,0.8) )
	ax.set_title( "Biased0" )

	ax = fig.add_subplot( 2 , 3 , 2 )
	ax.hist( uX0 , bins = bins , color = "green" , density = True )
	ax.set_ylim( (0,0.8) )
	ax.set_title( "Correction period 0" )

	ax = fig.add_subplot( 2 , 3 , 3 )
	ax.hist( Y0 , bins = bins , color = "blue" , density = True )
	ax.set_ylim( (0,0.8) )
	ax.set_title( "Ref0" )

	ax = fig.add_subplot( 2 , 3 , 4 )
	ax.hist( X1 , bins = bins , color = "red" , density = True )
	ax.set_ylim( (0,0.8) )
	ax.set_title( "Biased1" )
	
	ax = fig.add_subplot( 2 , 3 , 5 )
	ax.hist( uX1 , bins = bins , color = "green" , density = True )
	ax.set_ylim( (0,0.8) )
	ax.set_title( "Correction period 1" )

	ax = fig.add_subplot( 2 , 3 , 6 )
	ax.plot( bins , rvY0.cdf(bins) , color = "blue" , label = "ref" )
	ax.plot( bins , rvX0.cdf(bins) , color = "red" , label = "X0" )
	ax.plot( bins , rvX1.cdf(bins) , color = "red" , linestyle = "--" , label = "X1" )
	ax.plot( bins , rvUX0.cdf(bins) , color = "green" , label = "corr0" )
	ax.plot( bins , rvUX1.cdf(bins) , color = "green" , linestyle = "--" , label = "corr1" )
	ax.legend( loc = "upper left" )
	ax.set_title( "CDF" )

	plt.tight_layout()
	plt.show()

	print("Done")
