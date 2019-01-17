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

import time
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import Apyga.stats.bc as apysbc


##########
## main ##
##########

if __name__ == "__main__":
	
	## Construction of X0 (biased period 0), X1 (biased period 1) and Y0 (reference period 0)
	size = 20000
	sized0 = 10000
	sized1 = 15000
	
	## Just a gaussian for X0
	X0 = np.random.multivariate_normal( mean = [0.,0.] , cov = np.identity(2) , size = size )
	Y = np.zeros( (size,2) )
	
	## A lightly complex gaussian for X1
	X1 = np.random.multivariate_normal( mean = [1.,2.] , cov = [ [2.,0] , [0,0.5] ] , size = size )
	
	## A very complex law for Y0
	Y[:sized0,:] = np.random.multivariate_normal( mean = [7.,7.] , cov = np.array( [2,0,0,0.5] ).reshape( (2,2) ) , size = sized0 )
	Y[sized0:sized1,:] = np.random.multivariate_normal( mean = [5.,9.] , cov = np.array( [0.5,0,0,2] ).reshape( (2,2) ) , size = (sized1-sized0) )
	Y[sized1:] = np.random.multivariate_normal( mean = [5.,12.5] , cov = 0.2 * np.identity(2) , size = (size-sized1) )
	meanY = np.mean( Y , axis = 0 )
	meanX = np.mean( X0 , axis = 0 )
	Y = np.apply_along_axis( lambda x : x - meanY + meanX , 1 , Y )
	
	## Construction of corrector period0
	time0 = time.clock()
	otc = apysbc.OTC( bin_width = [ 0.1 , 0.1 ] )
	otc.fit( Y , X0 )
	
	## Correction period0
	time1 = time.clock()
	uX0 = otc.predict(X0)
	
	## Construction of corrector period1
	time2 = time.clock()
	cov_factor = "std"
#	cov_factor = "cholesky"
#	cov_factor = "identity"
	dotc = apysbc.dOTC( bin_width = [0.1,0.1] , cov_factor = cov_factor )
	dotc.fit( Y , X0 , X1 )
	
	## Correction period1
	time3 = time.clock()
	uX1 = dotc.predict(X1)
	
	## Print time
	time4 = time.clock()
	print( "Corrector0 estimation time = {}s".format( round( time1 - time0 , 2 ) ) )
	print( "Correction0 time = {}s".format( round( time2 - time1 , 2 ) ) )
	print( "Corrector1 estimation time = {}s".format( round( time3 - time2 , 2 ) ) )
	print( "Correction1 time = {}s".format( round( time4 - time3 , 2 ) ) )
	
	## Pearson correlation
	pY,_   = sc.spearmanr( Y[:,0] , Y[:,1] )
	pX0,_  = sc.spearmanr( X0[:,0] , X0[:,1] )
	pX1,_  = sc.spearmanr( X1[:,0] , X1[:,1] )
	pUX0,_ = sc.spearmanr( uX0[:,0] , uX0[:,1] )
	pUX1,_ = sc.spearmanr( uX1[:,0] , uX1[:,1] )
	
	## Histogram for plot
	bins = [ np.arange( -8 , 8 , 0.1 ) for i in range(2) ]
	extent = [-8,8,-8,8]
	
	HX0,_,_ = np.histogram2d( X0[:,0] , X0[:,1] , bins = bins )
	HX0 = HX0 / np.sum(HX0)
	HX0[HX0 == 0] = np.nan
	
	HX1,_,_ = np.histogram2d( X1[:,0] , X1[:,1] , bins = bins )
	HX1 = HX1 / np.sum(HX1)
	HX1[HX1 == 0] = np.nan
	
	HY,_,_ = np.histogram2d( Y[:,0] , Y[:,1] , bins = bins )
	HY = HY / np.sum(HY)
	HY[HY == 0] = np.nan
	
	HuX0,_,_ = np.histogram2d( uX0[:,0] , uX0[:,1] , bins = bins )
	HuX0 = HuX0 / np.sum(HuX0)
	HuX0[HuX0 == 0] = np.nan
	
	HuX1,_,_ = np.histogram2d( uX1[:,0] , uX1[:,1] , bins = bins )
	HuX1 = HuX1 / np.sum(HuX1)
	HuX1[HuX1 == 0] = np.nan
	
	vmin = min( [ np.nanmin(X) for X in [ HX0 , HY , HuX0 ] ] )
	vmax = max( [ np.nanmax(X) for X in [ HX0 , HY , HuX0 ] ] )
	
	## Plot
	fig = plt.figure( figsize = (15,10) )
	
	ax = fig.add_subplot( 2 , 3 , 1 )
	ax.imshow( np.rot90(HX0) , cmap = plt.cm.inferno , extent = extent , vmin = vmin , vmax = vmax )
	ax.set_title( "Biased 0, pearson = {}".format( round( pX0 , 2 ) ) )
	
	ax = fig.add_subplot( 2 , 3 , 2 )
	ax.imshow( np.rot90(HuX0) , cmap = plt.cm.inferno , extent = extent , vmin = vmin , vmax = vmax )
	ax.set_title( "Correction, pearson = {}".format( round( pUX0 , 2 ) ) )
	
	ax = fig.add_subplot( 2 , 3 , 3 )
	ax.imshow( np.rot90(HY) , cmap = plt.cm.inferno , extent = extent , vmin = vmin , vmax = vmax )
	ax.set_title( "Reference, pearson = {}".format( round( pY , 2 ) ) )
	
	ax = fig.add_subplot( 2 , 3 , 4 )
	ax.imshow( np.rot90(HX1) , cmap = plt.cm.inferno , extent = extent , vmin = vmin , vmax = vmax )
	ax.set_title( "Biased 1, pearson = {}".format( round( pX1 , 2 ) ) )
	
	ax = fig.add_subplot( 2 , 3 , 5 )
	ax.imshow( np.rot90(HuX1) , cmap = plt.cm.inferno , extent = extent , vmin = vmin , vmax = vmax )
	ax.set_title( "Correction, pearson = {}".format( round( pUX1 , 2 ) ) )
	
	plt.tight_layout()
	plt.show()
	
	print("Done")
