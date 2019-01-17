
#############################
## Yoann Robin             ##
## yoann.robin.k@gmail.com ##
#############################

base::rm( list = base::ls() )

################
## Librairies ##
################

library(methods)
library(devtools)
library(ggplot2)
library(gridExtra)
library(rgl)
load_all("ARyga")


###############
## Fonctions ##
###############

pairwise_distances_test = function() ##{{{
{
	X = matrix( nrow = 2 , ncol = 50 )
	Y = matrix( nrow = 2 , ncol = 30 )
	
	for( i in 1:2 )
	{
		X[i,] = runif( 50 )
		Y[i,] = runif( 30 )
	}
	
	distXY = ARyga::pairwise_distances( X , Y ) ## pairwise distance between each column of X and Y for euclidean distance (default )
	distXX = ARyga::pairwise_distances( X , metric = "sqeuclidean" )  ## pairwise distance between each column of X for sqeuclidean distance
	distXYinf = ARyga::pairwise_distances( X , Y , metric = function( x , y ) { max( abs( x - y ) ) } ) ## pairwise distance between each column of X and Y for custom metric

} ##}}}

optimal_transport_test = function() ## {{{
{
	## Two Gaussians
	size = 10000
	X = matrix( nrow = 1 , ncol = size )
	Y = matrix( nrow = 1 , ncol = size )
	
	X[1,] = rnorm( size )
	Y[1,] = rnorm( size , mean = 10 )
	
	
	## Histograms
	bin_width = c( 0.1 )
	muX = ARyga::SparseHist( X , bin_width )
	muY = ARyga::SparseHist( Y , bin_width )
	plot( muX$c() , muX$p() ) ## Bins centers
	
	
	## Optimal Transport
	otPlan = ARyga::TrPlan( muX , muY )
	print(otPlan$cost())
	print(otPlan$plan())
	
} ##}}}

distances_test = function() ##{{{
{
	## Data
	size = 10000
	X = matrix( nrow = 1 , ncol = size )
	Y = matrix( nrow = 1 , ncol = size )
	
	X[1,] = rnorm( size )
	Y[1,] = rnorm( size , mean = 10 )
	
	## Measures
	bin_width = c(0.2)
	muX = ARyga::SparseHist( X , bin_width )
	muY = ARyga::SparseHist( Y , bin_width )
	
	print( dist.wasserstein( muX , muY ) )
	print( dist.minkowski( muX , muY , 7 ) )
	print( dist.euclidean( muX , muY ) )
	print( dist.energy( muX , muY ) )
}
## }}}

rv_hist_test = function() ##{{{
{
	X = stats::rnorm( 10000 )
	
	rvX = ARyga::rv_histogram$new(X)
	
	x = base::seq( min(X), max(X) , 1e-2 )
	q = base::seq( 0 , 1 , 1e-3 )
	
	graphics::par( mfrow = base::c(2,2) )
	plot( X , col = "blue" )
	
	plot( rvX$c , rvX$p , col = "blue" , type = "l" , main = "density" )
	
	plot( x , rvX$cdf(x) , col = "blue" , type = "l" , main = "CDF and SF" )
	lines( x , rvX$sf(x) , col = "red" )
	
	plot( q , rvX$icdf(q) , col = "blue" , type = "l" , main = "inverse CDF an SF" )
	lines( q , rvX$isf(q) , col = "red" )
}

##}}}

qm_test = function() ##{{{
{
	## Data
	X = matrix( NA , nrow = 2 , ncol = 10000 )
	Y = matrix( NA , nrow = 2 , ncol = 10000 )
	X[1,] = rnorm( 10000 )
	X[2,] = rexp( 10000 )
	Y[1,] = rexp( 10000 )
	Y[2,] = rnorm( 10000 )
	
	## Bias correction by margins
	qm = QM$new()
	qm$fit( Y , X )
	uX = qm$predict(X)
	
	## Plot
	p1 = ggplot2::ggplot( data.frame( x = X[1,]  , y = X[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased")
	p2 = ggplot2::ggplot( data.frame( x = Y[1,]  , y = Y[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Ref")
	p3 = ggplot2::ggplot( data.frame( x = uX[1,] , y = uX[2,] ) , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Unbiased")
	gridExtra::grid.arrange( p1 , p2 , p3 , nrow = 2 )
	graphics::par( mfrow = base::c( 1 , 2 ) )

} ##}}}

cdft_test = function() ##{{{
{
	## Data
	X0 = matrix( NA , nrow = 2 , ncol = 10000 )
	X1 = matrix( NA , nrow = 2 , ncol = 10000 )
	Y0 = matrix( NA , nrow = 2 , ncol = 10000 )
	X0[1,] = rnorm( 10000 )
	X0[2,] = rexp( 10000 )
	X1[1,] = rnorm( 10000 , mean = 2 )
	X1[2,] = rexp( 10000 ) + 2
	Y0[1,] = rexp( 10000 )
	Y0[2,] = rnorm( 10000 )
	
	## Bias correction by margins
	cdft = ARyga::CDFt$new()
	cdft$fit( Y0 , X0 , X1 )
	uX1 = cdft$predict(X1)
	
	## Plot
	p1 = ggplot2::ggplot( data.frame( x = X0[1,]  , y = X0[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased 0") + ggplot2::xlim( -5 , 12 ) + ggplot2::ylim( -5 , 12 )
	p2 = ggplot2::ggplot( data.frame( x = Y0[1,]  , y = Y0[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Ref 0") + ggplot2::xlim( -5 , 12 ) + ggplot2::ylim( -5 , 12 )
	p3 = ggplot2::ggplot( data.frame( x = X1[1,]  , y = X1[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased 1") + ggplot2::xlim( -5 , 12 ) + ggplot2::ylim( -5 , 12 )
	p4 = ggplot2::ggplot( data.frame( x = uX1[1,] , y = uX1[2,] ) , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Unbiased 1") + ggplot2::xlim( -5 , 12 ) + ggplot2::ylim( -5 , 12 )
	gridExtra::grid.arrange( p1 , p2 , p3 , p4 , nrow = 2 )
} ##}}}

OTC_test = function() ##{{{
{
	## Parameters
	size = 10000
	Dim = 2
	bin_width = rep( 0.1 , Dim )

	## Data
	Y = matrix( nrow = Dim , ncol = size )
	X = matrix( nrow = Dim , ncol = size )
	Y[1,] = rexp( size )
	Y[2,] = rnorm( size )
	X[1,] = rnorm( size )
	X[2,] = rexp( size )
	
	## Correction
	otc = ARyga::OTC$new()
	otc$fit( Y , X )
	uX = otc$predict( X )
	
	## Plot
	p1 = ggplot2::ggplot( data.frame( x = X[1,]  , y = X[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased")
	p2 = ggplot2::ggplot( data.frame( x = Y[1,]  , y = Y[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Ref")
	p3 = ggplot2::ggplot( data.frame( x = uX[1,] , y = uX[2,] ) , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Unbiased")
	gridExtra::grid.arrange( p1 , p2 , p3 , nrow = 2 )

} ##}}}

dOTC_test = function() ##{{{
{
	## Parameters
	size = 10000
	bin_width = rep( 0.1 , 2 )

	## Data
	Y0 = matrix( nrow = 2 , ncol = size )
	X0 = matrix( nrow = 2 , ncol = size )
	X1 = matrix( nrow = 2 , ncol = size )
	
	mY0 = c(0,10)
	mX0 = c(0,0)
	mX1 = c(10,0)
	
	for( i in 1:2 )
	{
		Y0[i,] = rnorm( size , mean = mY0[i] , sd = 0.5 )
		X0[i,] = rnorm( size , mean = mX0[i] , sd = 2 )
		X1[i,] = rnorm( size , mean = mX1[i] , sd = 0.5 )
	}
	
	
	## Correction
	dotc = ARyga::dOTC$new( cov_factor = "cholesky" )
	dotc$fit( Y0 , X0 , X1 )
	uX1 = dotc$predict(X1)
	
	## Plot
	p1 = ggplot2::ggplot( data.frame( x = X0[1,]  , y = X0[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased 0") + ggplot2::xlim( -5 , 15 ) + ggplot2::ylim( -5 , 15 )
	p2 = ggplot2::ggplot( data.frame( x = Y0[1,]  , y = Y0[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Ref 0") + ggplot2::xlim( -5 , 15 ) + ggplot2::ylim( -5 , 15 )
	p3 = ggplot2::ggplot( data.frame( x = X1[1,]  , y = X1[2,] )  , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Biased 1") + ggplot2::xlim( -5 , 15 ) + ggplot2::ylim( -5 , 15 )
	p4 = ggplot2::ggplot( data.frame( x = uX1[1,] , y = uX1[2,] ) , ggplot2::aes( x = x , y = y ) ) + ggplot2::geom_bin2d() + ggplot2::theme_bw() + ggplot2::ggtitle("Unbiased 1") + ggplot2::xlim( -5 , 15 ) + ggplot2::ylim( -5 , 15 )
	gridExtra::grid.arrange( p1 , p2 , p3 , p4 , nrow = 2 )
}
##}}}

orbit_snap_l63_test = function( show ) ##{{{
{
	t = base::seq( 0 , 100 , 0.005 )
	size = 1000
	l63 = ARyga::Lorenz63$new( size = size )
	X = l63$orbit(t)
	
	
	## Plot each trajectory
	if( show == "orbit" )
	{
		colors = grDevices::rainbow( base::min(size,5) )
		rgl::plot3d( X[1,1,] , X[1,2,] , X[1,3,] , col = colors[1] )
		if( size > 1 )
		{
			for( i in 2:base::min(size,5) )
			{
				rgl::points3d( X[i,1,] , X[i,2,] , X[i,3,] , col = colors[i] )
			}
		}
	}
	
	
	## Plot snapshot
	if( show == "snap" )
	{
		l = dim(X)[1]
		colors = grDevices::rainbow(5)
		rgl::plot3d( X[,1,l] , X[,2,l] , X[,3,l] , col = colors[1] )
		
		for( i in 1:4 )
		{
			j = l - 100 * i
			rgl::points3d( X[,1,j] , X[,2,j] , X[,3,j] , col = colors[i] )
		}
	}
}##}}}

localdim_test = function() ##{{{
{
	l63 = ARyga::Lorenz63$new()
	t = base::seq( 0 , 100 , 0.005 )
	n_time = length(t)
	X = l63$orbit(t)
	
	R = X[,base::sample(1:n_time,1000)]
	
	res = ARyga::localDimension( R , X )
	print(mean(res$ld))
}##}}}


##########
## main ##
##########

#pairwise_distances_test()
#optimal_transport_test()
#distances_test()
#rv_hist_test()
#qm_test()
#cdft_test()
#QMrs_test()
#OTC_test()
#dOTC_test()
#orbit_snap_l63_test( "orbit" )
#orbit_snap_l63_test( "snap" )
#localdim_test()


