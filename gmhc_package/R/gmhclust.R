gmhclust <- function(dataMatrix, threshold, kind) {

	dim<-dim(dataMatrix)[1]
	count<-dim(dataMatrix)[2] 

	merge<-matrix(0L,count-1,2)
    height<-matrix(0.0,count-1,1)

	if (!is.double(dataMatrix))
		stop("argument x must be real")

	res<-.C('gmhclust_C', dataMatrix, count, dim, threshold, kind, m=merge, h=height)

	return(list(merge=res$m, height=res$h))
} 