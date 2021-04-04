#' Run Hierararchical Mahalanobis clustering accelerated on GPU
#'
#' @param dataMatrix Real matrix of cells in column major order (each column one cell)
#' @param threshold Real number in the interval of (0,1) defining
#' the minimal relative size of cluster (relative to the number of
#' observations) whose distance to other clusters will be computed as a
#' pure Mahalanobis distance.
#' @param subthreshHandling One of string values ('mahal','mahal0','euclid','euclidMahal'); 
#' it is a method how subthreshold clusters distort the space around them - i.e. how
## their size and shape gets reflected in distance computation.
#' @param normalize Boolean; if TRUE, cluster size
#' will be ignored when computing Mahalanobis distance from the cluster.
#' @param quick Boolean; If TRUE, inter-cluster
#' distances will be computed using centroids only. If FALSE,
#' all observations contained in the clusters will be used.
#' @return Returns hclust object that describes the clustering process
gmhclust <- function(dataMatrix, threshold=0.5, subthreshHandling="mahal", normalize=FALSE, quick=FALSE) {

	if (!is.double(dataMatrix))
		stop("argument dataMatrix must be real")
	if (threshold<0 || threshold>1)
		stop("argument threshold must be real value in <0,1>")
	if (!is.logical(normalize))
		stop("argument normalize must be logical")
	if (!is.logical(quick))
		stop("argument quick must be logical")

	if (subthreshHandling == "mahal")
		kind = 0
	else if (subthreshHandling == "euclid")
		kind = 1
	else if (subthreshHandling == "mahal0")
		kind = 2
	else if (subthreshHandling == "euclidMahal")
		kind = 3
	else {
		warning("Unknown subthreshHandling, setting to mahal")
		kind = 0
	}

	dim<-dim(dataMatrix)[1]
	count<-dim(dataMatrix)[2] 

	merge<-matrix(0L,count-1,2)
    height<-rep(0.0, count-1)
    ordering<-rep(0L, count)

	res<-.C('c_gmhclust', dataMatrix, count, dim, threshold, as.integer(kind), normalize, quick, m=merge, h=height, o=ordering)

	clust<-list(merge=res$m,height=res$h,order=res$o,labels=rownames(dataMatrix),
        method='mahalanobis-average',dist.method='euclidean')
    class(clust)<-'hclust'

	return(clust)
} 