## My QPress functions

# uncertainSampleProbsDf <- data.frame(
#   Pair = unique(edges$Pair[uncertain]),
#   Probability = c(rep(0.3, 5), rep(0.5, 4))
# )


my.community.sampler <- function (edges, required.groups = c(0), uncertainSampleProbsDf=NULL) 
{
  n.nodes <- length(node.labels(edges))
  weight.labels <- edge.labels(edges)
  n.edges <- nrow(edges)
  W <- matrix(0, n.nodes, n.nodes)
  lower <- ifelse(edges$Type == "U" | edges$Type == "N", -1L, 
                  0L)
  upper <- ifelse(edges$Type == "U" | edges$Type == "P", 1L, 
                  0L)
  k.edges <- as.vector(unclass(edges$To) + (unclass(edges$From) - 
                                              1) * n.nodes)
  uncertain <- which(!(edges$Group %in% required.groups))
  expand <- match(edges$Pair[uncertain], unique(edges$Pair[uncertain]))
  n.omit <- max(0, expand)
  zs <- rep(1, n.omit)
  
  # My addition
  if(!is.null(uncertainSampleProbsDf)){
    edgePairs <- unique(edges$Pair[uncertain])
    pairMatchIdx <- match(edgePairs, uncertainSampleProbsDf$Pair)
    p <- uncertainSampleProbsDf$Probability[pairMatchIdx]
    
    if (any(is.na(p))) {
      warning("Some uncertain edge pairs in `edges` were not found in `uncertainSampleProbsDf$Pair`.")
    }
    # for(i in 1:n.omit){
    #   zs[i] <- rbinom(n=1, size=1, prob=p[i])
    # }
  }
  
  community <- if (n.omit > 0) {
    function() {
      r <- runif(n.edges, lower, upper)
      r[uncertain] <- r[uncertain] * zs
      W[k.edges] <- r
      W
    }
  }
  else {
    function() {
      W[k.edges] <- runif(n.edges, lower, upper)
      W
    }
  }
  
  select <- if (n.omit > 0) {
    function(p) {
      # zs <<- rbinom(n.omit, 1, p)[expand]
      newzs <- rep(1, n.omit)
      for(i in 1:n.omit){
        if (!is.na(p[i])) {
          zs[i] <<- rbinom(n = 1, size = 1, prob = p[i])
        } else {
          zs[i] <<- 0  # Or 1, or NA — whatever default makes sense
        }
      }
      zs
    }
  }
  else {
    function(p = 0) {
      zs
    }
  }
  weights <- function(W) {
    W[k.edges]
  }
  list(community = community, select = select, weights = weights, 
       weight.labels = weight.labels, uncertain.labels = weight.labels[uncertain])
}














my.system.simulate <- function (n.sims, edges, required.groups = c(0), uncertainSampleProbsDf=NULL, sampler = my.community.sampler(edges, 
                                                                             required.groups, uncertainSampleProbsDf), validators = NULL) 
{
  As <- vector("list", n.sims)
  ws <- matrix(0, n.sims, nrow(edges))
  total <- 0
  stable <- 0
  accepted <- 0
  while (accepted < n.sims) {
    total <- total + 1
    z <- sampler$select(runif(1))
    W <- sampler$community()
    if (!stable.community(W)) 
      next
    stable <- stable + 1
    if (!all(as.logical(lapply(validators, function(v) v(W))))) 
      next
    accepted <- accepted + 1
    As[[accepted]] <- -solve(W)
    ws[accepted, ] <- sampler$weights(W)
  }
  colnames(ws) <- sampler$weight.labels
  list(edges = edges, A = As, w = ws, total = total, stable = stable, 
       accepted = accepted)
}
