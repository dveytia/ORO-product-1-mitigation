vlte_with_lags <- function (Y, X, maxLag=5, nboot=100, lx = 1, ly = 1, VLflag = TRUE, 
          autoLagflag = TRUE, alpha = 0.05) 
{
  follOut <- c()
  if (missing(maxLag)) 
    maxLag <- 0.2 * length(Y)
  if (VLflag) {
    if (autoLagflag == TRUE) {
      follOut <- followingRelation(Y = Y, X = X, timeLagWindow = maxLag)
      if (follOut$optDelay >= 20) {
        lx <- 20
        ly <- 20
      }
      else {
        lx <- follOut$optDelay
        ly <- follOut$optDelay
      }
    }
    else {
      follOut <- followingRelation(Y = Y, X = X, maxLag)
    }
    follX <- c(follOut$nX[-1], 0)
    X <- follX
  }
  else if (autoLagflag) {
    follOut <- followingRelation(Y = Y, X = X)
    if (follOut$optDelay >= 20) {
      lx <- 20
      ly <- 20
    }
    else {
      lx <- follOut$optDelay
      ly <- follOut$optDelay
    }
  }
  lx <- max(1, lx)
  ly <- max(1, ly)
  res <- transfer_entropy(x = X, y = Y, nboot = nboot, lx = lx, 
                          ly = ly, quiet = TRUE)
  pval <- res$coef[1, 4]
  TEratio <- res$coef[1]/res$coef[2]
  XgCsY_trns <- FALSE
  if (!is.na(TEratio)) 
    if (TEratio > 1) {
      if (!is.na(pval)) {
        if (pval <= alpha) 
          XgCsY_trns = TRUE
      }
      else XgCsY_trns = TRUE
    }
  return(list(TEratio = TEratio, res = res, follOut = follOut, 
              XgCsY_trns = XgCsY_trns, pval = pval))
}
