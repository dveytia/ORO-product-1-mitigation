# Taken mostly from https://tradeblotter.wordpress.com/2013/02/28/the-paul-tol-21-color-salute/

# Function for plotting colors side-by-side
pal <- function(col, border = "light gray", ...){
  n <- length(col)
  title <- deparse(substitute(col))
  plot(0, 0, type="n", xlim = c(0, 1), ylim = c(0, 1),
       axes = FALSE, main=title,xlab = "", ylab = "", ...)
  rect(0:(n-1)/n, 0, 1:n/n, 1, col = col, border = border)
}

# Function for plotting colors as pie
palpie <- function(col)
{
  par(mar=c(1,1,1,1))
  ones <- rep(1, length=length(col))
  names(ones) <- col
  
  title <- deparse(substitute(col))
  
  pie(ones, col=col, cex=1, main = title)
  par(mar=c(5.1, 4.1, 4.1, 2.1))
}



########## DEFINE Color palettes #########


#From Paul Tol: https://personal.sron.nl/~pault/
Tol_bright <- c('#EE6677', '#228833', '#4477AA', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB')

Tol_muted <- c('#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933','#CC6677', '#882255', '#AA4499', '#DDDDDD')

Tol_light <- c('#BBCC33', '#AAAA00', '#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#DDDDDD')

#From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
Okabe_Ito <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")




########################## SHOW as multiple strips ###################


par(mfrow=c(2,2))

pal(Okabe_Ito)
pal(Tol_bright)
pal(Tol_muted)
pal(Tol_light)

par(mfrow=c(1,1))



########################## SHOW as multiple PIE Charts ###################
# 
# par(mfrow=c(2,2))
# 
# palpie(Okabe_Ito)
# palpie(Tol_bright)
# palpie(Tol_muted)
# palpie(Tol_light)





