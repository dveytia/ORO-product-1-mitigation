#' Plot on second axis by automatically scaling
#'
#' @param dat.prim a dataframe with values to be plotted by the first axis, 
#' where y axis is named value, x axis is named x_axis, colour variable is named col_var
#' @param dat.sec a dataframe with values to be plotted by the second axis
#' @param name.prim The name of the first y axis
#' @param name.sec The name of the second y axis
#' @param lwd
#' 
my_sec_axis_lineplot <- function(dat.prim, dat.sec, name.prim, name.sec, lwd = 1){
  require(ggplot2)
  
  ylim.prim <- range(dat.prim$value, na.rm = T)   
  ylim.sec <- range(dat.sec$value, na.rm = T)
  b <- diff(ylim.prim)/diff(ylim.sec)
  a <- ylim.prim[1] - b*ylim.sec[1]
  
  ggp <- ggplot(dat.prim, aes(x = x_axis, y = value, color = col_var))+
    geom_line(linewidth = lwd)+
    geom_line(data = dat.sec, aes(x = x_axis, y = a + value*b, color = col_var), inherit.aes = FALSE, linewidth = lwd)+
    scale_y_continuous(name.prim,sec.axis = sec_axis(~ (. - a)/b, name = name.sec)) +
    theme_bw()
  
  return(ggp)
  
}

