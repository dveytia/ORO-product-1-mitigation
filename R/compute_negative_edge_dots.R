require(ggraph)
require(igraph)
require(tidygraph)
require(dplyr)
require(ggplot2)
require(grid)
require(tibble)

# Compute dot positions for negative edges that respect an absolute end_cap (mm).
compute_negative_edge_dots <- function(g, layout = "circle", end_cap_mm = 10, end_cap_mult_factor=3) {
  # g: igraph or tbl_graph
  # layout: layout string passed to ggraph
  # end_cap_mm: radius (in mm) of end_cap the edge is visually trimmed by
  # edge_layer_aes: a function that returns the aes(...) object you use for the edge layer,
  #                but MUST include an edge identifier aes like aes(edge_eid = as.factor(eid))
  #
  # Returns: data.frame with columns: eid (edge id), dot_x, dot_y
  
  # ensure tbl_graph and add stable edge id
  g_tidy2 <- as_tbl_graph(g) %>% activate(edges) %>% as_tibble() %>% mutate(eid = row_number())
  
  
  neg_edges_data <- g_tidy2 %>%
    filter(sign == -1) 
    
  
  p_edges <- ggraph(g_tidy2, layout = layout) +
    geom_edge_fan(
      aes(
        width = abs(weight),
        alpha = relative_nsig,
        color = color
        # filter = sign==-1

      ),
      lineend = "round",
      end_cap = circle(end_cap_mm, 'mm')
    ) +
    theme_void()
  
  pb <- ggplot_build(p_edges)
  edge_paths <- pb@data[[1]]  # contains x, y, group, width, alpha, color, etc.
  
  panel_scales <- pb@layout$panel_params[[1]]
  
  offset_data <- mm_to_data(end_cap_mm*end_cap_mult_factor, panel_scales)  # convert 10mm into approx data units
  
  edge_dots <- edge_paths %>%
    group_by(group) %>%
    group_modify(~ compute_dot(.x, offset = offset_data))

  
  # # For each edge, extract the path
  # edge_dots <- edge_paths %>%
  #   group_by(group) %>%
  #   summarise(
  #     dot_x = mean(xend),
  #     dot_y = mean(yend)
  #   )
  
  # Join back attributes?
  edge_dots <- edge_dots %>%
    left_join(g_tidy2,by = c("group"="eid"))%>%
    filter(sign == -1)

  
  dot_positions = edge_dots
  
  return(dot_positions)
}


mm_to_data <- function(mm, panel_scales) {
  # panel_scales is from ggplot_build(p)$layout$panel_params
  # assumes equal scaling in x and y
  xrange <- diff(panel_scales$x.range)
  yrange <- diff(panel_scales$y.range)
  # pick one axis (since circle end_cap uses radial distance)
  units_per_mm <- xrange / (panel_scales$x.range[2] - panel_scales$x.range[1]) / 100
  mm * units_per_mm
}

compute_dot <- function(path, offset) {
  dx <- diff(path$x)
  dy <- diff(path$y)
  seglen <- sqrt(dx^2 + dy^2)
  cumlen <- c(0, cumsum(seglen))
  total <- tail(cumlen, 1)
  target <- total - offset
  if (target < 0) target <- 0
  # linear interpolation
  i <- which(cumlen <= target)
  i <- max(i)
  t <- (target - cumlen[i]) / seglen[i]
  tibble(
    dot_x = path$x[i] + t * dx[i],
    dot_y = path$y[i] + t * dy[i]
  )
}
