## Plot on second axis by automatically scaling

ylim.prim <- range(oecd_transport_trends_global$value, na.rm = T)   
ylim.sec <- range(wb_capture_fisheries_global$value, na.rm = T)
b <- diff(ylim.prim)/diff(ylim.sec)
a <- ylim.prim[1] - b*ylim.sec[1]

all_volume_global_ggp <- ggplot(oecd_transport_trends_global, 
                                aes(year, value, color = vehicle_type)) +
  geom_line() +
  geom_line(data = wb_capture_fisheries_global,
            aes(x=year, y = a + value*b, color = vehicle_type), 
            inherit.aes = FALSE) +
  scale_y_continuous("Fisheries production\n(metric tons)", 
                     sec.axis = sec_axis(~ (. - a)/b, name = "Maritime transport\n(Tonnes)")) +
  scale_x_date("Year") +
  scale_color_discrete(name="Sector")+
  labs(caption = "Data sources:\nWorld bank capture fisheries production (tonnes)\nOECD Annual transport trends")+
  theme_bw()+
  theme(
    legend.position = "bottom"
  )

all_volume_global_ggp
