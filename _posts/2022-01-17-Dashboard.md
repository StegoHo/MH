---
layout: page
title: Dashboard with Bokeh
description: Dashboard with Bokeh
image: 

---


Data Dashboard is essential for decision making. It offers up-to-date trend and critical number in an easy interpreting visualized chart. Furthermore, using packages such as Bokeh, it is possible to run on local network and other users just need to open the browser to visit the dashboard. 


An easy example is shown below using TSMC’s stock price in Taiwan stock market. Since I want to use the full time series on the price chart, the absence of price between days would be the time that the stock market isn’t open here (could be national holidays or weekend).


Besides chart, there are lots of interactive buttons and functions that can be served. Users can then change chart or gather information from the period that they are interested in. When hovering over bars on charts, users could see the pop-up information, which is quite convenient.

 
Dashboard created by Bokeh can look as beautiful as modern website this day. It supports examples on Bootstrap and awesome icons which come in handy when decorating the website. The picture below provides some examples using both Bootstrap and icons to decorate the interface.

![designed interface](assets/images/2021-10-17_Dashboard.png)

The code is revised from the example provided by [Bokeh](https://github.com/bokeh/bokeh/tree/branch-3.0/examples/app/dash).



The video below demonstrates how Bokeh server would eventually work on the user end.

<center>
<video width="720" height="405" controls>
  <source src="assets/images/2021-10-17_Dashboard.mp4" type="video/mp4">
</video>
</center>
