---
layout: landing
title: Data
description: Seaborn, Pandas, and other Data Manipulations
image: assets/images/DSC_8290.jpg
author: Stego
show_tile: true
nav-menu: true
---

<div id="main">
<section id="visualization" >
<div class="inner" markdown="1">
  
  This page serves several purposes: cheat sheets, reminders of what I've done and codes that others may find helpful. For now, it only contains visualization part. 



## <u>Visualization Using Seaborn</u>



Python is a very convenient visualization tool. Besides interactive visualized tool (ex: Bokeh, Dash, Plotly), we could also generate static charts and graphs through matplotlib and seaborn. This article would be a cheat sheet that I found helpful when plotting. 

 

#### Setting 

Before starting to plot, there are some set ups that I would do beforehand. Since the default font doesn't contain non-ascii characters, we would have to use the code below to display Mandarin or non-ascii characters. 

```python
sns.set(font=['sans-serif'])
sns.set_style("whitegrid",{"font.sans-serif":['Microsoft JhengHei']})
plt.rcParams['axes.unicode_minus'] = False
```



#### Start off

There are couple ways to start a chart:

```python
#The easiest way
plt.subplots() 
#This way offers more to adjust ex:ticker format
fig, ax = plt.subplots(figsize=(8,5))
#Create multiple plots, axes would be a list, usually requires 'for loop' if using this method
fig, axes = plt.subplots(3,3, figsize=(10,10))
```



#### Adjusting

Some settings that I often use when plotting the chart

```python
#rotate x ticks by certain degree
plt.xticks(rotation=85)
#set up the title for the chart
plt.title('Chart Title')
#set up the label
plt.ylabel('Quantity')
#set up display limit
plt.xlim(xmin, xmax)

#if adding a vertical line
plt.vlines(y, xmin, xmax, color='orange', label='label')

#formatting the tickers
import matplotlib.ticker as ticker
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: ""{:,}".format(int(x)))) #comma as thousand seperator
ax.yaxis.set_major_formatter(ticker.EngFormatter()) #Quantity unit ex: k, m
                                                     
                                                     
```

#### Plotting graphs

I prefer seaborn more than other plotting tools since it looks more elegant to me. There are some functions that I find handy dealing with multi-dimension data.

The codes could be found in seaborn official website.

```python
g = sns.FacetGrid(tips, col="sex", row="time", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")

sns.relplot(
    data=tips, x="total_bill", y="tip", col="time",
    hue="time", size="size", style="sex",
    palette=["b", "r"], sizes=(10, 100)
)

g = sns.catplot(x="alive", col="deck", col_wrap=4,
                data=df, 
                kind="count", height=2.5, aspect=.8)
```

#### Two Y-axis

If using two y axis in the chart, we could use the function below to achieve the goal. However, remember to turn off grid of one axis or the chart would be quite messy.

```python
ax2 = ax1.twinx()
ax2.grid(False)
```

#### Saving figure

Use this at the very end and before showing the graph. The 'tight' parameter would normally save the figure you saw in jupyter notebook.

```python
plt.savefig('name.png', bbox_inches='tight', dpi=600)
```



#### Example

For this example, we are showing the distribution for 9 columns in one chart. First, create a 3x3 figure. All the 'ax' would be stored in 'axes'. Hence, we are running for loop on it to plot each graph. If there are only 8 columns and we don't want to display the empty canvas, we could set visible to false using the code that I annotate before showing the figure.

```python
fig, axes = plt.subplots(3,3, figsize=(10,10))
fig.suptitle('Time Distribution', fontsize=16, y=1.03)
time_col = ['STS1', 'ZQT2', 'ZQT3', 'ZQT4', 'ZQT5', 'ZQT6', 'ZQT7', 'ZQT8', 'STSX']
a, b = 0, 0
for i, column in enumerate(time_col, 1):
    sns.distplot(df[column], kde=False, ax=axes[a,b])
    plt.tight_layout()
    axes[a,b].set_title(column)
    axes[a,b].set_ylim(0, 135000)
    axes[a,b].set_xlim(0, 17)
    axes[a,b].yaxis.set_major_formatter(ticker.EngFormatter())
    b+=1
    if b >= 3:
        a+=1
        b-=3
#axes.flat[-1].set_visible(False)
plt.show()
```
![Time Distribution chart](/assets/images/TimeDist.jpg){:width="500px"}
</div>
</section>
</div>
