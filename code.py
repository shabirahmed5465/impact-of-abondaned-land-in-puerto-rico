# Import Library
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import wbgapi as wb
import warnings
warnings.filterwarnings('ignore')

# Take the dataset from World-bank 
# Function for take dataset based name from World-bank
def search_data(indicator): 
    data = wb.data.DataFrame(indicator)
    dataframe = pd.DataFrame(data)
    transposed_dataframe = dataframe.T 
    return dataframe, transposed_dataframe

forest, transposed_forest = search_data('AG.LND.FRST.ZS')
agriculture, transposed_agriculture = search_data('AG.LND.AGRI.ZS')
population, transposed_population = search_data('SP.URB.TOTL.IN.ZS')
gas_emission, transposed_gas_emission = search_data('EN.ATM.GHGT.ZG')


# Data Cleaning
# Function for renaming data index and columns
def data_renaming(dataframe):
    renamed_dataframe = dataframe.copy()
    renamed_dataframe.columns.names = ['Country Code']
    renamed_dataframe.index = np.arange(1960,2022)
    renamed_dataframe.index.names = ['Year']
    return renamed_dataframe

renamed_forest = data_renaming(transposed_forest)
renamed_agriculture = data_renaming(transposed_agriculture)
renamed_population = data_renaming(transposed_population)
renamed_gas_emission = data_renaming(transposed_gas_emission)

# Function for handling missing value
def data_null_handling(dataframe):
    cleaned_dataframe = dataframe.copy()
    cleaned_dataframe.dropna(how='all', axis = 1, inplace = True)
    cleaned_dataframe.dropna(how='all', axis = 0, inplace = True)
    
    for country in cleaned_dataframe.columns:
        mean = cleaned_dataframe[country].mean()
        cleaned_dataframe[country].fillna(mean, inplace = True)
        
    return cleaned_dataframe

cleaned_forest = data_null_handling(renamed_forest)
cleaned_agriculture = data_null_handling(renamed_agriculture)
cleaned_population = data_null_handling(renamed_population)
cleaned_gas_emission = data_null_handling(renamed_gas_emission)


# Function for manipulating data
# Function for aggregate to global data
def get_global_data(data, name):
    year_list = data.index
    avg_by_time = [data.loc[year,:].mean() for year in year_list]
    global_data = pd.DataFrame({
        'Year': year_list, 
        name: avg_by_time}
    )
    global_data.set_index('Year', inplace= True)
    return global_data

# Function for calculate avg global data
def get_avg_per_year(dataframe):
    difference_each_year = []
    start_year = dataframe.index[0]
    end_year = dataframe.index[-1]
    
    for i in range(1, len(dataframe)):
        difference_each_year.append(abs(dataframe.iloc[i-1] - dataframe.iloc[i]))

    difference_avg = sum(difference_each_year) / len(difference_each_year)
    return round(difference_avg, 3)

# Function for aggregate country data
def agg_data(country):
    country_forest = cleaned_forest.loc[1990:2020,country]
    country_agriculture = cleaned_agriculture.loc[1990:2020,country]
    country_gas_emission = cleaned_gas_emission.loc[1990:2020,country]
    country_population = cleaned_population.loc[1990:2020,country]
    
    concatenated_data = pd.concat([
        country_forest, country_agriculture, 
        country_gas_emission, country_population
    ], axis=1)
    
    concatenated_data.columns = [
        'Forest Area (%)', 'Agricultural Land (%)', 
        'Greenhouse Gas Emissions (%)', 'Urban Population (%)'
    ]
    
    return concatenated_data

# Function for create correlation table
def corr_table(country):
    country_corr_table = agg_data(country)
    return country_corr_table.corr()

# Function for plotting
# Function for add line chart in timeseries plot
def add_line(fig,data,name,label,color):
    x_start = data.index[0]
    x_end = data.index[-1]
    
    # Add line
    fig.add_trace(go.Scatter(
        x = data.index, 
        y = data[name],
        name = label,
        line= dict(width = 3, 
                   color = color), 
        line_shape = 'spline'
        ),
    )
    # Add start point
    fig.add_trace(
        go.Scatter(
            x = np.array(x_start),
            y = np.array(data.loc[x_start,name]),
            mode='markers',
            marker = dict(size=10, 
                          color= color
                     ),
        ),
    )
    # Add end point
    fig.add_trace(
        go.Scatter(
            x = np.array(x_end + 1.5), 
            y = np.array(data.loc[x_end,name]),
            mode = 'markers',
            marker = dict(size=10, 
                          color= color
                    ),
        ),
    )
    return fig

# Function for custom plot template
def fig_template(fig):
    fig.update_layout(
        width=700,
        xaxis = dict(
            showline = True,
            showgrid = False,
            showticklabels = True,
            linecolor = 'rgb(204, 204, 204)',
            linewidth = 2,
            ticks = 'outside',
            tickfont = dict(
                family = 'Arial',
                size = 12,
                color = 'rgb(82, 82, 82)',
            ),
        ),
        yaxis = dict(
            showgrid = False,
            zeroline = False,
            showline = False,
            showticklabels = False,
        ),
        autosize = False,
        margin = dict(
            autoexpand=False,
            l = 100,
            r = 20,
            t = 110,
        ),
        showlegend = False,
        plot_bgcolor = 'white'
    )
    
    return fig

# Function for reset annotation

def reset_annot():
    global annotations
    annotations = []
    
# Function for annotating plot 
def annotate(fig, y_data, labels):
    # Labeling the left_side of the plot
    annotations.append(dict(
        xref = 'paper', 
        x = 0.05, 
        y = y_data[0],
        xanchor = 'right', yanchor = 'middle',
        text = labels +' {:.0f}%'.format(y_data[0]),
        font = dict(
            family = 'Arial',
            size = 16
        ),
        showarrow = False
        
    ),
    )
    # Labeling the right_side of the plot
    annotations.append(dict(
        xref = 'paper', 
        x = 0.95, 
        y = y_data[-1],
        xanchor = 'left', yanchor = 'middle',
        text = '{:.0f}%'.format(y_data[-1]),
        font = dict(
            family = 'Arial',
            size = 16
        ),
        showarrow = False
    ),
    )
    
# Function for add title in plot
def add_title(fig, title, size = 30):
    annotations.append(dict(
        xref = 'paper', yref = 'paper', 
        x = 0.0, y = 1.05,
        xanchor = 'left', yanchor='bottom',
        text = title,
        font = dict(
            family='Arial',
            size = size,
            color = 'rgb(37,37,37)'
        ),
        showarrow = False
    ),
    )

# Function for add text in plot 
def add_text(fig, text):
    annotations.append(dict(
        xref = 'paper', yref = 'paper', 
        x = 0.5, y = -0.1,
        xanchor = 'center', yanchor = 'top',
        text = text,
        font = dict(
            family = 'Arial',
            size = 12,
            color = 'rgb(150,150,150)'
        ),
        showarrow = False
    ),
    )
    
# Timeseries of Global Forest Area 
# Aggregate all country 
global_forest = get_global_data(cleaned_forest, 'Forest Area (%)')
y_data = global_forest['Forest Area (%)'].values
annual_decrease = get_avg_per_year(global_forest)[0]

# Creating plot
fig = go.Figure()
fig = fig_template(fig)
fig = add_line(fig, global_forest, 'Forest Area (%)', 'Global', 'red')

# Customizing plot
reset_annot()
annotate(fig, y_data, 'Global')
add_title(fig, 'Global Forest Area (1990 - 2020)(%)')
text = 'Average annual decrease: {}%'.format(annual_decrease)
add_text(fig, text)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()


# Timeseries of Gas Emission 
global_emission = get_global_data(cleaned_gas_emission, 'Greenhouse Gas Emissions (%)')
y_data = global_emission['Greenhouse Gas Emissions (%)'].values
annual_decrease = get_avg_per_year(global_emission)[0]

# Creating plot
fig = go.Figure()
fig = fig_template(fig)
fig = add_line(fig, global_emission, 'Greenhouse Gas Emissions (%)','Global', 'red')

# Customizing plot
reset_annot()
annotate(fig, y_data, 'Global')
add_title(fig, 'Global Gas Emissions (1990 - 2020)(%)')
text = 'Average annual increase: {}%'.format(annual_decrease)
add_text(fig, text)

# Generate plot
fig.update_layout(annotations = annotations)
plot(fig, auto_open = True)
reset_annot()

# Exploring loss of forest area in each country
# Manipulating the data
forest_area_difference = []
countries = cleaned_forest.columns

for country in countries:
    forest_area_difference.append(
        cleaned_forest.loc[2020,country] - cleaned_forest.loc[1990,country]
    )
    
forest_area = pd.DataFrame({
    'Country': countries,
    'Forest Area Loss (%)': forest_area_difference
}).sort_values('Forest Area Loss (%)', 
               ascending = 0)

forest_area['Status'] = forest_area['Forest Area Loss (%)'].apply(lambda x : 'Increase' 
                                                                  if x > 0 
                                                                  else 'Decrease'
                                                            )
# Make a barplot
forest_status_barplot = forest_area['Status'].value_counts().reset_index()

forest_status_barplot.rename(
    columns = {
        'index': 'Status',
        'Status': 'Number of countries'
    }, 
    inplace = True
)

fig = px.bar(forest_status_barplot,
             x = 'Status', 
             y = 'Number of countries',
             color = 'Status',
             color_discrete_map = {
                 'Increase': 'green', 
                 'Decrease': 'red'
             }, 
             width = 600,
             title = 'Number of countries by forest area status (1990 - 2020)'
)

fig.update_layout(
    yaxis = dict(
            showgrid = True,
            zeroline = False,
            showline = False,gridcolor='black'
    ),
    plot_bgcolor = 'white'
)

plot(fig, auto_open = True)

# Make a pieplot
forest_status_pieplot = forest_area['Status'].value_counts(normalize = True).reset_index()
forest_status_pieplot['Status'] = forest_status_pieplot['Status'] * 100

forest_status_pieplot.rename(
    columns = {
        'index': 'Status',
        'Status': 'Number of countries (%)'
    }, 
    inplace = True
)

fig = px.pie(
    forest_status_pieplot,
    values='Number of countries (%)', names = 'Status', 
    width = 600,
    title = 'Number of countries by forest area status (1990 - 2020)'
)

fig.update_traces(hoverinfo = 'label+percent', 
                  textfont_size = 20,
                  marker = dict(
                      colors = ['red','green'], 
                      line = dict(
                          color = 'black', 
                          width = 2
                      ),
                  ),
)
plot(fig, auto_open = True)

# Puerto Rico vs Nicara Gua Forest Area
pri_forest = agg_data('PRI')
nic_forest = agg_data('NIC')
y_data = [pri_forest['Forest Area (%)'].values,
          nic_forest['Forest Area (%)'].values]
labels = ['PRI','NIC']

fig = go.Figure()
fig = fig_template(fig)
fig = add_line(fig,pri_forest,'Forest Area (%)', 'Puerto Rico (PRI)','green')
fig = add_line(fig,nic_forest,'Forest Area (%)', 'Nicaragua (NCI)','red')

# Annotating
reset_annot()
for y_trace, label in zip(y_data, labels):
    annotate(fig,y_trace,label)

# Add title    
add_title(fig, 'Puerto Rico vs Nicaragua Forest Area (%)')

fig.update_layout(annotations = annotations)
plot(fig, auto_open=True)
reset_annot()

# Puerto Rico vs Nicaguara Agriculture Land
pri_agriculture = agg_data('PRI')
nic_agriculture = agg_data('NIC')
y_data = [pri_agriculture['Agricultural Land (%)'].values,
          nic_agriculture['Agricultural Land (%)'].values]
labels = ['PRI','NIC']

fig = go.Figure()
fig = fig_template(fig)
fig = add_line(fig, pri_agriculture, 'Agricultural Land (%)', 'Puerto Rico (PRI)','green')
fig = add_line(fig, nic_agriculture, 'Agricultural Land (%)', 'Nicaragua (NCI)','red')

# Annotating
reset_annot()

for y_trace, label in zip(y_data, labels):
    annotate(fig,y_trace,label)

# Add title    
add_title(fig, 'Puerto Rico vs Nicaragua Agriculture Land (%)',25)

fig.update_layout(annotations = annotations)
plot(fig, auto_open=True)
reset_annot()


# Puerto Rico Urban Population
pri_emission = agg_data('PRI')
y_data = pri_emission['Urban Population (%)'].values

fig = go.Figure()
fig = fig_template(fig)
fig = add_line(fig, pri_emission, 'Urban Population (%)', 'Puerto Rico (PRI)','green')

# Annotating
reset_annot()

# Add title    
add_title(fig, 'Puerto Rico Urban Population (%)')

fig.update_layout(annotations = annotations)
plot(fig, auto_open = True )
reset_annot()

# Nicaragua heatmap
nic_heatmap = px.imshow(
    corr_table('NIC'),
    text_auto = True, 
    color_continuous_scale='RdYlGn',
    title = '<b>Nicaragua</b>'
)

nic_heatmap.update_layout(
    width = 650,
    title_x = 0.55,
    font = dict(
            family = 'Arial',
            size = 14,
            color = 'black'
    )
)

plot(nic_heatmap, auto_open=True)

# Puerto Rico heatmap
nic_pri = px.imshow(
    corr_table('PRI'),
    text_auto = True,
    color_continuous_scale = 'RdYlGn',
    title = '<b>Puerto Rico</b>')

nic_pri.update_layout(
    width = 650,
    title_x = 0.55,
    font = dict(
            family = 'Arial',
            size = 14,
            color = 'black'
    )
)

plot(nic_pri, auto_open = True)



