"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_table
from sklearn import datasets
import plotly.express as px
import pandas as pd
import loss_and_train as lt

# import stylesheets
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

# creat an app for page 3: data visualisation tool
# load the dataset
bikesharing = pd.read_csv('./data/SeoulBikeData.csv', encoding='unicode_escape')

bs_columns_chinese = ['日期', '单车租借数', '时刻(24小时)', '温度(°C)',
                      '湿度(%)', '风速(m/s)', '可见度(10m)',
                      '体感温度(°C)', '日照强度(MJ/m2)',
                      '降水量(mm)', '降雪量(cm)', '季节',
                      '是否假期', '是否正常运转']
bikesharing.columns = bs_columns_chinese  # prepare the dataset

fig_options = {
    '箱型图': ['季节', '时刻(24小时)'],
    '散点图': ['温度(°C)', '湿度(%)',
            '风速(m/s)', '可见度(10m)', '体感温度(°C)', '日照强度(MJ/m2)',
            '降水量(mm)', '降雪量(cm)']
}

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("图形种类"),
                dcc.Dropdown(
                    id="figure-choice",
                    options=[
                        {"label": col, "value": col} for col in fig_options.keys()
                    ],
                    value="箱型图",
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("X 轴元素"),
                dcc.Dropdown(
                id="bar-x-variable",
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Y 轴元素"),
                dcc.Dropdown(
                id="bar-y-variable",
                options=[
                    {'label':'单车租借数', 'value':'单车租借数'}
                ],
                value='单车租借数',
                )
            ]
        ),
    ],
    body=True,
)

@app.callback(
    Output('bar-x-variable', 'options'),
    Input('figure-choice', 'value'))
def set_x_options(selected_figure):
    return [{'label': i, 'value': i} for i in fig_options[selected_figure]]


@app.callback(
    Output('bar-x-variable', 'value'),
    Input('bar-x-variable', 'options'))
def set_x_values(available_options):
    return available_options[0]['value']

@app.callback(
    Output("bike-graph", "figure"),
    [
        Input("figure-choice", "value"),
        Input("bar-x-variable", "value"),

    ],
)
def make_graph(figchoice, xvariable):
    if figchoice == "箱型图":
        fig = px.box(bikesharing, x=xvariable, y='单车租借数', color='是否假期')
        return fig
    else:
        fig = px.scatter(bikesharing, x=xvariable,
                         y='单车租借数',
                         facet_col="季节", facet_col_wrap=2)
        return fig


# creat another data app for page 4, which visualises the linear regression
# load the dataset
tipdata = px.data.tips()
advdata = pd.read_csv('./data/apple_store.csv')
tipcolumns_chinese = ['消费额度(美元)', '小费数额', '性别', '是否吸烟', '星期', '时段', '人数']
tipdata.columns = tipcolumns_chinese

#
reg_controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("选择数据集"),
                dcc.Dropdown(
                    id='reg-data-choice',
                    options=[
                        {"label": col, "value": col} for col in ('广告销售', '餐饮服务')
                    ],
                    value="广告销售",
                )
            ]
        )
    ]
)

@app.callback(
    Output('reg-data-table', 'children'),  # the output should always be children
    Input('reg-data-choice', 'value'))
def reg_render_table(table_choice):
    if table_choice == '餐饮服务':
        return [dash_table.DataTable(
            data=tipdata.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in tipdata.columns],
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
)]
    elif table_choice == "广告销售":
        return [dash_table.DataTable(
            data=advdata.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in advdata.columns],
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        )]

@app.callback(
    Output('reg-scatter-plot', 'figure'),
    Input('reg-data-choice', 'value'))
def reg_render_figure(data_choice):
    if data_choice == '餐饮服务':
        figure = px.scatter(tipdata, x='消费额度(美元)', y='小费数额', color='性别',
                            title='顾客就餐消费与给出小费额度的散点图', trendline='ols')
        return figure
    elif data_choice == '广告销售':
        figure = px.scatter(advdata, x='广告费(元)', y='苹果销量(斤)', title='水果店广告费用与苹果售量散点图',
                            trendline='ols', trendline_color_override='#E2141E')
        return figure


# creat a slider to control training session and use different colors to compare learning rate
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return '你选择机器训练的次数为{}次'.format(value)


# extract X and Y as matrix and add constant vector
advdata_x = np.asmatrix(advdata.iloc[:, 1].astype(float)).transpose()/100  # scale down 100
advdata_y = np.asmatrix(advdata.iloc[:, 2].astype(float)).transpose()
advdata_matrix_x = np.hstack([np.ones([advdata_x.shape[0], 1]), advdata_x])
advdata_theta = np.zeros([2, 1])  # initialize theta


# render the regression line
@app.callback(
    Output('train-reg-figure', 'figure'),
    Input('my-slider', 'value'))
def render_reg_figure(train_iteration):
    adv_parameters, adv_cost_history, adv_hs = lt.gradient_descent(
        advdata_matrix_x, advdata_y, advdata_theta, 0.01, train_iteration
    )
    adv_parameters[1, 0] = adv_parameters[1, 0]/100
    adv_hs = np.array(adv_hs).reshape(-1, 2)
    adv_hs[:, 1] = adv_hs[:, 1] / 100
    adv_reg_x = np.linspace(0, 1000, 1000).reshape([1000, 1])
    adv_reg_y = adv_reg_x * adv_parameters[1] + adv_parameters[0]
    adv_reg_x = adv_reg_x.flatten()
    adv_reg_y = np.asarray(adv_reg_y).flatten()
    fig = px.scatter(advdata, x='广告费(元)', y='苹果销量(斤)', title='线性回归模型训练动态图')
    fig.add_traces(go.Scatter(x=adv_reg_x, y=adv_reg_y, name='线性回归模型'))
    fig.update_layout(yaxis_range=[-1, 550])
    return fig

@app.callback(
    Output('train-error-figure', 'figure'),
    Input('my-slider', 'value'))
def render_error_figure(train_iteration):
    adv_parameters, adv_cost_history, adv_hs = lt.gradient_descent(
        advdata_matrix_x, advdata_y, advdata_theta, 0.01, train_iteration
    )
    fig = go.Figure(data=go.Scatter(x=np.linspace(0, train_iteration, train_iteration),
                                    y=adv_cost_history[0:train_iteration, 0]))
    fig.update_layout(title='误差动态统计图',
                      xaxis_title='训练次数',
                      yaxis_title='误差计算')
    return fig


#################### HTML Format Section #############################################################################

# creat a sidebar and design the main layout of html
sidebar = html.Div(
            children=[
                html.Nav([
                    html.H5([html.A("人工智能体验课", href="/")],
                            style={'color': '#000000',
                                   'padding': '3px',
                                   'text-align': 'center'}),
                    html.Ul([
                        html.A("理论框架", href="/page-1"),
                        html.A("共享单车-数据", href="/page-2"),
                        html.A("共享单车-可视化", href="/page-3"),
                        html.A("单变量线性回归", href="/page-4"),
                        html.A("简单的线性分类", href='/page-5'),
                        html.A("共享单车-模型", href="/page-6")
                    ])
                ])
            ],
            className='sidenav'
        )

app.layout = html.Div(
    [dcc.Location(id='navurl', refresh=False),
     sidebar,
     html.Div(id='page-content')
    ]
)


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('navurl', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return html.Div(
            [
                html.Div(
                    children=[
                        html.P("人工智能好像在向生活中的各个方面渗透，目前的应用主要依托以下两个领域的研究。在这一节的体验课中，我们将会首先体验机器学习中"
                       "一个较为常用的模型-线性回归模型。在理解机器学习的简单模型之后，我们将会带领大家去体验深度神经学习。届时，"
                       "你将会从0开始，通过图形化的方式来搭建你的第一个'神经网络'。"),
                        html.Div([
                    html.Div([
                        dcc.Graph(
                            id='home-graph1',
                            figure={
                                'data': [
                                    {'x': ['数据','可视化','模型与算法'], 'y': [1, 2, 5], 'type': 'bar', 'name': '一步一脚印'},
                                ],
                                'layout': {
                                    'title': '机器学习三步走'
                                }
                            }
                        )
                    ],
                        className="six columns"),
                    html.Div([
                        dcc.Graph(
                            id='home-graph2',
                            figure={
                                'data': [
                                    {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'y': [0, 1, 4, 9, 16, 25, 36, 49, 64],
                                     'type': 'scatter', 'name': 'trend'},
                                ],
                                'layout': {
                                    'title': '深度学习的曲线可以很长'
                                }
                            }
                        )
                    ],
                        className="six columns")],
                            className="row"),
                        html.Br(),
                        html.P("现在就让我们开始体验吧，请点击左边的目录栏。")
                    ],
                    className='main'
                )
            ]
        )
    elif pathname == '/page-1':
        return html.Div(
            [
                html.Div(
                    children=[
                        html.P("随着我们进入了数字时代，生活中的方方面面都会关联着无穷无尽的数据。这些数据一方面记录着我们的生活，"
                               "一方面也帮助我们做出更好的决策，从而为我们提供更为便捷地生活方式。"),
                        html.P("你想不想知道，网店老板在搜集了数据之后是如何进行广告投放的？你想不想知道各类APP是如何根据用户"
                               "特征进行‘用户画像’分类的？你想不想知道智慧城市管理者是如何利用大数据进行车辆投放的？"
                               "如果给你很多数据，你该怎样去预测房价呢？机器学习算法能不能帮助我们防止森林火灾？"
                               "在这节机器学习体验课，我们将带你去洞察数据背后的故事并体验机器学习中一些简单的算法。"),
                        html.P("在正式开始我们的体验课之前，我们先要铺垫一个简单的理论框架，"
                               "即我们每一个案例都会有以下四个部分组成:"),
                        html.Ul(
                            children=[html.Li("数据: 记录和描述着我们的生活"),
                                      html.Li('可视化: 呈现简单的规律'),
                                      html.Li('模型与算法: 寻找规律的工具箱'),
                                      html.Li('结果和预测: 指导生活的参考性建议')],
                            style={
                                'textAlign': 'left'
                            }
                        ),
                        html.Div([
                            html.Div([
                                html.Center(html.Img(src=app.get_asset_url('mlbanner.png'),
                                         style={'width':"80%",
                                                'padding-top': '36px'}))
                            ],
                            className='twelve columns')
                        ],
                            className='row')
                    ],
                    className='main'
                )

            ]
        )
    elif pathname == '/page-2':
        return html.Div(
            [
                html.Div(
                    children=[
                        html.H6("那些传说中的大数据"),
                        html.P("下面是韩国首都首尔市共享单车的数据。目前我们收集的数据集有" +
                               str(bikesharing.shape[0]) + "行" +
                               str(bikesharing.shape[1]) + "列。你可以下滑感受一下这个数据的大小。"
                                                           "在大数据时代，样本数成千上万已经很常见，"
                                                           "很多公司收集的数据，随便就可以达到几千万行。"),
                        html.Div([
                            html.Div([
                                dash_table.DataTable(
                                    id='table1',
                                    data=bikesharing.to_dict('records'),
                                    columns=[{"name": i, "id": i} for i in bikesharing.columns],
                                    page_size=100,  # we have less data in this example, so setting to 20
                                    style_table={'height': '600px', 'overflowY': 'auto'}
                                )
                            ],
                            className="ten columns",
                            style={'font-size':"14px"})
                        ],
                        className='row')
                    ],
                    className='main'
                )

            ]
        )
    elif pathname == '/page-3':
        return html.Div(
            [
                html.Div(
                    children=[
                        html.H6("一画胜千言"),
                        html.P(
                            "上一节中，你已经体会到了数据的规模，应该也能感受到它的杂乱。"
                            "人工智能专家在创建机器学习模型时，第一步往往是通过简单的数据可视化来呈现一些比较明显的规律。"
                            "下面，你可以通过左边的控制面板，来选择你喜欢的图形样式，并选取想要探索的元素，来生成具体的图形。"),
                        html.Div([
                            html.Div([
                                controls
                            ],
                            className='four columns',
                            style={'padding-top': '80px'}),
                            html.Div([
                                dcc.Graph(id="bike-graph")
                            ],
                            className='eight columns',
                            style={'item-align':'center'})
                        ],
                        className='row'),
                        html.P(["当你在尝试不同的图形时，是否发现了一些有趣的现象呢？如果现在让你去管理城市的共享单车，怎样才能避免"
                                "出现下面这样的情况呢？"]),
                        html.Center(html.Img(src=app.get_asset_url('wastedbike.jpg'),
                                 style={'width': '50%',
                                        'padding-top': '20px'}))

                    ],
                    className='main'
                )

            ]
        )
    elif pathname == '/page-4':
        return html.Div(
            [
                html.Div(
                    children=[
                        html.H6("那些常见的线性规律"),
                        html.P(
                            "很多人在选择 Machine Learning (机器学习) 这门课程时，往往都是从线性回归模型开始的。"
                            "本节中，我们将会通过简单的单变量线性回归模型，让大家体验机器学习的过程。"),
                        html.P("线性回归模型假定了要描述的变量之间，存在线性关系。那什么是线性关系呢？请同学们在下面的图形中，"
                               "选择不同的数据集，来观察右边可视化的结果，并思考: 这些数据之间表现出来的简单的规律是什么？"),
                        html.Br(),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        reg_controls,
                                        html.Br(),
                                        html.Div(id='reg-data-table'),
                                    ],
                                    className='four columns'
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id='reg-scatter-plot')
                                    ],
                                    className='eight columns'
                                )
                            ],
                            className='row'
                        ),
                        html.P('在收集了以上数据之后，我们想要从这些数据中总结出一些简单的规律，从而帮助我们做出更好的判断。'
                               '比如: 1) 水果店老板可以根据上面的数据来研判应该投放多少广告； 2）咨询公司可以帮助银行来研判'
                               '是否应该给餐厅老板继续贷款； 等等。'),
                        html.P("在人工智能领域，你经常听到'监督式学习'和'非监督式学习',以线性回归为依托的学习过程，就是一个典型的"
                               "监督式学习案例: 即最初由人类来设定相应的模型(这里我选择了最简单的线性回归模型),之后让计算机按照一定"
                               "的算法去寻找最优的模型参数。下面我就来以'广告销售'的数据集案例，来体验机器学习的训练过程。"),
                        html.H6("机器学习的训练过程：以线性回归为例"),
                        html.P("首先请同学们，使用鼠标点击上面散点图中不同的点，查看具体的数据值；然后，在找到相应广告费用位置，"
                               "查看红色回归线上具体的数值。下面的图片中，选取了x=596的位置。"),
                        html.Center([
                            html.Img(src=app.get_asset_url('errorfigure2.png'), style={'width': '60%'})
                        ]),
                        html.P("不难注意到，红色回归线上计算的数值给真实的数值有差异，这个差异我们称之为'误差'。那么，机器学习模型"
                               "就是要通过算法来寻找误差最小的直线去与数据对应。其具体的过程为:"),
                        html.Ul([
                            html.Li("随机选择参数a, b 生成直线 y = ax + b"),
                            html.Li('计算现有直线与所有数据之间的误差总和'),
                            html.Li('利用梯度下降法来计算新的参数a, b'),
                            html.Li('利用新计算的参数a, b 再一次计算误差，并回到上一步，再次使用梯度下降法'),
                            html.Li('其中梯度下降法，可以保证每一次循环误差都会下降，而且多数情况下(需要数学去论证的部分)保证会逼近'
                                    '最优化结果'),
                            html.Li("另外，我们将梯度下降法每一次循环时逼近的速率称为机器学习的'学习率', 标注为alpha")
                        ]),
                        html.Br(),
                        html.P("接下来，我们就来体验这个过程。请使用鼠标滑动选择你要机器训练的次数:"),
                        html.Center(
                            [
                                html.Br(),
                                dcc.Slider(
                                    id='my-slider',
                                    min=10,
                                    max=500,
                                    step=5,
                                    value=100,
                                ),
                                html.Div(id='slider-output-container'),
                            ]
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(id='train-error-figure')
                                    ],
                                    className="five columns"
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id='train-reg-figure')
                                    ],
                                    className="seven columns"
                                )
                            ],
                            className='row'
                        )
                    ],
                    className='main'
                )

            ]
        )
    elif pathname == '/page-5':
        return html.Div(
            [
                html.Div(
                    [
                        html.H6('简单的线性分类'),
                        html.P("上一节中，我们使用的线性方程Y = a x + b建模后，其描述的场景可以用图(a)表示。类似于线性回归模型，"
                               "线性分类模型依托了的方程为sigmoid方程，对该方程我们不做过多介绍，只是把它的图像在图(b)中"
                               "展现给大家。"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            figure={
                                                'data': [
                                                    {'x': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                     'y': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                                     'type': 'line', 'name': 'trend'},
                                                ],
                                                'layout': {
                                                    'title': '(a)简单的直线方程'
                                                }
                                            }
                                        )
                                    ],
                                    className='five columns'
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            figure={
                                                'data': [
                                                    {'x': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                                                     'y': [0.0066928509242848554, 0.01798620996209156,
                                                           0.04742587317756678, 0.11920292202211755, 0.2689414213699951,
                                                           0.5, 0.7310585786300049, 0.8807970779778823,
                                                           0.9525741268224334, 0.9820137900379085, 0.9933071490757153],
                                                     'type': 'line', 'name': 'trend'},
                                                ],
                                                'layout': {
                                                    'title': '(b) Sigmoid 方程',
                                                    "xaxis": {"title": "X"},
                                                    "yaxis": {"title": "sigmoid(X)"}
                                                }
                                            }
                                        )
                                    ],
                                    className='five columns'
                                )
                            ],
                            className='row'
                        ),
                        html.H6("为什么需要sigmoid方程？")
                    ],
                    className='main'
                )

            ]
        )















if __name__=="__main__":
    app.run_server(debug=True, port=5001)
