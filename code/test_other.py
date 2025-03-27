# 用pygal生成趋势图示例（需提前准备数据）
import pygal
chart = pygal.Line()
chart.title = 'XAI论文增长趋势'
chart.x_labels = ['2018','2019','2020','2021','2022']
chart.add('ICML', [45, 67, 112, 158, 203])
chart.add('NeurIPS', [32, 55, 89, 143, 187])
chart.render_to_file('trend.svg')