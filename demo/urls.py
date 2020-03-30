from django.urls import path

from . import views



app_name = 'demo'
urlpatterns = [
    path('', views.index, name='index'),
    path('searchform/', views.searchForm, name='searchForm'),
    path('search/', views.search, name='search'),
    path('results/', views.results, name='results'),
    path('results_graph/', views.results_graph, name='results_graph'),
    path('api/graph_data', views.graph_data, name='api-graph-data'),
    path('boost/', views.boost, name='boost'),
]

#path('<slug:keywords>/',views.search, name='search'),
