from django.urls import path

from . import views



app_name = 'demo'
urlpatterns = [
    path('', views.index, name='index'),
    path('searchform/', views.searchForm, name='searchForm'),
    path('search/', views.search, name='search'),
    path('results/', views.results, name='results'),
    path('results_graph/', views.results_graph, name='results_graph'),
    path('search/api/graphdata', views.graph_data, name='search-api-graphdata'),
    path('search/api/expand', views.expand_with_neighbors, name='search-api-expand'),
    path('search/api/train', views.train, name='search-api-train'),
    path('api/search/', views.demoSearch, name='api-search'),
    path('boost/', views.boost, name='boost'),
    path('demo/', views.demov1, name='demov1'),
]

#path('<slug:keywords>/',views.search, name='search'),
