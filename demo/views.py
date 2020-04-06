from django.shortcuts import render


# Create your views here.

from django.http import HttpResponse
from django.http import JsonResponse
from django.template import loader
from django.views.generic import View


from . import demoscript

# def index(request):
#     output = ['Amira', 'Ahmed', 'Shawky', 'Mohamed']
#     template = loader.get_template('demo/index.html')
#     context = {
#         'output': output,
#     }
#     return HttpResponse(template.render(context, request))
    
#     #return HttpResponse(output)
#     #return HttpResponse("Hello, world. You're at the demo index.")

tmpgraph_path = '/Users/princess/Documents/PhD/thesis/code/djangoProject/discoverysite/demo/static/data/'

def index(request):
	context = {'results': '',}
	return render(request, 'demo/searchForm.html', context)

def searchForm(request):
    template = loader.get_template('demo/searchForm.html')
    context = {}
    return HttpResponse(template.render(context, request))
    #return HttpResponse(output)
    #return HttpResponse("Hello, world. You're at the demo index.")

def search(request):
	inputKeywords = request.POST['keywords']
	
	# ----- 
	# simple return table 
	# -----
	# val = demoscript.dataDicoveryDemo_main(inputKeywords,'1')
	# return results(request, inputKeywords, val)
	
	# ----- 
	# return gragh 
	# -----
	valjson = demoscript.dataDicoveryDemo_main_graph(inputKeywords,'3', tmpgraph_path)
	return results_graph(request, inputKeywords, valjson)

	
	
		
class demoSearch(View):
	gdatajson = {}
	keywords = 'purdue'
	
	def get(self, request): #, *args,  **kwargs):
		gdatajson = demoscript.dataDicoveryDemo_main_graph(keywords,'3',tmpgraph_path)
		context = {'values': [],}
		return render(request, 'results_graph.html', context)

# =====================
# 	List the results in a table
# 	tabular view 
# =====================
def results(request, keywords, values): 
	result = []
	for r in values:
		result.append(values[r])
	
	context = {'values': result,}
	return render(request, 'demo/results.html', context)
	#return HttpResponse("You're searching with those keywords %s." % values['3058248119'])	

def results_graph(request,keywords='', values=''): 
	result = []

	# for r in values:
	# 	result.append(values[r])
	
	context = {'values': result,}
	return render(request, 'demo/results_graph.html', context)

def graph_data(request):
	
	keywords = request.POST['keywords']

	print ('\ngraph_data function\n')
	gdatajson = {}
	gdatajson = demoscript.dataDicoveryDemo_main_graph(keywords,'4', tmpgraph_path)
		
	return JsonResponse(gdatajson) #, safe=False)

def expand_with_neighbors(request):
	print ('\nexpand_with_neighbors function\n')
	gdatajson = {}
	gdatajson = demoscript.expand_with_neighbors('4', tmpgraph_path)
	
	print('\n\nExpanded Graph:')
	print(gdatajson)
	return JsonResponse(gdatajson)
	

def boost(request):
	context = {'results': '',}
	return render(request, 'demo/boost.html', context)

def demov1(request):
	context = {'results': '',}
	return render(request, 'demo/demov1.html', context)
