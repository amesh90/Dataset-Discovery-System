from django.shortcuts import render


# Create your views here.

from django.http import HttpResponse
from django.http import JsonResponse
from django.template import loader


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
	val = demoscript.dataDicoveryDemo_main(inputKeywords,'1')
	# return results(request, inputKeywords, val)
	return results_graph(request, inputKeywords, val)
	#return HttpResponse("You're searching with those keywords %s." % val['3058248119'])

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

	for r in values:
		result.append(values[r])
	
	context = {'values': result,}
	return render(request, 'demo/results_graph.html', context)

def graph_data(request): #, *args,  **kwargs):
	data = {
	"sales" : 10,
	"customer": 100,
	}
	return JsonResponse(data)

def boost(request):
	context = {'results': '',}
	return render(request, 'demo/boost.html', context)