from django.shortcuts import render


# Create your views here.

from django.http import HttpResponse
from django.http import JsonResponse
from django.template import loader
from django.views.generic import View


#from . import demoscript # old aurum file 
from . import demoscript2 as demoscript # new 2aurum file
from . import active_demo

import numpy as np

# ================
# Global VAriables

tmpgraph_path = '/Users/princess/Documents/PhD/thesis/code/djangoProject/discoverysite/demo/static/data/'
debug = 1

nids_keyword_Search = {}
# modelobj = None
# train_cnt = 1

# ===============
# Function views

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

	print ('\ngraph_data ()\n')
	gdatajson = {}
	result = demoscript.dataDicoveryDemo_main_graph(keywords,'4', tmpgraph_path)
	gdatajson = result['g_json']

	if debug == 1:
		print(result)
	# nids_keyword_Search = request.session.get("nids_keyword_Search")
	# if not nids_keyword_Search:
		# nids_keyword_Search =
	global nids_keyword_Search
	nids_keyword_Search = result['nids']

	global train_cnt
	train_cnt = 1
	global modelobj
	modelobj = None

	# if debug == 1:
	# 	print()
	# 	print(nids_keyword_Search)

	return JsonResponse(gdatajson) #, safe=False)

def expand_with_neighbors(request):
	print ('\nexpand_with_neighbors function\n')
	gdatajson = {}
	gdatajson = demoscript.expand_with_neighbors('4', tmpgraph_path)
	
	print('\n\nExpanded Graph:')
	print(gdatajson)
	return JsonResponse(gdatajson)
	
def train(request):
	print ('\ntrain(): \n')
	

	# create model
	# modelobj = request.session.get("modelobj")
	global modelobj
	global train_cnt
	global nids_keyword_Search

	if debug == 1:
		print()
		print(train_cnt)
		print(nids_keyword_Search)

	
	if train_cnt == 1: # first learning iteration
		train_cnt += 1
		modelobj = active_demo.activelearning()

		# relevant nids
		# nids_keyword_Search = request.session.get("nids_keyword_Search")
		nids = list(nids_keyword_Search.keys())
		k_samples_nids = np.array(nids)
		labels = np.ones(len(nids))

		# initialize the learning model, prepare the graph
		
		modelobj.train_nids = k_samples_nids.tolist()	
		modelobj.train_labels = labels.tolist()
		modelobj.mainlogic('4', k_samples_nids,labels )

		selected_nids = []
	else: # learning iteration

		if debug == 1:
			print('\n==================\n')
			print('Re-Train .... Start ')
			print()
		# ==================
		# get the selection from the oracle (user)
		relevence = request.POST.get('relevence', '')
		relevence_nids = relevence.split(',')
		
		relevent = []
		notRelevent = []
		notrel = 0
		for i in relevence_nids:
			if 'not' in i:
				notrel = 1
				continue
			if notrel == 0:
				try:
					relevent.append(int(i))
				except Exception as e:
					print(str(e))
			else:
				try:
					notRelevent.append(int(i))
				except Exception as e:
					print(str(e))


		if debug == 1:
			print('===============================')
			print('relevent: ')
			print (relevent)
			print('\nnotRelevent: ')
			print (notRelevent)
			print('===============================')


		# =======================
		# prepare for another learning iteration

		k_samples_nids = modelobj.train_nids 
		labels = modelobj.train_labels 
		
		if debug == 1:
			print('Old k_samples_nids: ')
			print (k_samples_nids)
			print (labels)
			print()

			
		# k_samples_nids = []
		# labels = []

		for i in relevent:
			if i in k_samples_nids:
				# update label
				idx_o = k_samples_nids.index(i)
				labels[idx] = 1
			else:
				k_samples_nids.append(i)
				labels.append(1)

		for i in notRelevent:
			if i in k_samples_nids:
				# update label
				idx_o = k_samples_nids.index(i)
				labels[idx] = 0
			else:
				k_samples_nids.append(i)
				labels.append(0)

		k_samples_nids = np.array(k_samples_nids)
		labels = np.array(labels)

		if debug == 1:
			print('New k_samples_nids: ')
			print (k_samples_nids)
			print (labels)
			print()
		# ===========
		# do training
		modelobj.oneiter(k_samples_nids, labels)


	# select nodes to be labeled by oracle
	selected_nids =  modelobj.ask_oracle(10)
	selected_nids_data = demoscript.get_data('4', selected_nids)
	
	# return the selected nodes for manual labeling 
	# return the train, loss accuracy
	# update the returned graph to include all the relevant nodes after first training iteration
		

	if debug == 1:
		print()
		print("train(): ")
		print(train_cnt)
		print(k_samples_nids)
		print(labels)
		print(selected_nids)

	# gdatajson = demoscript.expand_with_neighbors('4', tmpgraph_path)
	gdatajson = {}

	relevant_nids = modelobj.get_relevant_nodes()

	if debug == 1:
		print()
		print('train(): relevant_nids')
	gdata = demoscript.snapshotOfGraph_std('4', relevant_nids, tmpgraph_path)

	tables = []
	for i in gdata['nids']:

		if gdata['nids'][i]['source_name']  not in tables:
			tables.append(gdata['nids'][i]['source_name'] )


	result = {}
	result['graph'] = gdata['graph']
	result['nids_for_label'] = selected_nids_data
	result['training_acc'] = modelobj.train_acc.item()
	result['tables'] = ';'.join(tables)
	return JsonResponse(result)


def boost(request):
	context = {'results': '',}
	return render(request, 'demo/boost.html', context)

def demov1(request):
	context = {'results': '',}
	return render(request, 'demo/demov1.html', context)
