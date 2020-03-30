import sys
sys.path.insert(0, '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master')

debug=1
argumentPassing=1

import sys
from main import init_system
from collections import namedtuple
from modelstore.elasticstore import KWType
from api.apiutils import Relation
# from algebra import API, DRS
from mock import MagicMock, patch
from ddapi import API
from modelstore.elasticstore import StoreHandler
from knowledgerepr.fieldnetwork import deserialize_network

from collections import defaultdict
from collections import OrderedDict
import itertools
from DoD.utils import FilterType
# from DoD import DoD
# import DoD

# import run_dod

def virtual_schema_exhaustive_search(api, list_attributes: [str], list_samples: [str]):
    # Align schema definition and samples
    assert len(list_attributes) == len(list_samples)
    sch_def = {attr: value for attr, value in zip(list_attributes, list_samples)}

    # Obtain sets that fulfill individual filters
    filter_drs = dict()
    for attr in sch_def.keys():
        drs = api.search_attribute(attr)
        filter_drs[(attr, FilterType.ATTR)] = drs

    for cell in sch_def.values():
        drs = api.search_content(cell)
        filter_drs[(cell, FilterType.CELL)] = drs

    # We group now into groups that convey multiple filters.
    # Obtain list of tables ordered from more to fewer filters.
    table_fulfilled_filters = defaultdict(list)
    for filter, drs in filter_drs.items():
        drs.set_table_mode()
        for table in drs:
            table_fulfilled_filters[table].append(filter)
    # sort by value len -> # fulfilling filters
    a = sorted(table_fulfilled_filters.items(), key=lambda el: len(el[1]), reverse=True)
    table_fulfilled_filters = OrderedDict(
        sorted(table_fulfilled_filters.items(), key=lambda el: len(el[1]), reverse=True))

    # Find all combinations of tables...
    # Set cover problem, but enumerating all candidates, not just the minimum size set that covers the universe
    candidate_groups = set()
    num_tables = len(table_fulfilled_filters)
    while num_tables > 0:
        combinations = itertools.combinations(list(table_fulfilled_filters.keys()), num_tables)
        for combination in combinations:
            candidate_groups.add(frozenset(combination))
        num_tables = num_tables - 1

    # ...and order by coverage of filters
    candidate_group_filters = defaultdict(list)
    for candidate_group in candidate_groups:
        filter_set = set()
        for table in candidate_group:
            for filter_covered_by_table in table_fulfilled_filters[table]:
                filter_set.add(filter_covered_by_table)
        candidate_group_filters[candidate_group] = list(filter_set)
    candidate_group_filters = OrderedDict(
        sorted(candidate_group_filters.items(), key=lambda el: len(el[1]), reverse=True))

    # return candidate_group_filters #table_fulfilled_filters
    # Now do all-pairs join paths for each group, and eliminate groups that do not join (future -> transform first)
    joinable_groups = []
    for candidate_group, filters in candidate_group_filters.items():
        if len(candidate_group) > 1:
            join_paths = self.joinable(candidate_group)
            if join_paths > 0:
                joinable_groups.append((join_paths, filters))
        else:
            joinable_groups.append((candidate_group, filters))  # join not defined on a single table, so we add it

    return joinable_groups

def main():
	
    #========================================== 
    # initialization
    #========================================== 
    path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel1/'
    
    # ddapi:
    network = deserialize_network(path)
    print("\n\n====================\nVisulaize Graph\n=====================\n\n")
    network._visualize_graph()

    return

    api = API(network)
    api.init_store()

    # algebra
    # network = deserialize_network(path)
    # store_client = StoreHandler()
    # api = API(network=network, store_client=store_client)

    #========================================== 
    print ("press 'h' for help\n'quit' for exit\n")
    while True:
        
        inputStr = str(input())
        if(inputStr == 'quit'):
            break
        elif(inputStr == 'h'):
            print("\n====================================\nHelp:\n")
            print("Enter Keywords with comma separated\n")
            print("====================================\n")
        else:
            keywords = inputStr.split(',')
            print("Searching for the keywords:\n")
            print(keywords)
            # print("\n\n")
            print("\n\nSearching ... \n\n")
            
            #==============
            # Module 1: 
            # - input: sample of attributes and values pairs
            # - output: all tables that contain similar values 
            #==============
            
            # list_attributes = ['name', 'type']
            # list_samples = ['Purdue', 'public']
            # results = virtual_schema_exhaustive_search(api, list_attributes, list_samples)
            # print ("results:\n")
            # print(results)

            # OR .... 
            # var = DoD(network, store_client)
            # res = var.virtual_schema_exhaustive_search(list_attributes,list_samples)
            # print(res)

            #==============
            # Module 2:
            #==============
            # for key in keywords:["tables"]=[]
            resultMap = {}
            # resultMap["tables"]=[]
            # resultMap["fields"]={}

            # resultMap["fieldnames"]=[]
            # resultMap["fieldIds"]=[]

            # we can do fuzzy or strict mathcing 
            
            # res = api.keywords_search(keywords, max_results=10)
            res = api.fuzzy_keywords_match(keywords, max_results=10)
            for el in res:
                
                nid,db_name,source_name,field_name,score = el
                
                #=============
                # to get the contents
                content = api.getContentValuesOfIndex(nid)
                #=============

                # print("Source: %s, Field: %s"%(source_name, field_name))
                # print("\n\n======================== Contents \n")
                
                # print(content)
                if nid not in resultMap:
                    resultMap[nid] = {}
                    resultMap[nid]["source_name"] = source_name
                    resultMap[nid]["field_name"] = field_name
                    resultMap[nid]["score"] = score

            
            
            # print("\n\n======================== Entities \n")
            # fields, entities = api.get_all_fields_entities()
            # print(fields)
            # print(entities)
            # print("\n\n======================== SCHEMA \n")
            #TODO: instead of split each keyword and search in the schema, I should search with ontology
            for key in keywords:
                keySplit = key.split(' ')
                for k in keySplit:
                    res2 = api.schema_name_search(k, max_results=10)
                    for el in res2:
                        # print(str(el))
                        nid,db_name,source_name,field_name,score = el
                        if nid not in resultMap:
                            resultMap[nid] = {}
                            resultMap[nid]["source_name"] = source_name
                            resultMap[nid]["field_name"] = field_name
                            resultMap[nid]["score"] = score

        
            print("\n================================\nFinally:\n\n")
            # print (resultMap)
            print("\n\nElements:\n")
            for i in resultMap:
                # print(i)
                print(resultMap[i])
                print("\n")



def snapshotOfGraph(api, resultMap, drs):
    if debug == 1:
        print('\nSnapshotOfGraph ..... Begin')
        print(resultMap)
    resultMapWithProv = drs.__dict__()
    if debug == 1:
        print(str(resultMapWithProv))
        # print(resultMapWithProv['sources'])

# =================
# Web app demo (return snapshot of the resutl graph)
# =================
# =================
# for the Webapp demo
# =================
def dataDicoveryDemo_main(keywords: str, model: str):
    
    #========================================== 
    # initialization
    #========================================== 
    if debug == 1:
        print ('testmode'+model+' is loading!')
    if '1' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel1/'
    elif '2' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel2/'
    elif '3' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel3/'
    keywords = keywords.split(',')

    if debug == 1:
        print (keywords)
    # ddapi:
    network = deserialize_network(path)
    api = API(network)
    api.init_store()

    # algebra
    # network = deserialize_network(path)
    # store_client = StoreHandler()
    # api = API(network=network, store_client=store_client)

    #========================================== 
    # print ("press 'h' for help\n'quit' for exit\n")
    # while True:
        
    #     inputStr = str(input())
    #     if(inputStr == 'quit'):
    #         break
    #     elif(inputStr == 'h'):
    #         print("\n====================================\nHelp:\n")
    #         print("Enter Keywords with comma separated\n")
    #         print("====================================\n")
    #     else:
    #         keywords = inputStr.split(',')
    #         print("Searching for the keywords:\n")
    #         print(keywords)
    #         # print("\n\n")
    #         print("\n\nSearching ... \n\n")
            
            #==============
            # Module 1: 
            # - input: sample of attributes and values pairs
            # - output: all tables that contain similar values 
            #==============
            
            # list_attributes = ['name', 'type']
            # list_samples = ['Purdue', 'public']
            # results = virtual_schema_exhaustive_search(api, list_attributes, list_samples)
            # print ("results:\n")
            # print(results)

            # OR .... 
            # var = DoD(network, store_client)
            # res = var.virtual_schema_exhaustive_search(list_attributes,list_samples)
            # print(res)

    #==============
    # Module 2:
    #==============
    # for key in keywords:["tables"]=[]
    resultMap = {}
    # resultMap["tables"]=[]
    # resultMap["fields"]={}

    # resultMap["fieldnames"]=[]
    # resultMap["fieldIds"]=[]

    # we can do fuzzy or strict mathcing 
    
    # res = api.keywords_search(keywords, max_results=10)
    res = api.fuzzy_keywords_match(keywords, max_results=10)
    for el in res:
        
        nid,db_name,source_name,field_name,score = el
        
        #=============
        # to get the contents
        content = api.getContentValuesOfIndex(nid)
        #=============
        if debug == 1:
            print("Source: %s, Field: %s"%(source_name, field_name))
        # print("\n\n======================== Contents \n")
        
        # print(content)
        if nid not in resultMap:
            resultMap[nid] = {}
            resultMap[nid]["source_name"] = source_name
            resultMap[nid]["field_name"] = field_name
            resultMap[nid]["score"] = score
            resultMap[nid]["nid"] = nid

    
    
    # print("\n\n======================== Entities \n")
    # fields, entities = api.get_all_fields_entities()
    # print(fields)
    # print(entities)
    # print("\n\n======================== SCHEMA \n")
    #TODO: instead of split each keyword and search in the schema, I should search with ontology
    for key in keywords:
        keySplit = key.split(' ')
        for k in keySplit:
            res2 = api.schema_name_search(k, max_results=10)
            for el in res2:
                # print(str(el))
                nid,db_name,source_name,field_name,score = el
                if nid not in resultMap:
                    resultMap[nid] = {}
                    resultMap[nid]["source_name"] = source_name
                    resultMap[nid]["field_name"] = field_name
                    resultMap[nid]["score"] = score
                    resultMap[nid]["nid"] = nid

    if debug == 1:
        print("\n================================\nFinally:\n\n")
        # print (resultMap)
        print("\n\nElements:\n")
        for i in resultMap:
            # print(i)
            print(resultMap[i])
            print("\n")


    # -----------
    # create the snapshot of the graph
    # -----------
    # snapshotOfGraph(api, resultMap, res)

    return resultMap
# =================
# for the Webapp demo ( Keywords + return list of fields as table )
# =================
def dataDicoveryDemo_main(keywords: str, model: str):
    
    #========================================== 
    # initialization
    #========================================== 
    if debug == 1:
        print ('testmode'+model+' is loading!')
    if '1' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel1/'
    elif '2' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel2/'
    elif '3' in model:
        path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel3/'
    keywords = keywords.split(',')

    if debug == 1:
        print (keywords)
    # ddapi:
    network = deserialize_network(path)
    api = API(network)
    api.init_store()

    # algebra
    # network = deserialize_network(path)
    # store_client = StoreHandler()
    # api = API(network=network, store_client=store_client)

    #========================================== 
    # print ("press 'h' for help\n'quit' for exit\n")
    # while True:
        
    #     inputStr = str(input())
    #     if(inputStr == 'quit'):
    #         break
    #     elif(inputStr == 'h'):
    #         print("\n====================================\nHelp:\n")
    #         print("Enter Keywords with comma separated\n")
    #         print("====================================\n")
    #     else:
    #         keywords = inputStr.split(',')
    #         print("Searching for the keywords:\n")
    #         print(keywords)
    #         # print("\n\n")
    #         print("\n\nSearching ... \n\n")
            
            #==============
            # Module 1: 
            # - input: sample of attributes and values pairs
            # - output: all tables that contain similar values 
            #==============
            
            # list_attributes = ['name', 'type']
            # list_samples = ['Purdue', 'public']
            # results = virtual_schema_exhaustive_search(api, list_attributes, list_samples)
            # print ("results:\n")
            # print(results)

            # OR .... 
            # var = DoD(network, store_client)
            # res = var.virtual_schema_exhaustive_search(list_attributes,list_samples)
            # print(res)

    #==============
    # Module 2:
    #==============
    # for key in keywords:["tables"]=[]
    resultMap = {}
    # resultMap["tables"]=[]
    # resultMap["fields"]={}

    # resultMap["fieldnames"]=[]
    # resultMap["fieldIds"]=[]

    # we can do fuzzy or strict mathcing 
    
    # res = api.keywords_search(keywords, max_results=10)
    res = api.fuzzy_keywords_match(keywords, max_results=10)
    for el in res:
        
        nid,db_name,source_name,field_name,score = el
        
        #=============
        # to get the contents
        content = api.getContentValuesOfIndex(nid)
        #=============
        if debug == 1:
            print("Source: %s, Field: %s"%(source_name, field_name))
        # print("\n\n======================== Contents \n")
        
        # print(content)
        if nid not in resultMap:
            resultMap[nid] = {}
            resultMap[nid]["source_name"] = source_name
            resultMap[nid]["field_name"] = field_name
            resultMap[nid]["score"] = score
            resultMap[nid]["nid"] = nid

    
    
    # print("\n\n======================== Entities \n")
    # fields, entities = api.get_all_fields_entities()
    # print(fields)
    # print(entities)
    # print("\n\n======================== SCHEMA \n")
    #TODO: instead of split each keyword and search in the schema, I should search with ontology
    for key in keywords:
        keySplit = key.split(' ')
        for k in keySplit:
            res2 = api.schema_name_search(k, max_results=10)
            for el in res2:
                # print(str(el))
                nid,db_name,source_name,field_name,score = el
                if nid not in resultMap:
                    resultMap[nid] = {}
                    resultMap[nid]["source_name"] = source_name
                    resultMap[nid]["field_name"] = field_name
                    resultMap[nid]["score"] = score
                    resultMap[nid]["nid"] = nid

    if debug == 1:
        print("\n================================\nFinally:\n\n")
        # print (resultMap)
        print("\n\nElements:\n")
        for i in resultMap:
            # print(i)
            print(resultMap[i])
            print("\n")


    # -----------
    # create the snapshot of the graph
    # -----------
    # snapshotOfGraph(api, resultMap, res)

    return resultMap
	

if __name__== "__main__":

    #if argumentPassing == 1:
    if len(sys.argv) > 1:
        keywords = str(sys.argv[1])
        dataDicoveryDemo_main(keywords)
    else:      
        main()
  

# python run_dod.py --model_path 'mytest/testmodel1/' --list_attributes '' --list_values 'purdue;'

