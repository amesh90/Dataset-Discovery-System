import sys
sys.path.insert(0, '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master')

debug=1
argumentPassing=1

import sys
#from main import init_system
from collections import namedtuple
#from modelstore.elasticstore import KWType
from api.apiutils import Relation
# from algebra import API, DRS
#from mock import MagicMock, patch
#from ddapi import API
#from modelstore.elasticstore import StoreHandler
from knowledgerepr.fieldnetwork import deserialize_network
from knowledgerepr.fieldnetwork import serialize_network_to_csv
from knowledgerepr.fieldnetwork import serialize_network_to_json
from knowledgerepr.fieldnetwork import serialize_network_to_onecsv
from knowledgerepr.fieldnetwork import serialize_network_to_onecsv_detailed

from collections import defaultdict
from collections import OrderedDict
import itertools
from DoD.utils import FilterType
from inputoutput import inputoutput as io

from modelstore.elasticstore import StoreHandler
# from DoD import DoD
# import DoD

# import run_dod
 

def main():
	
    #========================================== 
    # initialization
    #========================================== 
    path = '/Users/princess/Documents/PhD/thesis/code/aurum-datadiscovery-master/mytest/testmodel4/'
    store = StoreHandler()

    # ddapi:
    network = deserialize_network(path)
    
    path_schsim = path + "/schema_sim_index.pkl"
    schema_sim_index = io.deserialize_object(path_schsim)
    
    path_cntsim = path + "/content_sim_index.pkl"
    content_sim_index = io.deserialize_object(path_cntsim)
    
    print ("\n\n============\ncontent_sim_index:\n")
    print (content_sim_index)
    print ("\n\n============\nmh_signatures:\n")

    # mh_signatures = store.get_all_mh_text_signatures()
    # with open("/Users/princess/Documents/PhD/thesis/code/sigma.js-master/examples/data/" + "nodesmhsig", 'w') as f:
    #     for k, minhash in mh_signatures:
    #         print(k)
    #         print(minhash)
    #         f.write(str(k)+" " + ' '.join(['{}'.format((r)) for r in minhash] ) + "\n")
   
    # with open("/Users/princess/Documents/PhD/thesis/code/sigma.js-master/examples/data/" + "nodesmhsig", 'r') as f:
    #     s = f.read()
    #     for r in s:
    #         print(s) 
    #         print("\n\n")
       


    # serialize_network_to_onecsv_detailed(network, content_sim_index, schema_sim_index, '/Users/princess/Documents/PhD/thesis/code/sigma.js-master/examples/data/')
    # for r in content_sim_index:
    #     print("\nr:\n")
    #     print (r)
    #     print("\na:\n")
    #     for a in r:
    #         print(a)
    # print ("\n\n============\nschema_sim_index:\n")
    # print (schema_sim_index)
    # print("\n\n====================\nVisulaize Graph\n=====================\n\n")
    # network._visualize_graph()
    #serialize_network_to_csv(network, '/Users/princess/Documents/PhD/thesis/code/')
    # serialize_network_to_json(network, '/Users/princess/Documents/PhD/thesis/code/sigma.js-master/examples/data/')
    #serialize_network_to_onecsv(network, '/Users/princess/Documents/PhD/thesis/code/sigma.js-master/examples/data/')

    return


if __name__== "__main__":

    main()
  

# python run_dod.py --model_path 'mytest/testmodel1/' --list_attributes '' --list_values 'purdue;'

