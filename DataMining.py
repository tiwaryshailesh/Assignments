import pandas as pd
import math
import glob

def entropy(probs):
    import math
    return sum( [-prob*math.log(prob, 2) for prob in probs] )

def entropy_of_list(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)*1.0
    probs = [x / num_instances for x in cnt.values()]
    return entropy(probs)

def information_gain(df, split_attribute_name, target_attribute_name, trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index) * 1.0
    df_agg_ent = df_split.agg({target_attribute_name : [entropy_of_list, lambda x: len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy', 'PropObservations']
    if trace:
        print (df_agg_ent)
    new_entropy = sum( df_agg_ent['Entropy'] * df_agg_ent['PropObservations'] )
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy


def id3(df, target_attribute_name, attribute_names, default_class=None):
	from collections import Counter
	cnt = Counter(x for x in df[target_attribute_name])
	if len(cnt) == 1:
		return list(cnt.keys())[0]
	elif df.empty or (not attribute_names):
		return default_class
	else:
		#print(cnt)
		index_of_max = list(cnt.values()).index(max(cnt.values()))
		default_class = list(cnt.keys())[index_of_max]
		gainz = [information_gain(df, attr, target_attribute_name) for attr in attribute_names]
		index_of_max = gainz.index(max(gainz))
		best_attr = attribute_names[index_of_max]
		tree = {best_attr:{}}
		remaining_attribute_names = [i for i in attribute_names if i != best_attr]
		for attr_val, data_subset in df.groupby(best_attr):
			subtree = id3(data_subset,
			                target_attribute_name,
			                remaining_attribute_names,
			                default_class)
			tree[best_attr][attr_val] = subtree
		return tree



###############################################################################

files=f=[str(i)+".csv" for i in range(1,57)]

def classify(instance, tree, default=None):
    attribute = list(tree.keys())[0]
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result, dict): # this is a tree, delve deeper
            return classify(instance, result)
        else:
            return result # this is a label
    else:
        return default


def assignmentpart1():
	writer = pd.ExcelWriter('test.xlsx',engine='xlsxwriter')

	for file in files:
		print("[+] " + file)
		df = pd.read_csv(file)
		cols=["F"+str(i) for i in range(df.shape[1]-1)]+["Class"]
		df.columns=cols
		temp=[]
		for i in df.columns[:-1]:
			temp.append([i, str(information_gain(df, i , 'Class') )])

		tempdf=pd.DataFrame(temp,columns=["Feature","Info Gain"])
		tempdf.to_excel(writer,sheet_name=file, index=False)

	writer.save()

def assignmentpart2(file="45.csv"):
	df = pd.read_csv(file)
	cols=["F"+str(i) for i in range(df.shape[1]-1)]+["Class"]
	df.columns=cols
	attribute_names = list(df.columns)
	attribute_names.remove('Class')
	from pprint import pprint
	tree = id3(df, 'Class', attribute_names)
	pprint(tree)
	df['predicted'] = df.apply(classify, axis=1, args=(tree,14) )
	print('Accuracy is ' + str( sum(df['Class']==df['predicted'] ) / (1.0*len(df.index)) ))
	print(df[['Class', 'predicted']])


assignmentpart1()
assignmentpart2()
