#process for latex
#given a results folder, make it not crap by regularizing the names/values


import sys
import os
import pandas as pd
import re


if len(sys.argv) < 2:
 raise Exception("Please provide input folder")

if len(sys.argv) < 3:
	raise Exception("Please provide output folder")


base = os.getcwd()

folder = base + "/" + sys.argv[1]

output_folder = base + "/" + sys.argv[2]

dir_list = os.listdir(folder)
dir_list.sort()

print("Reading {} files from {}".format(len(dir_list),  folder))


filepaths = [(folder + "/" + f, f) for f in dir_list if os.path.isfile(folder + "/" + f)]

print("After pruning {} files remain.".format(len(filepaths)))


output_dataframes = {}


graph_data = {}


for file, name in filepaths:



	res = None

	temp = re.search(r'[a-z]', name, re.I)

	if temp is not None:
		res = temp.start()

	clipped_name = str(name[res:])

	print("Opening {} - local {}".format(file, clipped_name))


	dataframe = []

	if "oom" in clipped_name:
		dataframe = pd.read_csv(file, header=1).fillna(0).transpose()
	else:
		dataframe = pd.read_csv(file).fillna(0).transpose()

	#replace headers

	new_header = dataframe.iloc[0]
	dataframe = dataframe[1:]

	dataframe.columns = new_header

	dataframe = dataframe.replace('',0)

	dataframe.rename(columns=lambda x: x.strip(), inplace=True)

	print(dataframe)


	for column in dataframe:

		# print("column {}".format(column))
		# print(dataframe[column].astype(str).str)
		dataframe.loc[dataframe[column].astype(str).str.contains("Ran longer"), column] = "0"

		#dataframe[dataframe.str.contains()] = 0
		#print(dataframe)

		#trouble children that needs special tuning

		#1 frag - top bar spits out increments of 4,8,12,16 but it is really powers of 2 - 4,8,16,32...
		#also contains - "Ran longer than ..." - this needs to be nan'ed out

		#2 perf-mixed alloc + free - need to clip 16- from headers. Lower range does not change so who needs it.

		#3 graphs - need to be split per-allocator


	#report median performance - these cases have many rows which are neat but not needed for the graphing.
	if "perf" in clipped_name or "scale" in clipped_name:

		print("Clipping non-median columns")
		dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='std_dev')))]
		dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='max')))]
		dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='min')))]
		dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='mean')))]

		for column in dataframe:

			if "median" in column:

				new_column = column.split("- median")[0]
				#print("{} -> {}".format(column, new_column))

				dataframe.rename({column: new_column}, axis=1, inplace=True)


	if "frag" in clipped_name:
		print("Special case for fragmentation test {}".format(clipped_name))

		alloc_range = clipped_name.split("_")[-1].split(".")[0].split("-")

		alloc_total = int(clipped_name.split("_")[-2])

		lower = int(alloc_range[0])
		upper = int(alloc_range[1])

		alloc_range_list = []

		while lower <= upper:

			alloc_range_list.append(lower)
			lower = lower*2


		#print("alloc_range {}".format(alloc_range_list))


		dataframe = dataframe.head(len(alloc_range_list))


		range_list = [f*alloc_total for f in alloc_range_list]


		#print(dataframe)


		dataframe.index = alloc_range_list

		dataframe["ActualSize - range"] = range_list
		dataframe["ActualSize - static range"] = range_list

		#dataframe.set_index(alloc_range_list)

		#idx = dataframe.columns.get_loc("Bytes")

		#print("Column idx: {}".format(idx))

		#print(dataframe)



	if "graph" in clipped_name:
		print("Special case for graph processing {}".format(clipped_name))


		operation_list = clipped_name.split(".")[0].split("_")

		operation = ""

		if len(operation_list) == 2:
			operation = operation_list[-1]
		elif len(operation_list) == 3:
			operation = operation_list[-1]
		else:
			print(len(operation_list))
			operation = operation_list[-2] + "_" + operation_list[-1]

		#.split("_")[-1]

		for column in dataframe:
			print(column)


		gfilenames = [str(i) for i in dataframe.index]
		print (gfilenames)

		for index, row in dataframe.iterrows():

			#print(index)

			output = ""

			for column in dataframe:


				lookup_key = column + "-" + index.split(".mtx")[0]

				print("Key {} for op {}".format(lookup_key, operation))

				if lookup_key not in graph_data:
					graph_data[lookup_key] = []


				graph_data[lookup_key].append((operation, row[column]))




		#continue - don't add to dataframe list.
		continue

	if "perf_mixed" in clipped_name:


		#print("Special case for perf-mixed {}".format(clipped_name))

		alloc_range = clipped_name.split("_")[-1].split(".")[0].split("-")

		alloc_total = int(clipped_name.split("_")[-2])

		lower = int(alloc_range[0])
		upper = int(alloc_range[1])

		alloc_range_list = []

		while lower <= upper:

			alloc_range_list.append(lower)
			lower = lower*2


		dataframe.index = alloc_range_list

		#dataframe = dataframe.index.rename("Bytes")

		#dataframe.index.name = "Bytes"

		#dataframe.rename({"Byte-Range": "Bytes"}, axis=1, inplace=True)

		#print(dataframe)


	dataframe.reset_index(drop=False)

	#print("Final DF view")
	#print(dataframe)

	output_dataframes[clipped_name] = dataframe



print("Processing Done")

isExist = os.path.exists(output_folder)

if not isExist:
	os.mkdir(output_folder)


#output graphs.
for key in graph_data:

	output_filename = output_folder + "/" + key

	with open(output_filename + ".csv", "w") as outputfile:

		#print(key, graph_data[key])
		print("op perf\n")

		for op,val in graph_data[key]:
			#print(op, val)

			output_op = ""

			split_string = op.split("_");

			if len(split_string) != 1:

				output_op = split_string[0] + split_string[1].capitalize()

			else:
				output_op = op

			outputfile.write("{} {}\n".format(output_op, val))



for key in output_dataframes:
	

	output_filename = output_folder + "/" + key
	print("Writing {}".format(output_filename))


	my_index_label = "Bytes"

	if "scale" in key or "synth" in key:
		my_index_label = "Threads"
	if "reg" in key or "oom" in key:
		my_index_label = "Approach"
	output_dataframes[key].to_csv(output_filename, index=True, index_label=my_index_label)  





	#print(dataframe)


