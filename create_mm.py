
#given a .txt file that is a matrix, read it into a matrixMarket format.


import sys
import progressbar
import scipy
from scipy import sparse
import numpy as np
from scipy.io import mmwrite
import subprocess

def line_count(filename):
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

if len(sys.argv) != 2:
 raise Exception("Please provide a filename")

filename = sys.argv[1]

print("Reading {}".format(filename))

num_lines = line_count(filename)

print("{} has {} lines".format(filename, num_lines))

max_id = 0
#output_lines = []

seen_dict = {}

count = 0;

rows = []
cols = []

with open(filename) as file:

	with progressbar.ProgressBar(max_value=num_lines) as bar:

		for line in file:

			if line[0] == '#':
				count += 1
				continue

			#else, add to lists
			row_str, col_str = line.split()

			row_int = int(row_str)
			col_int = int(col_str)

			min_val = min(row_int, col_int)
			max_val = max(row_int, col_int)

			max_id = max(max_id, max_val)

			combined_tuple = (min_val, max_val)
			if combined_tuple in seen_dict:
				count+=1
				continue;

			rows.append(row_int)
			cols.append(col_int)

			#output_lines.append("{} {}\n".format(row_int, col_int))

			seen_dict[combined_tuple] = 1

			count +=1
			if count % 1000000 == 0:
				bar.update(count)

	#rows = []
	#cols = []

#output = zip(rows, cols)



	# for i in progressbar.progressbar(range(len(Lines))):

	# 	row_str, col_str = Lines[i].split()

	# 	row_int = int(row_str)
	# 	col_int = int(col_str)

	# 	min_val = min(row_int, col_int)
	# 	max_val = max(row_int, col_int)

	# 	max_id = max(max_id, max_val)

	# 	combined_tuple = (min_val, max_val)
	# 	if combined_tuple in seen_dict:
	# 		continue;

	# 	output_lines.append("{} {}\n".format(row_int, col_int))


max_id = max_id + 1



print("Max node id is {}".format(max_id))

count = 0
with open(filename+".mtx", "w") as output:

	output.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
	data_line = "{} {} {}\n".format(max_id, max_id, len(rows))
	output.write(data_line)

	with progressbar.ProgressBar(max_value=len(rows)) as bar:

		max_count = len(rows)
		while (count < max_count):

			output.write("{} {}\n".format(rows[count], cols[count]))
			count +=1

			if count % 1000000 == 0:
				bar.update(count)


# output_lines.append("%%MatrixMarket matrix coordinate pattern symmetric\n")
# output_lines.append("{} {} {}\n".format(max_id, max_id, len(Lines)))

# for i in progressbar.progressbar(range(len(Lines))):

# 	min_val = min(rows[i], cols[i])
# 	max_val = max(rows[i], cols[i])

# 	combined_tuple = (min_val, max_val)
# 	if combined_tuple in seen_dict:
# 		continue;


#mat = sparse.csr_matrix((np.ones_like(cols), (rows,cols)), shape=(max_id,max_id))

#mat[rows, cols] = mat[cols, rows]
#mat = mat.maximum(mat.transpose())

# print("Done...Writing")

# with open(filename+".mtx", "w") as output:

# 	output.writelines(["%%MatrixMarket matrix coordinate pattern symmetric\n"])
# 	data_line = "{} {} {}\n".format(max_id, max_id, len(output_lines))
# 	output.writelines([data_line])
# 	output.writelines(output_lines)



	#scipy.io.mmwrite(filename + ".mtx", mat)
