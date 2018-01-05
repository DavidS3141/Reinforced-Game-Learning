import tables
import numpy as np

# Step 1: Create an extendable EArray storage

filename = 'outarray.h5'
ROW_SIZE = 3
NUM_COLUMNS = 10

f = tables.open_file(filename, mode='w')
atom = tables.Float64Atom()

array_c = f.create_earray(f.root, 'game_data', atom, (0, ROW_SIZE))

for idx in range(NUM_COLUMNS):
    x = np.random.rand(2, ROW_SIZE)
    array_c.append(x)
f.close()

# Step 2: Append rows to an existing dataset (if needed)

f = tables.open_file(filename, mode='a')
f.root.game_data.append(x)
f.close()

# Step 3: Read back a subset of the data

f = tables.open_file(filename, mode='r')
print(f.root.game_data[:,:]) # e.g. read from disk only this part of the dataset

nparr = np.array(f.root.game_data)
print(nparr.shape)
