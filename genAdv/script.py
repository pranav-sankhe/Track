import numpy as np


a = [[
		[1,0,1,0,0,1,0,1,0,1],
		[0,0,1,1,1,1,1,0,1,0],
		[0,1,1,0,0,1,0,0,0,0]
    ]]

# print(np.random.permutation(b[0]))
	
b = [[
		[1,2,5,4,8,2,4,8,5,1],
		[1,2,6,5,4,7,8,9,1,3],
		[5,3,2,1,4,7,8,9,6,3]
    ]]

a = a[0]
b = b[0]

c = np.subtract(a,b)
print (c)

indices = np.where(c == 0) 
i = indices[0]
j = indices[1]
print(i,j)
print (c[i[:],j[:]])


# a = [[
# 		[
# 		 [1,2,3,5,4,8,4,2,5,4], 
# 		 [2,5,4,7,8,9,6,3,5,4],
# 		 [2,1,4,7,8,5,6,9,3,4]
# 		], 

# 		[
# 		 [0,1,1,5,8,2,5,2,2,5], 
# 		 [2,1,1,2,5,8,8,2,5,7],
# 		 [1,2,5,4,7,8,9,6,3,2]
# 		], 		

# 	]]

# a = np.array(a)
	
# arrayshape = a.shape
# width = arrayshape[2]
# height = arrayshape[3]

# print (a)
# print (b)

# output = np.empty([width, height])

# b = b[0]

# for i in range(width):
# 	for j in range(height):
# 		index = b[i][j]
# 		output[i][j] =  a[0,index,i,j]

# print (output)



# # for i in range(arrayshape[1]):
# # 	print(a[0][0][i,0])

