import numpy as np

array = [[
            [1,1,2,3,4,5,4,5,2,5,8,4],
            [1,2,4,6,6,9,4,2,5,2,5,1],
            [2,2,1,5,8,8,5,1,5,5,4,5],
        ]]

array = np.array(array)        

exclude_items = [1,2]
allindices_array = []

for i in range(array.shape[1]):
    for j in range(array.shape[2]):
        allindices_array.append([i,j])


for excluded_item in exclude_items:
    exclude_indices = np.argwhere(array==excluded_item)     #check which pixel is labeled as the excluded label
    exclude_indices = exclude_indices[:,[1,2]]              #convert the excluded indices in (x,y) format                             
    included_indices = []                                   #array which stores the indices of the allowed labels                
    
    #prepare the included index list 
    for x in allindices_array:      
        dist = []                                                                                        
        for y in exclude_indices:
            dist.append(np.linalg.norm(x-y))                #check if the indice is equal to the one in the excluded index list
            states = [0]
            mask = np.in1d(dist, states)
        if np.any(mask == True):                                            
            pass
        else:
            included_indices.append(x)  
    
    # check the nearest neighbour for each excluded index and fill in the array with the nearest neighbour
    for x in exclude_indices:
        dist = []                                            #store the distances   
        for y in included_indices:
            dist.append(np.linalg.norm(x-y))                 
        min_index_temp = np.argmin(dist)                       
        min_index = included_indices[min_index_temp]              # find the index of the closest pixel  
        array[0,x[0],x[1]] = array[0, min_index[0], min_index[1]]

print(array)