def count_pair(indices):
    counts = defaultdict(int)
    for byte_string, count in indices.items():
        for index1, index2 in zip(byte_string,byte_string[1:]):
            counts[(index1,index2)]+=count
    # print("one chunk complete")    
    return counts

def merge(indices, pair, index, total_counts):
    new_indicies = Counter()
    count_deltas = defaultdict(int)
    A , B = pair
    for byte_string, count in indices.items():
        i = 0
        new_sequence = []
        
        while i < len(byte_string):
            
            if i < len(byte_string) - 1 and (byte_string[i], byte_string[i+1]) == pair:
                if i > 0 :
                    old_pair = (byte_string[i-1], A)
                    count_deltas[old_pair] -= count
                if i + 2 < len(byte_string):
                    old_pair = (B, byte_string[i+2])
                    count_deltas[old_pair] -= count
                    

                # Track added pairs
                if i > 0:
                    new_pair = (byte_string[i-1], index)
                    count_deltas[new_pair] += count
                
                if i + 2 < len(byte_string):  # new pair after
                    new_pair = (index, byte_string[i+2])
                    count_deltas[new_pair] += count
                new_sequence.append(index)
                i+=2
            else:
                new_sequence.append(byte_string[i])  
                i +=1
        new_indicies[tuple(new_sequence)]+=count
    for k , delta in count_deltas.items():
            total_counts[k] +=delta
            if total_counts[k]<= 0:
                del total_counts[k]
                
    return new_indicies