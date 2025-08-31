from collections import Counter, defaultdict
import os
import regex as re
from multiprocessing import Process, Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO
from time import perf_counter
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)
SPECIAL_SPLIT_RE = None
PATH = "TinyStories/TinyStories-valid.txt"
def train_bpe(input_path:str, vocab_size:int, special_tokens: list[str]):
    
    ## Usage
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, cpu_count()-2, "<|endoftext|>".encode("utf-8"))
 
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        vocab:dict[int,bytes] = {x:bytes([x]) for x in range(256)}
        merges:list[tuple[bytes,bytes]] = []
        token_to_id:list[tuple[int,bytes]] = {}
        special_token_index = [] 
        for index, token in enumerate(special_tokens):
            token_to_id[index +(len(vocab))] = token.encode("utf-8")
            special_token_index += [index +(len(vocab))]
        vocab.update(token_to_id)    
        special_token_bytes_to_id = {token.encode("utf-8"): 256 + index for index, token in enumerate(special_tokens)}
        # print(vocab)
        # print(special_token_index)
        # print(special_token_bytes_to_id)
        vocab_length = len(vocab)
        # print("start pretokenizing")
        bytes_to_index = {v: k for k, v in vocab.items()}
        escaped = [re.escape(t) for t in special_tokens]
        escaped.sort(key=len, reverse=True) 
        
        global SPECIAL_SPLIT_RE
        SPECIAL_SPLIT_RE = re.compile("|".join(escaped)) if escaped else None  # compile once
        with Pool(processes=max(1, min(cpu_count()-2, 4))) as pool:
            chunks =[(start, end, input_path,special_token_bytes_to_id) for start, end in zip(boundaries[:-1], boundaries[1:])]
            results = pool.starmap(process_chunks, chunks)
            # After pool.starmap returns results
            indices:dict[tuple[bytes],int] = defaultdict(int)
        
            for result in results:
                for token_tuple, count in result.items():
                    indices[token_tuple] += count
            
            # print(indices)
            # print("done with chunking")
        # start_time = perf_counter()
        # print("starting timer")
        # loop_counter = 0
        total_counts = count_pair(indices, special_token_index)
        for i in range( vocab_size - vocab_length):
                # loop_counter +=1
                 
                
                # print("pairs counted")
                if not total_counts :
                    break
                # print(total_counts)

                pair = max(
                    total_counts,
                    key=lambda p: (total_counts[p], -min(p[0], p[1]), p)
                )
             
                #  print(pair)
                # print("got max pair")
                index1, index2 = pair
                merges.append((vocab[index1],vocab[index2]))
                vocab[vocab_length + i] = vocab[index1] + vocab[index2]
                indices = merge(indices, pair, vocab_length + i, total_counts, special_token_index)
                # print("indices updated")
                # print(loop_counter)
        # end_time = perf_counter()
        # elasped_time = end_time-start_time
        # print("Timer ended: ", elasped_time)
        # # print(vocab)
        # print(merges)
    return vocab, merges

def encode(text:str):
    return 0

def process_chunks(start, end, input_str, special_token_bytes_to_id):
    with open(input_str, "rb") as f:   
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore") 
        piece_counter:dict[tuple[int],int] = defaultdict(int)
        if SPECIAL_SPLIT_RE is not None:
            pieces = SPECIAL_SPLIT_RE.split(chunk)
            matches = list(SPECIAL_SPLIT_RE.finditer(chunk))
            
        else:
            pieces = [chunk]
            matches = []
        for i, piece in enumerate(pieces):
            if piece:
                for m in PAT_RE.finditer(piece):
                    piece_tokens = m.group(0).encode('utf-8')
                    piece_counter[tuple(piece_tokens)] += 1
            # Insert the special token as a tuple of its vocab id
            if SPECIAL_SPLIT_RE is not None and i < len(pieces) - 1:
                special_token_bytes = matches[i].group(0).encode('utf-8')
                special_token_id = special_token_bytes_to_id[special_token_bytes]
                piece_counter[(special_token_id,)] += 1
        return piece_counter

def count_pair(indices, special_token_index):
    counts = defaultdict(int)
    for byte_string, count in indices.items():
        for index1, index2 in zip(byte_string,byte_string[1:]):
            if index1 in special_token_index or index2 in special_token_index:
                continue
            counts[(index1,index2)]+=count
    # print("one chunk complete")    
    return counts
    
def merge(indices, pair, index, total_counts, special_token_index):
    new_indicies = Counter()
    count_deltas = defaultdict(int)
    A , B = pair
    
    for byte_string, count in indices.items():
        i = 0
        new_sequence = []
        
        while i < len(byte_string):
            
            if (i < len(byte_string) - 1
                and (byte_string[i], byte_string[i + 1]) == pair
                and byte_string[i] not in special_token_index
                and byte_string[i + 1] not in special_token_index):
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


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at + len(split_special_token)
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
# if __name__ == "__main__":                               
#     uber_token = ["<|endoftext|>"]

#     train_bpe(PATH, 10000, uber_token)
# seqs = Counter({(1,2,3,2,3): 1})
# print(seqs)
# merged = merge(seqs, (2,3), 300)
# print(merged)  # Expect {(1,300,300): 1}
