#from partition_while.core import partition_while

from partition_while import PartitionWhile#, partition_while_longest, partition_while_next


print(PartitionWhile([1,2,3,4,5,6,7,8,9,10], lambda x: sum(x) <= 10))

print(PartitionWhile([-5,8,1,2,6,-20,8,9,-5,7,3], lambda x: sum(x) <= 10))

print(PartitionWhile([-5,8,1,2,6,-20,8,9,-5,7,3], lambda x: sum(x) <= 10,shortest=False))