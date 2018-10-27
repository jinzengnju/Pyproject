import os
import json
# dict={"天才":1,"南京":2,"大学":3}
#
# f_write=open("vocab.dict",'w')
# json.dump(dict,f_write,ensure_ascii=False)
f_read=open("vocab.dict",'r')
dict=json.load(f_read)
print(dict)